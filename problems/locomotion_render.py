"""MuJoCo renderings of Walker2D and Hopper windows.

The locomotion problems are trained and scored entirely in state space, so
the figures in ``problems/locomotion.py`` are height traces. Those show that
the roof holds, but not what holding it *looks like*: a window that ducks
under a ceiling and a window that simply walks lower are the same curve until
you draw the robot. This module turns a generated window back into MuJoCo
poses and renders them as a gait tile.

The reconstruction is exact except for one coordinate. A D4RL observation is
``qpos[1:] + qvel``, dropping the root x-position because the reward is
translation-invariant, so the x of each frame is recovered by integrating the
root x-velocity across the window at the control timestep. Nothing else is
inferred: every joint angle is read straight out of the observation.

Rendering needs the ``render`` dependency group (``uv sync --group render``),
which brings in ``mujoco`` and the ``gymnasium`` copies of the two model XMLs
-- the same files the D4RL demonstrations were collected on.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

from problems.locomotion_spec import LocomotionSpec

# One D4RL transition is one env step: frame_skip=4 at the XML's 0.002 s
# timestep, for both environments.
CONTROL_DT = 0.008

# The model file each spec is rendered with, inside gymnasium's asset
# directory. These are the ``-v4`` assets, which differ from the ``-v2`` ones
# the data was collected on only in foot friction -- nothing visible.
ASSETS = {"walker2d": "walker2d.xml", "hopper": "hopper.xml"}

# Default framing, in metres of world height covered by the image. Both
# robots stand about 1.25 m tall, so this leaves room above the roof.
VIEW_EXTENT = 2.1
CAMERA_Z = 0.95


def _require(module: str):
    """Import an optional render dependency, or explain how to get it."""
    try:
        return __import__(module)
    except ImportError as err:
        raise ImportError(
            f"rendering locomotion windows needs {module!r}; install the "
            f"render extras with `uv sync --group render`"
        ) from err


def asset_path(spec: LocomotionSpec) -> Path:
    """Path to the MuJoCo XML for one environment."""
    gymnasium = _require("gymnasium")
    try:
        name = ASSETS[spec.name]
    except KeyError:
        raise KeyError(
            f"no render asset registered for {spec.name!r}"
        ) from None
    return Path(gymnasium.__file__).parent / "envs" / "mujoco" / "assets" / name


def num_coordinates(spec: LocomotionSpec) -> int:
    """Length of ``qpos`` for one environment.

    The observation is ``qpos[1:] + qvel`` and these models have ``nq == nv``,
    so the width of the observation pins the size of the configuration:
    ``obs_dim = 2 * nq - 1``. Walker2D gives 9, Hopper 6.
    """
    return (spec.obs_dim + 1) // 2


def x_velocity_index(spec: LocomotionSpec) -> int:
    """Column of the root x-velocity in a sample, actions included.

    It is the first entry of the ``qvel`` block, which starts right after the
    ``nq - 1`` positions.
    """
    return spec.action_dim + num_coordinates(spec) - 1


def window_to_qpos(
    spec: LocomotionSpec, window: np.ndarray, dt: float = CONTROL_DT
) -> np.ndarray:
    """Recover MuJoCo configurations from one generated window.

    Args:
        spec: Which environment the window came from.
        window: One window, shape ``(horizon, transition_dim)``.
        dt: Control timestep, used to integrate the root x-velocity.

    Returns:
        Configurations of shape ``(horizon, nq)``, with the first frame
        placed at ``x = 0``.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError(f"expected one window, got shape {window.shape}")

    nq = num_coordinates(spec)
    obs = window[:, spec.action_dim:]
    qpos = np.zeros((len(window), nq))
    qpos[:, 1:] = obs[:, :nq - 1]
    # Forward Euler on the root velocity: the first frame is the origin, and
    # each later frame advances by the velocity reported at the frame before.
    vx = window[:, x_velocity_index(spec)]
    qpos[1:, 0] = np.cumsum(vx[:-1]) * dt
    return qpos


class LocomotionRenderer:
    """An offscreen MuJoCo scene, reused for every frame of a figure.

    A renderer owns a GL context. Building one per frame exhausts the EGL
    contexts on the first figure and every render after that silently comes
    back blank, so a whole figure goes through one instance.

    The camera is orthographic and looks along the world y-axis, which is the
    plane both robots move in. That makes the image an exact scaled copy of
    the (x, z) plane -- no foreshortening -- so :meth:`row_of_z` turns a
    world height into an image row exactly, and a figure can annotate the
    render in the same units its other panels use.

    Frames come back RGBA with everything but the robot transparent, so a
    render drops onto a figure background instead of carrying a slab of sky
    with it. The cutout is a segmentation pass rather than a colour key, so a
    shadowed limb is never mistaken for background. The floor goes with the
    sky: seen exactly edge-on it is a zero-thickness line anyway, and a
    ground line drawn at ``row_of_z(0)`` reads better.
    """

    def __init__(
        self,
        spec: LocomotionSpec,
        width: int = 260,
        height: int = 380,
        extent: float = VIEW_EXTENT,
        camera_z: float = CAMERA_Z,
    ):
        """Build the scene for one environment.

        Args:
            spec: Which environment to render.
            width: Image width in pixels.
            height: Image height in pixels.
            extent: World height covered by the image, in metres.
            camera_z: World height at the vertical centre of the image.
        """
        # EGL renders headless; a figure built over ssh or in CI would
        # otherwise fail looking for a display.
        os.environ.setdefault("MUJOCO_GL", "egl")
        mujoco = _require("mujoco")

        self.spec = spec
        self.width, self.height = width, height
        self.extent, self.camera_z = extent, camera_z

        self.model = mujoco.MjModel.from_xml_path(str(asset_path(spec)))
        self.data = mujoco.MjData(self.model)
        # In orthographic mode ``fovy`` is the vertical extent of the view in
        # metres rather than an angle.
        self.model.vis.global_.orthographic = 1
        self.model.vis.global_.fovy = extent
        self.renderer = mujoco.Renderer(self.model, height, width)

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.azimuth, self.camera.elevation = 90.0, 0.0
        # Only the sign matters under an orthographic projection; this puts
        # the whole scene in front of the near plane.
        self.camera.distance = 5.0

        # Geoms dropped from the cutout. The floor is the only one either
        # model names, but the lookup stays tolerant so a scene without it
        # still renders.
        self.excluded_geoms = {
            gid for gid in (
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
                for name in ("floor",)
            ) if gid >= 0
        }

    @property
    def metres_per_pixel(self) -> float:
        """Scale of the rendered image, equal on both axes."""
        return self.extent / self.height

    def row_of_z(self, z: float) -> float:
        """Image row a world height falls on.

        Exact, because the projection is orthographic: the camera target sits
        at the centre of the image and the scale is uniform.
        """
        offset = (z - self.camera_z) / self.metres_per_pixel
        return (self.height - 1) / 2 - offset

    def col_of_x(self, x: float, center_x: float) -> float:
        """Image column a world x falls on, for a tile centred on
        ``center_x``."""
        return (self.width - 1) / 2 + (x - center_x) / self.metres_per_pixel

    def frame(self, qpos: np.ndarray, center_x: Optional[float] = None):
        """Render one configuration.

        Args:
            qpos: Configuration of length ``nq``.
            center_x: World x at the horizontal centre of the image. Defaults
                to the root x of ``qpos``, which keeps the robot in the middle
                of every tile.

        Returns:
            A ``(height, width, 4)`` uint8 RGBA image, transparent
            everywhere but the robot.
        """
        mujoco = _require("mujoco")
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"expected qpos of length {self.model.nq}, got {qpos.shape}"
            )

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        # Positions only: mj_forward would also run the dynamics, which is
        # both wasted work and a chance for a generated pose to blow up.
        mujoco.mj_kinematics(self.model, self.data)

        self.camera.lookat[:] = (
            qpos[0] if center_x is None else center_x, 0.0, self.camera_z
        )
        self.renderer.update_scene(self.data, self.camera)
        rgb = self.renderer.render()

        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, self.camera)
        segments = self.renderer.render()
        self.renderer.disable_segmentation_rendering()

        # Segmentation gives (object id, object type) per pixel, with -1 for
        # background.
        ids, types = segments[..., 0], segments[..., 1]
        robot = (ids >= 0) & (types == mujoco.mjtObj.mjOBJ_GEOM)
        for gid in self.excluded_geoms:
            robot &= ids != gid

        rgba = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = np.where(robot, 255, 0)
        return rgba

    def window_frame(
        self, window: np.ndarray, step: int = 0, dt: float = CONTROL_DT
    ):
        """Render one timestep of a window.

        Args:
            window: One window, shape ``(horizon, transition_dim)``.
            step: Which timestep to draw.
            dt: Control timestep, for the x-position integration. Irrelevant
                to the pose; it only decides where along the floor the robot
                is drawn, and the frame is centred on it anyway.

        Returns:
            A ``(height, width, 4)`` uint8 RGBA image.
        """
        return self.frame(window_to_qpos(self.spec, window, dt=dt)[step])

    def ground_clearance(self, qpos: np.ndarray) -> float:
        """Height of the lowest point of the robot above the floor.

        Near zero in stance, positive in flight, and slightly negative where
        a capsule dips into the ground -- which the demonstrations do too,
        since a window is a recorded state sequence and not a simulation this
        code re-runs. Used to pick a pose that stands on the floor line
        instead of hovering over it.
        """
        mujoco = _require("mujoco")
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        return min(
            float(self.data.geom_xpos[g][2] - self.model.geom_size[g][0])
            for g in range(self.model.ngeom)
            if g not in self.excluded_geoms
        )

    def close(self) -> None:
        """Release the GL context."""
        renderer, self.renderer = getattr(self, "renderer", None), None
        if renderer is not None:
            renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


__all__ = [
    "ASSETS",
    "CAMERA_Z",
    "CONTROL_DT",
    "VIEW_EXTENT",
    "LocomotionRenderer",
    "asset_path",
    "num_coordinates",
    "window_to_qpos",
    "x_velocity_index",
]
