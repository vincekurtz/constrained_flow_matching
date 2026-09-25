"""MuJoCo renderings of Walker2D and Hopper windows.

D4RL observations are ``qpos[1:] + qvel``, so root x is recovered by
integrating the x-velocity. Requires ``uv sync --group render``.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

from problems.locomotion_spec import LocomotionSpec

CONTROL_DT = 0.008  # frame_skip=4 at a 0.002 s timestep

# Gymnasium -v4 assets; they differ from -v2 only in foot friction.
ASSETS = {"walker2d": "walker2d.xml", "hopper": "hopper.xml"}

VIEW_EXTENT = 2.1  # metres of world height in the image
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
    """Length of ``qpos``, from ``obs_dim = 2 * nq - 1``."""
    return (spec.obs_dim + 1) // 2


def x_velocity_index(spec: LocomotionSpec) -> int:
    """Column of the root x-velocity in a sample."""
    return spec.action_dim + num_coordinates(spec) - 1


def window_to_qpos(
    spec: LocomotionSpec, window: np.ndarray, dt: float = CONTROL_DT
) -> np.ndarray:
    """Recover ``(horizon, nq)`` configurations from one window, starting
    at x = 0."""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError(f"expected one window, got shape {window.shape}")

    nq = num_coordinates(spec)
    obs = window[:, spec.action_dim:]
    qpos = np.zeros((len(window), nq))
    qpos[:, 1:] = obs[:, :nq - 1]
    vx = window[:, x_velocity_index(spec)]
    qpos[1:, 0] = np.cumsum(vx[:-1]) * dt
    return qpos


class LocomotionRenderer:
    """Offscreen orthographic MuJoCo scene, returning RGBA robot cutouts.

    Reuse one instance per figure: creating one per frame exhausts EGL
    contexts and later renders come back blank.
    """

    def __init__(
        self,
        spec: LocomotionSpec,
        width: int = 260,
        height: int = 380,
        extent: float = VIEW_EXTENT,
        camera_z: float = CAMERA_Z,
    ):
        """``extent``: image height in metres; ``camera_z``: world z at the
        image centre."""
        os.environ.setdefault("MUJOCO_GL", "egl")
        mujoco = _require("mujoco")

        self.spec = spec
        self.width, self.height = width, height
        self.extent, self.camera_z = extent, camera_z

        self.model = mujoco.MjModel.from_xml_path(str(asset_path(spec)))
        self.data = mujoco.MjData(self.model)
        # Orthographic fovy is the view height in metres.
        self.model.vis.global_.orthographic = 1
        self.model.vis.global_.fovy = extent
        self.renderer = mujoco.Renderer(self.model, height, width)

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.azimuth, self.camera.elevation = 90.0, 0.0
        # Just keeps the scene in front of the near plane.
        self.camera.distance = 5.0

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
        """Image scale, equal on both axes."""
        return self.extent / self.height

    def row_of_z(self, z: float) -> float:
        """Image row of a world height."""
        offset = (z - self.camera_z) / self.metres_per_pixel
        return (self.height - 1) / 2 - offset

    def col_of_x(self, x: float, center_x: float) -> float:
        """Image column of a world x, for a tile centred on ``center_x``."""
        return (self.width - 1) / 2 + (x - center_x) / self.metres_per_pixel

    def frame(self, qpos: np.ndarray, center_x: Optional[float] = None):
        """Render one configuration as ``(height, width, 4)`` uint8 RGBA.

        ``center_x`` defaults to the root x of ``qpos``.
        """
        mujoco = _require("mujoco")
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"expected qpos of length {self.model.nq}, got {qpos.shape}"
            )

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
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
        """Render timestep ``step`` of a window."""
        return self.frame(window_to_qpos(self.spec, window, dt=dt)[step])

    def ground_clearance(self, qpos: np.ndarray) -> float:
        """Height of the robot's lowest point above the floor."""
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
