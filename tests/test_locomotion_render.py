"""Turning a locomotion window back into MuJoCo poses.

The reconstruction is the one place a figure can lie without looking wrong:
read the observation off by one column and the renderer draws a perfectly
plausible robot in the wrong pose, with the ceiling line still passing
through wherever the code thinks the torso is. These tests pin the layout
against the specs and check that the pixel map the roof line is drawn with is
the one the renderer actually renders.

The pure reconstruction tests need neither MuJoCo nor the D4RL files. The
rendering tests skip unless the ``render`` dependency group is installed.
"""

import numpy as np
import pytest

from problems.locomotion_render import (
    CONTROL_DT,
    num_coordinates,
    window_to_qpos,
    x_velocity_index,
)
from problems.locomotion_spec import HOPPER, WALKER2D

SPEC_LIST = [WALKER2D, HOPPER]
IDS = [spec.name for spec in SPEC_LIST]

# nq for each model, read off its XML: Walker2D has a free-ish root plus six
# leg joints, Hopper the same root plus three.
EXPECTED_NQ = {"walker2d": 9, "hopper": 6}


def make_window(spec, horizon=8, seed=0):
    """A window of arbitrary but distinct entries, in sample layout."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(horizon, spec.transition_dim)).astype(np.float32)


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_num_coordinates_matches_model(spec):
    assert num_coordinates(spec) == EXPECTED_NQ[spec.name]


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_observation_splits_into_qpos_and_qvel(spec):
    """``obs = qpos[1:] + qvel`` has to account for every column."""
    nq = num_coordinates(spec)
    assert (nq - 1) + nq == spec.obs_dim


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_torso_height_is_the_first_position(spec):
    """The spec's z column is ``qpos[1]``, the first entry of the
    observation."""
    assert spec.z_obs_index == 0
    assert spec.z_index == spec.action_dim


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_velocity_columns_are_consistent(spec):
    """The x-velocity heads the qvel block and the z-velocity follows it.

    Both are asserted against the spec's own ``vz_obs_index``, which the
    constraint reads, so a spec edited to point somewhere else fails here
    rather than quietly rendering a different robot than it constrains.
    """
    nq = num_coordinates(spec)
    assert x_velocity_index(spec) == spec.action_dim + nq - 1
    assert spec.vz_obs_index == (nq - 1) + 1


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_window_to_qpos_copies_the_observed_positions(spec):
    window = make_window(spec)
    qpos = window_to_qpos(spec, window)
    nq = num_coordinates(spec)

    assert qpos.shape == (len(window), nq)
    obs = window[:, spec.action_dim:]
    np.testing.assert_allclose(qpos[:, 1:], obs[:, :nq - 1], rtol=0, atol=0)
    # In particular the rendered torso height is the constrained one.
    np.testing.assert_allclose(qpos[:, 1], window[:, spec.z_index])


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_window_to_qpos_integrates_the_root_x(spec):
    """x starts at the origin and advances by the previous frame's
    velocity."""
    window = make_window(spec)
    qpos = window_to_qpos(spec, window, dt=CONTROL_DT)
    vx = window[:, x_velocity_index(spec)]

    assert qpos[0, 0] == 0.0
    np.testing.assert_allclose(
        np.diff(qpos[:, 0]), vx[:-1] * CONTROL_DT, rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_window_to_qpos_rejects_a_batch(spec):
    """Batches are the easy mistake; one window at a time is the contract."""
    batch = np.stack([make_window(spec), make_window(spec, seed=1)])
    with pytest.raises(ValueError, match="one window"):
        window_to_qpos(spec, batch)


# --------------------------------------------------------------------------
# Rendering. Needs `uv sync --group render`.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def renderer_factory():
    """Build ``LocomotionRenderer`` instances, skipping if rendering is
    unavailable.

    A GL context is a machine property, not a code property: a box with the
    packages but no EGL device cannot render, and that is a skip rather than
    a failure.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from problems.locomotion_render import LocomotionRenderer

    built = []

    def make(spec, **kwargs):
        try:
            renderer = LocomotionRenderer(spec, **kwargs)
        except Exception as err:  # pragma: no cover - depends on the host
            pytest.skip(f"no usable GL context: {err}")
        built.append(renderer)
        return renderer

    yield make
    for renderer in built:
        renderer.close()


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_model_agrees_with_the_spec(spec, renderer_factory):
    """The XML's own ``nq`` is what ``num_coordinates`` claims."""
    renderer = renderer_factory(spec)
    assert renderer.model.nq == num_coordinates(spec)
    assert renderer.model.nv == renderer.model.nq


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_frames_are_transparent_cutouts(spec, renderer_factory):
    """A frame is RGBA, with the robot opaque and everything else clear."""
    renderer = renderer_factory(spec, width=120, height=180)
    qpos = np.zeros(num_coordinates(spec))
    qpos[1] = 1.25
    frame = renderer.frame(qpos)

    assert frame.shape == (180, 120, 4)
    assert frame.dtype == np.uint8
    alpha = frame[..., 3]
    assert set(np.unique(alpha)) <= {0, 255}
    # The robot is in shot but does not fill it.
    assert 0.02 < np.mean(alpha > 0) < 0.5


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_row_of_z_locates_the_torso(spec, renderer_factory):
    """The row the roof line is drawn on is where the torso really renders.

    The torso is the topmost geom of both models when they stand upright, so
    its top edge in the image has to sit half a torso above ``row_of_z(z)``.
    The check is what makes an orthographic camera worth the trouble: under
    perspective the offset would depend on where the robot stood.
    """
    renderer = renderer_factory(spec, width=160, height=300)
    half_length = renderer.model.geom("torso_geom").size[1]
    radius = renderer.model.geom("torso_geom").size[0]

    for z in (1.0, 1.3):
        qpos = np.zeros(num_coordinates(spec))
        qpos[1] = z
        frame = renderer.frame(qpos)
        top = int(np.argmax(np.any(frame[..., 3] > 0, axis=1)))
        expected = renderer.row_of_z(z + half_length + radius)
        assert abs(top - expected) <= 2, f"z={z}: top {top} vs {expected}"


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_window_frame_draws_the_requested_step(spec, renderer_factory):
    """Each tile is centred on the robot, so only the pose can differ."""
    renderer = renderer_factory(spec, width=160, height=300)
    nq = num_coordinates(spec)
    window = make_window(spec, horizon=3)
    # Two frames in the same pose and one crouched. The robot travels, so
    # any difference between the first two would be the framing drifting.
    window[:, x_velocity_index(spec)] = 3.0
    window[:, spec.action_dim:spec.action_dim + nq - 1] = 0.0
    window[:, spec.z_index] = 1.25
    window[2, spec.z_index] = 0.95

    frames = [renderer.window_frame(window, step) for step in range(3)]
    np.testing.assert_array_equal(frames[1], frames[0])
    assert not np.array_equal(frames[2], frames[0])


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_ground_clearance_tracks_the_root_height(spec, renderer_factory):
    """Lifting the whole robot lifts its lowest point by the same amount."""
    renderer = renderer_factory(spec)
    qpos = np.zeros(num_coordinates(spec))
    qpos[1] = 1.25
    low = renderer.ground_clearance(qpos)

    qpos[1] += 0.4
    assert renderer.ground_clearance(qpos) == pytest.approx(low + 0.4)
    # An upright robot at its nominal height stands on the floor.
    assert abs(low) < 0.05
