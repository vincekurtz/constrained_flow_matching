"""A 2D surrogate of the D3IL "Avoiding" robotic manipulation task.

In the original D3IL benchmark a Franka end-effector must travel from a fixed
start to a goal line while weaving around six fixed pillars, with many valid
(multi-modal) paths. The action space is end-effector velocity in a plane and
the obstacles are fixed, so the task reduces to 2D point navigation around six
circles.

This dataset provides expert-like demonstration *trajectories* for that 2D
task. Each sample is a sequence of ``TRAJ_LEN`` 2D waypoints starting at
``START`` and ending on the goal line ``GOAL_Y``, weaving through the gaps
between the pillars. The demonstrations are multi-modal (paths pass left or
right of the obstacle field) and collision-free with respect to the original
pillar radii.

The layout constants (``START``, ``GOAL_Y``, ``OBSTACLES``) are exported so the
example script and plotting utilities share a single source of truth.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# Number of waypoints per demonstration trajectory.
TRAJ_LEN = 32

# Fixed start point (bottom centre) and goal line (top).
START = (0.0, -1.0)
GOAL_Y = 1.0

# Six pillars in a 3-2-1 "bowling pin" triangle with the apex toward the start,
# ordered from nearest the start to nearest the goal as (center_x, center_y,
# radius). Obstacle 0 is the lone apex pillar (the "first" circle dead ahead of
# the start); obstacles 3 and 5 are the outer pair of the back row (the "last
# two" near the goal).
PILLAR_RADIUS = 0.14
OBSTACLES = (
    (0.0, -0.45, PILLAR_RADIUS),    # 0: apex, dead ahead of the start
    (-0.32, 0.0, PILLAR_RADIUS),    # 1: middle row (2 pillars)
    (0.32, 0.0, PILLAR_RADIUS),     # 2: middle row
    (-0.6, 0.5, PILLAR_RADIUS),     # 3: back row (3 pillars), outer-left
    (0.0, 0.5, PILLAR_RADIUS),      # 4: back row, centre
    (0.6, 0.5, PILLAR_RADIUS),      # 5: back row, outer-right
)


def _trajectory_collides(traj: np.ndarray, margin: float = 0.0) -> bool:
    """Return True if any waypoint lies within (radius + margin) of a pillar."""
    for cx, cy, r in OBSTACLES:
        d2 = (traj[:, 0] - cx) ** 2 + (traj[:, 1] - cy) ** 2
        if np.any(d2 < (r + margin) ** 2):
            return True
    return False


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Edge-preserving moving-average smoother for a 1-D signal."""
    kernel = np.ones(k) / k
    padded = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: x.shape[0]]


def _sample_trajectory(rng: np.random.Generator) -> np.ndarray:
    """Sample one smooth, multi-modal candidate path from START to the goal.

    The path is a vertical sweep ``y: -1 -> GOAL_Y`` whose lateral profile is
    routed through the gaps in the 3-2-1 pillar triangle. The apex pillar is
    passed on a random side, while the middle and back rows are threaded through
    laterally spread-out targets sampled from wide ranges. The actual gap is
    enforced by rejection sampling in ``_generate_data``; the wide sampling here
    gives the demonstrations broad spatial coverage. A light smoothing pass
    rounds the corners into pretty curves.
    """
    side_a = rng.choice([-1.0, 1.0])  # pass the apex pillar left or right

    y_knots = np.array([-1.0, -0.45, 0.0, 0.5, 1.0])
    x_row1 = side_a * rng.uniform(0.24, 0.42)   # around the apex pillar
    x_row2 = rng.uniform(-0.62, 0.62)           # through the middle row
    x_row3 = rng.uniform(-0.88, 0.88)           # through the back row
    x_goal = x_row3 + rng.uniform(-0.15, 0.15)  # continue to the goal line
    x_knots = np.array([0.0, x_row1, x_row2, x_row3, x_goal])

    s = np.linspace(0.0, 1.0, TRAJ_LEN)  # normalised arc length
    y = START[1] + (GOAL_Y - START[1]) * s
    x = np.interp(y, y_knots, x_knots)
    x = _smooth(x, k=3)
    x[0] = START[0]  # anchor the start exactly
    x = x + rng.normal(0.0, 0.01, size=x.shape)
    x[0] = START[0]

    return np.stack([x, y], axis=1)


class AvoidingDataset(Dataset):
    """Expert demonstration trajectories for the 2D avoiding task.

    Each item is a tensor of shape ``(TRAJ_LEN, 2)``.
    """

    def __init__(self, num_samples: int = 4096, seed: int = 0):
        super().__init__()
        self.num_samples = num_samples
        self.seed = seed
        self.data = self._generate_data()

    def _generate_data(self) -> torch.Tensor:
        rng = np.random.default_rng(self.seed)
        trajs = []
        # Rejection-sample collision-free paths (with a small margin so demos
        # keep clear of the original pillars).
        max_attempts = self.num_samples * 1000
        attempts = 0
        while len(trajs) < self.num_samples and attempts < max_attempts:
            attempts += 1
            traj = _sample_trajectory(rng)
            if not _trajectory_collides(traj, margin=0.02):
                trajs.append(traj)
        if len(trajs) < self.num_samples:
            raise RuntimeError(
                f"Only generated {len(trajs)}/{self.num_samples} "
                "collision-free trajectories; relax the layout or margin."
            )
        data = np.stack(trajs, axis=0).astype(np.float32)
        return torch.from_numpy(data)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


if __name__ == "__main__":
    # Quick visualisation of the demonstration trajectories and the layout.
    import matplotlib.pyplot as plt

    dataset = AvoidingDataset(num_samples=200)
    data = dataset.data.numpy()

    fig, ax = plt.subplots(figsize=(6, 7))
    for cx, cy, r in OBSTACLES:
        ax.add_patch(plt.Circle((cx, cy), r, color="gray", alpha=0.6))
    for traj in data:
        ax.plot(traj[:, 0], traj[:, 1], color="C0", alpha=0.15)
    ax.plot(*START, "ks", markersize=8, label="start")
    ax.axhline(GOAL_Y, color="green", linestyle="--", label="goal line")

    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Avoiding demonstrations")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
