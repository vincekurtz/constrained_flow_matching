import torch
from torch.utils.data import Dataset


def bezier_spline(knots: torch.Tensor, num_sub: int) -> torch.Tensor:
    """Sample the C1 cubic Bezier spline through ``knots`` (..., N, 2).

    Control points come from Catmull-Rom tangents m_i = (P_{i+1} - P_{i-1}) / 2
    (one-sided at the ends): B_1 = P_i + m_i / 3, B_2 = P_{i+1} - m_{i+1} / 3.
    Returns shape (..., num_sub * (N - 1) + 1, 2).
    """
    interior = 0.5 * (knots[..., 2:, :] - knots[..., :-2, :])
    first = knots[..., 1:2, :] - knots[..., 0:1, :]
    last = knots[..., -1:, :] - knots[..., -2:-1, :]
    tangents = torch.cat([first, interior, last], dim=-2)

    b0 = knots[..., :-1, :]
    b3 = knots[..., 1:, :]
    b1 = b0 + tangents[..., :-1, :] / 3.0
    b2 = b3 - tangents[..., 1:, :] / 3.0

    u = torch.arange(num_sub, dtype=knots.dtype) / num_sub
    u = u.reshape((1,) * (knots.dim() - 1) + (num_sub, 1))
    points = (
        (1 - u) ** 3 * b0[..., None, :]
        + 3 * (1 - u) ** 2 * u * b1[..., None, :]
        + 3 * (1 - u) * u**2 * b2[..., None, :]
        + u**3 * b3[..., None, :]
    )

    points = points.reshape(knots.shape[:-2] + (-1, 2))
    return torch.cat([points, knots[..., -1:, :]], dim=-2)


class ObstaclePathDataset(Dataset):
    """Smooth 2D paths from a fixed start to a fixed goal.

    Each sample is ``num_knots`` interior knots (T, 2); the path is the Bezier
    spline through [start, p_1, ..., p_T, goal]. Paths are a straight line plus
    a random Fourier wiggle, optimized out of random circular obstacles that
    are then discarded, so the dataset is unconditional.

    Args:
        num_obstacles: obstacles used to shape each path.
        clearance: margin kept between path and obstacles.
        wiggle_scale: magnitude of the Fourier perturbation.
        num_modes: number of Fourier modes in the perturbation.
        max_step_ratio: reject paths with a knot spacing longer than this
            multiple of the straight-line spacing.
        spline_samples: samples per segment for collision checks.
    """

    def __init__(
        self,
        num_samples: int = 4096,
        num_knots: int = 10,
        start: tuple = (-1.0, 0.0),
        goal: tuple = (1.0, 0.0),
        num_obstacles: int = 2,
        obstacle_radius_range: tuple = (0.15, 0.35),
        clearance: float = 0.05,
        wiggle_scale: float = 0.4,
        num_modes: int = 3,
        max_step_ratio: float = 2.5,
        spline_samples: int = 8,
        seed: int = 0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_knots = num_knots
        self.start = torch.tensor(start, dtype=torch.float32)
        self.goal = torch.tensor(goal, dtype=torch.float32)
        self.num_obstacles = num_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.clearance = clearance
        self.wiggle_scale = wiggle_scale
        self.num_modes = num_modes
        self.max_step_ratio = max_step_ratio
        self.spline_samples = spline_samples
        self.seed = seed
        self.data = self._generate_data()

    def full_knots(self, knots: torch.Tensor) -> torch.Tensor:
        """Add start and goal to interior knots (..., T, 2)."""
        lead = knots.shape[:-2]
        start = self.start.expand(*lead, 1, 2)
        goal = self.goal.expand(*lead, 1, 2)
        return torch.cat([start, knots, goal], dim=-2)

    def path(self, knots: torch.Tensor, num_sub: int = None) -> torch.Tensor:
        """Sample the spline path through interior knots (..., T, 2)."""
        if num_sub is None:
            num_sub = self.spline_samples
        return bezier_spline(self.full_knots(knots), num_sub)

    def _straight_line(self) -> torch.Tensor:
        s = torch.arange(1, self.num_knots + 1) / (self.num_knots + 1)
        return self.start + s[:, None] * (self.goal - self.start)

    def _sample_obstacles(self, num_paths: int) -> tuple:
        """Sample (centers, radii) of obstacles near the straight line."""
        shape = (num_paths, self.num_obstacles)
        r_min, r_max = self.obstacle_radius_range
        radii = r_min + (r_max - r_min) * torch.rand(shape)

        s = 0.15 + 0.7 * torch.rand(shape)
        along = self.start + s[..., None] * (self.goal - self.start)
        offset = 0.35 * (2 * torch.rand(shape + (2,)) - 1)
        return along + offset, radii

    def _random_wiggle(self, num_paths: int) -> torch.Tensor:
        """Smooth random perturbations that vanish at both endpoints."""
        s = torch.arange(1, self.num_knots + 1) / (self.num_knots + 1)
        k = torch.arange(1, self.num_modes + 1)

        basis = torch.sin(torch.pi * s[:, None] * k[None, :])  # (T, K)
        amps = torch.randn(num_paths, self.num_modes, 2) / k[:, None] ** 1.5

        # Mostly sideways, so paths don't double back.
        tangent = (self.goal - self.start) / torch.linalg.norm(
            self.goal - self.start
        )
        normal = torch.stack([-tangent[1], tangent[0]])
        directions = torch.stack([0.15 * tangent, normal])  # (2, 2)
        return self.wiggle_scale * (basis @ amps @ directions)

    def _penetration(
        self,
        knots: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        clearance: float,
    ) -> torch.Tensor:
        pts = self.path(knots)  # (N, P, 2)
        deltas = pts[:, :, None, :] - centers[:, None, :, :]  # (N, P, M, 2)
        dists = torch.linalg.norm(deltas, dim=-1)  # (N, P, M)
        return torch.relu(radii[:, None, :] + clearance - dists)

    def _optimize_paths(
        self,
        knots: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        num_steps: int = 150,
    ) -> torch.Tensor:
        """Push paths out of their obstacles with a few Adam steps."""
        p = knots.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.02)

        for _ in range(num_steps):
            opt.zero_grad()
            full = self.full_knots(p)

            penetration = self._penetration(p, centers, radii, self.clearance)
            obstacle_cost = torch.sum(penetration**2)

            # Keep knots from bunching up or zig-zagging.
            accel = full[:, 2:] - 2 * full[:, 1:-1] + full[:, :-2]
            smooth_cost = torch.sum(accel**2)

            steps = full[:, 1:] - full[:, :-1]
            length_cost = torch.sum(steps**2)

            cost = 100.0 * obstacle_cost + 2.0 * smooth_cost + 0.2 * length_cost
            cost.backward()
            opt.step()

        return p.detach()

    def _generate_data(self) -> torch.Tensor:
        torch.random.manual_seed(self.seed)
        line = self._straight_line()
        spacing = torch.linalg.norm(self.goal - self.start) / (
            self.num_knots + 1
        )

        paths = []
        num_found = 0
        attempts = 0
        while num_found < self.num_samples and attempts < 20:
            attempts += 1
            num_needed = self.num_samples - num_found
            centers, radii = self._sample_obstacles(num_needed)
            knots = line + self._random_wiggle(num_needed)
            knots = self._optimize_paths(knots, centers, radii)

            penetration = self._penetration(knots, centers, radii, 0.0)
            collision_free = torch.amax(penetration, dim=(1, 2)) <= 0.0

            full = self.full_knots(knots)
            steps = torch.linalg.norm(full[:, 1:] - full[:, :-1], dim=-1)
            tidy = torch.amax(steps, dim=1) <= self.max_step_ratio * spacing

            keep = collision_free & tidy
            paths.append(knots[keep])
            num_found += int(keep.sum())

        if num_found < self.num_samples:
            raise RuntimeError(
                f"Only generated {num_found}/{self.num_samples} paths in "
                f"{attempts} rounds."
            )

        return torch.cat(paths, dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = ObstaclePathDataset(num_samples=512)
    paths = dataset.path(dataset.data, num_sub=20).numpy()

    plt.axes().set_aspect("equal")
    for path in paths:
        plt.plot(path[:, 0], path[:, 1], "C0-", alpha=0.15, lw=1)
    plt.plot(*dataset.start, "go", ms=10, label="start")
    plt.plot(*dataset.goal, "r*", ms=14, label="goal")
    plt.title("Obstacle Path Training Data")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
