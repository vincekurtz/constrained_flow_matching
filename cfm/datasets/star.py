import torch
from torch.utils.data import Dataset


class StarDataset(Dataset):
    """A simple manifold-based dataset: all points lie on a smooth 2D curve.

    The star is defined in polar coordinates as

        r(θ) = r_inner + (r_outer - r_inner) *
                            ((1 + cos(n_points * θ)) / 2) ** smoothness

    where `smoothness` controls how rounded the inner corners are (higher
    values give sharper tips and more rounded valleys).
    """

    def __init__(
        self,
        num_samples: int = 1024,
        n_points: int = 5,
        r_inner: float = 0.4,
        r_outer: float = 1.0,
        smoothness: float = 2.0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.n_points = n_points
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.smoothness = smoothness
        self.data = self._generate_data()

    def _generate_data(self):
        torch.random.manual_seed(0)
        angles = torch.rand(self.num_samples) * 2 * torch.pi
        t = (1 + torch.cos(self.n_points * angles)) / 2
        r = self.r_inner + (self.r_outer - self.r_inner) * t ** self.smoothness
        x = r * torch.cos(angles)
        y = r * torch.sin(angles)
        return torch.stack([x, y], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # Make a quick plot to visualize the dataset
    import matplotlib.pyplot as plt

    dataset = StarDataset(num_samples=2048)
    data = dataset.data.numpy()

    plt.axes().set_aspect("equal")
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
    plt.title("Star Training Data")

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.tight_layout()
    plt.show()
