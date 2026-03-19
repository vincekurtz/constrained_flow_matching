import torch
from torch.utils.data import Dataset


class SpiralDataset(Dataset):
    """A simple spiral-shaped dataset in 2D.

    This is a classic flow matching/diffusion example.
    """

    def __init__(self, num_samples: int = 1024, scale: float = 1.0):
        super().__init__()
        self.num_samples = num_samples
        self.scale = scale
        self.data = self._generate_data()

    def _generate_data(self):
        torch.random.manual_seed(0)
        theta = 6 * torch.pi * torch.rand(self.num_samples)
        r = theta / (2 * torch.pi) * self.scale
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # Make a quick plot to visualize the dataset
    import matplotlib.pyplot as plt

    dataset = SpiralDataset(num_samples=1024)
    data = dataset.data.numpy()

    plt.axes().set_aspect("equal")
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
    plt.title("Spiral Training Data")

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.tight_layout()
    plt.show()
