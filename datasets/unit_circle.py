import torch
from torch.utils.data import Dataset

class UnitCircleDataset(Dataset):
    """A simple manifold-based dataset: all points lie on the 2D unit circle."""
    def __init__(self, num_samples: int = 1024):
        super().__init__()
        self.num_samples = num_samples
        self.data = self._generate_data()

    def _generate_data(self):
        torch.random.manual_seed(0)
        angles = torch.rand(self.num_samples) * 2 * torch.pi
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # Make a quick plot to visualize the dataset
    import matplotlib.pyplot as plt

    dataset = UnitCircleDataset(num_samples=128)
    data = dataset.data.numpy()

    plt.axes().set_aspect("equal")
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title("Unit Circle Training Data")

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.tight_layout()
    plt.show()
