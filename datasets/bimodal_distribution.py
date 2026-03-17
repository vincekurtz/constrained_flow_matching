import torch
from torch.utils.data import Dataset

class BimodalDataset(Dataset):
    """A simple bimodal distribution for testing flow-matching training."""

    def __init__(
        self,
        num_samples: int = 1000,
        mean1=[1.0, 5.0],
        mean2=[5.0, -5.0],
        std=1.0,
    ):
        """Create the dataset by sampling from a mixture of two Gaussians."""
        super().__init__()
        self.num_samples = num_samples
        self.mean1 = torch.tensor(mean1)
        self.mean2 = torch.tensor(mean2)
        self.std = std
        self.data = self._generate_data()

    def _generate_data(self) -> torch.Tensor:
        """Generate samples from a mixture of Gaussians."""
        torch.random.manual_seed(0)
        mode1_samples = (
            torch.randn(self.num_samples // 2, 2) * self.std + self.mean1
        )
        mode2_samples = (
            torch.randn(self.num_samples // 2, 2) * self.std + self.mean2
        )
        return torch.cat([mode1_samples, mode2_samples], dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    # Make a quick plot to visualize the dat
    import matplotlib.pyplot as plt

    dataset = BimodalDataset(num_samples=1000)
    data = dataset.data.numpy()

    plt.axes().set_aspect("equal")
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title("Bimodal Training Data")

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.tight_layout()
    plt.show()
