import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms.v2 as transforms


class MNISTDataset(Dataset):
    """Unlabeled MNIST images, float32 of shape (28, 28, 1) in [0, 1].

    ``digit`` keeps a single digit, or "all".
    """

    def __init__(self, train: bool = True, root: str = "data", digit="all"):
        super().__init__()
        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        raw = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )
        images = torch.stack(
            [img for img, d in raw if digit == "all" or d == int(digit)]
        )
        self.data = images.permute(0, 2, 3, 1)  # (N, 28, 28, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = MNISTDataset(train=True)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, i in zip(axes.flat, range(10)):
        ax.imshow(dataset[i].squeeze(-1), cmap="gray")
        ax.axis("off")
    plt.suptitle("MNIST Training Samples")
    plt.tight_layout()
    plt.show()
