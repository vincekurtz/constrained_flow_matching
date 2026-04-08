import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms.v2 as transforms


class MNISTDataset(Dataset):
    """MNIST handwritten-digit images, returned as normalised float tensors.

    Class labels are discarded; only the raw images are exposed.

    Each sample is a float32 tensor of shape ``(28, 28, 1)`` with pixel values
    scaled to ``[0, 1]``.
    """

    def __init__(self, train: bool = True, root: str = "data"):
        """Download (if necessary) and load the MNIST split.

        Args:
            train: If ``True`` load the training split (60 000 images),
                otherwise load the test split (10 000 images).
            root: Directory under which the raw dataset is cached.
        """
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
        # Stack all images into a single tensor of shape (N, 1, 28, 28),
        # then permute to (N, 28, 28, 1) to match the (H, W, C) convention.
        images = torch.stack([img for img, _ in raw])  # (N, 1, 28, 28)
        self.data = images.permute(0, 2, 3, 1)  # (N, 28, 28, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


if __name__ == "__main__":
    # Make a quick plot to visualise a few samples
    import matplotlib.pyplot as plt

    dataset = MNISTDataset(train=True)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, i in zip(axes.flat, range(10)):
        ax.imshow(dataset[i].squeeze(-1), cmap="gray")
        ax.axis("off")
    plt.suptitle("MNIST Training Samples")
    plt.tight_layout()
    plt.show()
