import torch
import pytest
from datasets.bimodal_distribution import BimodalDataset


@pytest.fixture
def dataset():
    return BimodalDataset(num_samples=100)


def test_len(dataset):
    """Dataset length matches num_samples."""
    assert len(dataset) == 100


def test_getitem_shape(dataset):
    """Each sample is a 2-D tensor."""
    sample = dataset[0]
    assert sample.shape == (2,)


def test_getitem_is_float(dataset):
    """Samples are floating-point tensors."""
    assert dataset[0].dtype == torch.float32


def test_data_is_finite(dataset):
    """All samples are finite (no NaN or Inf)."""
    assert torch.isfinite(dataset.data).all()


def test_bimodal_means():
    """Samples cluster around the two specified means."""
    mean1 = [1.0, 0.0]
    mean2 = [-1.0, 0.0]
    ds = BimodalDataset(num_samples=10000, mean1=mean1, mean2=mean2, std=0.1)

    first_half = ds.data[: len(ds) // 2]
    second_half = ds.data[len(ds) // 2 :]

    assert torch.allclose(
        first_half.mean(dim=0), torch.tensor(mean1), atol=0.05
    )
    assert torch.allclose(
        second_half.mean(dim=0), torch.tensor(mean2), atol=0.05
    )


def test_custom_num_samples():
    """Dataset respects a custom num_samples value."""
    ds = BimodalDataset(num_samples=50)
    assert len(ds) == 50
    assert ds.data.shape == (50, 2)
