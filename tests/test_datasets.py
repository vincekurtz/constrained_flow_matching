import pytest
import torch
from datasets.bimodal_distribution import BimodalDataset
from datasets.unit_circle import UnitCircleDataset
from datasets.star import StarDataset


# ---------------------------------------------------------------------------
# Common 2-D dataset tests (parametrized over all datasets)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[BimodalDataset, UnitCircleDataset, StarDataset])
def dataset_2d(request):
    return request.param(num_samples=100)


def test_len(dataset_2d):
    """Dataset length matches num_samples."""
    assert len(dataset_2d) == 100


def test_getitem_shape(dataset_2d):
    """Each sample is a 2-D tensor."""
    assert dataset_2d[0].shape == (2,)


def test_getitem_is_float(dataset_2d):
    """Samples are floating-point tensors."""
    assert dataset_2d[0].dtype == torch.float32


def test_data_is_finite(dataset_2d):
    """All samples are finite (no NaN or Inf)."""
    assert torch.isfinite(dataset_2d.data).all()


# ---------------------------------------------------------------------------
# BimodalDataset-specific tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# UnitCircleDataset-specific tests
# ---------------------------------------------------------------------------


def test_unit_circle_points_on_unit_circle():
    """All samples lie on the unit circle (radius ≈ 1)."""
    ds = UnitCircleDataset(num_samples=512)
    radii = ds.data.norm(dim=1)
    assert torch.allclose(radii, torch.ones(512), atol=1e-5)
