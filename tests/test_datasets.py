import pytest
import torch
from cfm.datasets.bimodal_distribution import BimodalDataset
from cfm.datasets.mnist import MNISTDataset
from cfm.datasets.unit_circle import UnitCircleDataset
from cfm.datasets.star import StarDataset
from cfm.datasets.spiral import SpiralDataset
from cfm.datasets.obstacle_paths import ObstaclePathDataset


# ---------------------------------------------------------------------------
# 2-D datasets
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[BimodalDataset, UnitCircleDataset, StarDataset, SpiralDataset]
)
def dataset_2d(request):
    return request.param(num_samples=100)


def test_len(dataset_2d):
    assert len(dataset_2d) == 100


def test_getitem_shape(dataset_2d):
    assert dataset_2d[0].shape == (2,)


def test_getitem_is_float(dataset_2d):
    assert dataset_2d[0].dtype == torch.float32


def test_data_is_finite(dataset_2d):
    assert torch.isfinite(dataset_2d.data).all()


# ---------------------------------------------------------------------------
# BimodalDataset
# ---------------------------------------------------------------------------


def test_bimodal_means():
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
    ds = BimodalDataset(num_samples=50)
    assert len(ds) == 50
    assert ds.data.shape == (50, 2)


# ---------------------------------------------------------------------------
# UnitCircleDataset
# ---------------------------------------------------------------------------


def test_unit_circle_points_on_unit_circle():
    ds = UnitCircleDataset(num_samples=512)
    radii = ds.data.norm(dim=1)
    assert torch.allclose(radii, torch.ones(512), atol=1e-5)


# ---------------------------------------------------------------------------
# MNISTDataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mnist_train():
    return MNISTDataset(train=True)


@pytest.fixture(scope="module")
def mnist_test():
    return MNISTDataset(train=False)


def test_mnist_chosen_digit():
    dataset = MNISTDataset(train=True, digit="3")
    assert len(dataset) > 0
    assert len(dataset) < 60000
    assert dataset[0].shape == (28, 28, 1)


def test_mnist_train_len(mnist_train):
    assert len(mnist_train) == 60000


def test_mnist_test_len(mnist_test):
    assert len(mnist_test) == 10000


def test_mnist_sample_shape(mnist_train):
    assert mnist_train[0].shape == (28, 28, 1)


def test_mnist_sample_dtype(mnist_train):
    assert mnist_train[0].dtype == torch.float32


def test_mnist_pixel_range(mnist_train):
    sample = mnist_train[0]
    assert sample.min() >= 0.0 and sample.max() <= 1.0


def test_mnist_data_tensor_shape(mnist_train):
    assert mnist_train.data.shape == (60000, 28, 28, 1)


# ---------------------------------------------------------------------------
# ObstaclePathDataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paths():
    return ObstaclePathDataset(num_samples=64, num_knots=8)


def test_obstacle_paths_shape(paths):
    assert len(paths) == 64
    assert paths.data.shape == (64, 8, 2)
    assert paths[0].shape == (8, 2)
    assert paths[0].dtype == torch.float32


def test_obstacle_paths_finite(paths):
    assert torch.isfinite(paths.data).all()


def test_obstacle_paths_endpoints(paths):
    full = paths.path(paths.data, num_sub=5)
    assert torch.allclose(full[:, 0], paths.start.expand(64, 2), atol=1e-6)
    assert torch.allclose(full[:, -1], paths.goal.expand(64, 2), atol=1e-6)


def test_obstacle_paths_spline_shape(paths):
    for num_sub in (1, 4, 10):
        full = paths.path(paths.data, num_sub=num_sub)
        assert full.shape == (64, num_sub * (8 + 1) + 1, 2)


def test_obstacle_paths_spline_interpolates_knots(paths):
    full = paths.path(paths.data, num_sub=4)
    assert torch.allclose(full[:, ::4], paths.full_knots(paths.data), atol=1e-6)


def test_obstacle_paths_are_smooth(paths):
    full = paths.path(paths.data, num_sub=20)
    steps = torch.linalg.norm(full[:, 1:] - full[:, :-1], dim=-1)
    accel = torch.linalg.norm(
        full[:, 2:] - 2 * full[:, 1:-1] + full[:, :-2], dim=-1
    )

    # Normalized so the bound is independent of sampling density.
    curvature = accel.max() / steps.mean() ** 2
    assert curvature < 40.0


def test_obstacle_paths_knots_are_tidy(paths):
    full = paths.full_knots(paths.data)
    steps = torch.linalg.norm(full[:, 1:] - full[:, :-1], dim=-1)
    spacing = torch.linalg.norm(paths.goal - paths.start) / (
        paths.num_knots + 1
    )
    assert steps.max() <= paths.max_step_ratio * spacing


def test_obstacle_paths_are_diverse():
    """Paths detour to both sides of the straight line from start to goal."""
    ds = ObstaclePathDataset(num_samples=256, num_knots=8)
    mean_offset = ds.data[..., 1].mean(dim=-1)
    assert (mean_offset > 0.05).any()
    assert (mean_offset < -0.05).any()
