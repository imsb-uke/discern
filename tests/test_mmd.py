"""Test mmd related functions."""
import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances

from discern.mmd import mmd


def _mmd_loop(dist_xy, dist_xx, dist_yy, scales, sigma):
    # pylint: disable=too-many-locals
    stat = np.zeros_like(scales)
    n_x = np.float(dist_xx.shape[0])
    n_y = np.float(dist_yy.shape[0])
    for i, k in enumerate(scales):
        val = k * sigma
        k_xx = np.exp(-dist_xx / (2 * val))
        np.fill_diagonal(k_xx, 0.0)
        k_xxnd = np.sum(k_xx) / (n_x * n_x - n_x)
        k_yy = np.exp(-dist_yy / (2 * val))
        np.fill_diagonal(k_yy, 0.0)
        k_yynd = np.sum(k_yy) / (n_y * n_y - n_y)
        res1 = k_xxnd + k_yynd
        res2 = np.exp(-dist_xy / (2 * val))
        res2 = np.sum(res2) * 2. / (n_x * n_y)
        stat[i] = res1 - res2
    return np.max(stat)


@pytest.mark.parametrize("n_rows", [10, 25, 100, 500, 1000])
@pytest.mark.parametrize("n_cols", [10, 25, 100, 500, 1000])
def test_calculate_distances(n_rows, n_cols):
    """Test _calculate_distances function."""
    x = np.random.rand(n_rows, n_cols)  # pylint: disable=invalid-name
    y = np.random.rand(n_rows, n_cols)  # pylint: disable=invalid-name
    expected = (euclidean_distances(x, y)**2, euclidean_distances(x, x)**2,
                euclidean_distances(y, y)**2)
    got = mmd._calculate_distances(x, y)  # pylint: disable=protected-access
    np.testing.assert_allclose(got[0], expected[0])
    np.testing.assert_allclose(got[1], expected[1])
    np.testing.assert_allclose(got[2], expected[2])


@pytest.mark.parametrize("shape", [25, 100, 500, 1000])
@pytest.mark.parametrize("sigma", [0.1, 1., 5., 7.5, 15.])
def test_mmd_loop_py(shape, sigma):
    """Test _mmd_loop_py function."""
    x = np.random.rand(shape, 1000).astype(np.float32)  # pylint: disable=invalid-name
    y = np.random.rand(shape, 1000).astype(np.float32)  # pylint: disable=invalid-name
    dist_xy, dist_xx, dist_yy = (euclidean_distances(x, y)**2,
                                 euclidean_distances(x, x)**2,
                                 euclidean_distances(y, y)**2)
    scales = np.linspace(0.8, 1.5, num=23, dtype=np.float32)
    sigma = np.float32(sigma)
    expected = _mmd_loop(dist_xy, dist_xx, dist_yy, scales, sigma)
    got = mmd._mmd_loop_py(dist_xy, dist_xx, dist_yy, scales, sigma)  # pylint: disable=protected-access
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize(
    "shape", (1000, 2000, pytest.param(4000, marks=pytest.mark.slow)))
def test_mmd_loop_py_unbalanced(shape):
    """Test _mmd_loop_py function."""
    x = np.random.rand(100, 1000).astype(np.float32)  # pylint: disable=invalid-name
    y = np.random.rand(shape, 1000).astype(np.float32)  # pylint: disable=invalid-name
    dist_xy, dist_xx, dist_yy = (euclidean_distances(x, y)**2,
                                 euclidean_distances(x, x)**2,
                                 euclidean_distances(y, y)**2)
    scales = np.linspace(0.8, 1.5, num=23, dtype=np.float32)
    sigma = np.float32(6.)
    expected = _mmd_loop(dist_xy, dist_xx, dist_yy, scales, sigma)
    got = mmd._mmd_loop_py(dist_xy, dist_xx, dist_yy, scales, sigma)  # pylint: disable=protected-access
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.skipif(not mmd.USE_C_IMPLEMENTATION,
                    reason="Testing C version required compiled binary")
@pytest.mark.parametrize("shape", [25, 100, 500, 1000])
@pytest.mark.parametrize("sigma", [0.1, 1., 5., 7.5, 15.])
def test_mmd_loop_c_version(shape, sigma):
    """Test _mmd_loop function."""
    x = np.random.rand(shape, 1000).astype(np.float32)  # pylint: disable=invalid-name
    y = np.random.rand(shape, 1000).astype(np.float32)  # pylint: disable=invalid-name
    dist_xy, dist_xx, dist_yy = (euclidean_distances(x, y)**2,
                                 euclidean_distances(x, x)**2,
                                 euclidean_distances(y, y)**2)
    scales = np.linspace(0.8, 1.5, num=23, dtype=np.float32)
    sigma = np.float32(sigma)
    expected = _mmd_loop(dist_xy, dist_xx, dist_yy, scales, sigma)
    got = mmd._mmd_loop_c(dist_xy, dist_xx, dist_yy, scales, sigma)  # pylint: disable=protected-access
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.skipif(not mmd.USE_C_IMPLEMENTATION,
                    reason="Testing C version required compiled binary")
@pytest.mark.parametrize(
    "shape", (1000, 2000, pytest.param(4000, marks=pytest.mark.slow)))
def test_mmd_loop_c_version_unbalanced(shape):
    """Test _mmd_loop function."""
    x = np.random.rand(100, 1000).astype(np.float32)  # pylint: disable=invalid-name
    y = np.random.rand(shape, 1000).astype(np.float32)  # pylint: disable=invalid-name
    dist_xy, dist_xx, dist_yy = (euclidean_distances(x, y)**2,
                                 euclidean_distances(x, x)**2,
                                 euclidean_distances(y, y)**2)
    scales = np.linspace(0.8, 1.5, num=23, dtype=np.float32)
    sigma = np.float32(6.)
    expected = _mmd_loop(dist_xy, dist_xx, dist_yy, scales, sigma)
    got = mmd._mmd_loop_c(dist_xy, dist_xx, dist_yy, scales, sigma)  # pylint: disable=protected-access
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_mmd():
    """Test if mmd_loss throws exception."""
    np.random.seed(42)
    x = np.random.rand(1000, 500).astype(np.float32)  # pylint: disable=invalid-name
    y = np.random.rand(950, 500).astype(np.float32)  # pylint: disable=invalid-name
    got = mmd.mmd_loss(x, y, 5.0)
    np.testing.assert_allclose(got, 1.418614e-06, rtol=0.0018)
    np.random.seed(None)  # To re-seed the generator for different functions
