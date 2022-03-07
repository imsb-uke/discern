"""Module to select the mmd loss function."""

import logging
from typing import Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    from discern.mmd._mmd import _mmd_loop as _mmd_loop_c
    USE_C_IMPLEMENTATION = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    _LOGGER.warning("Fallback to Python version, MMD computation may be slow")
    USE_C_IMPLEMENTATION = False
else:  # pragma: no cover
    _LOGGER.debug("Using cython version of MMD")


def _mmd_loop_py(dist_xy, dist_xx, dist_yy, scales, sigma):
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


if USE_C_IMPLEMENTATION:  # pragma: no cover
    _mmd_loop = _mmd_loop_c  # pylint: disable=invalid-name
else:  # pragma: no cover
    _mmd_loop = _mmd_loop_py  # pylint: disable=invalid-name


def _calculate_distances(
        x: np.ndarray,
        y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate euclidean distances.

    Faster implementation than calling
    sklearn.metrics.pairwise.euclidean_distance three times, but
    without multiprocessing.

    Args:
        x (np.ndarray): First array
        y (np.ndarray): Second array

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Euclidian distance between x-y, x-x and y-y.
    """
    # pylint: disable=invalid-name
    dot_x = np.einsum('ij,ij->i', x, x)[:, np.newaxis]
    dot_y = np.einsum('ij,ij->i', y, y)[np.newaxis, :]
    dist_xy = np.matmul(x, y.T)
    dist_xx = np.matmul(x, x.T)
    dist_yy = np.matmul(y, y.T)
    np.multiply(dist_xy, -2., out=dist_xy)
    np.multiply(dist_xx, -2., out=dist_xx)
    np.multiply(dist_yy, -2., out=dist_yy)
    np.add(dist_xy, dot_x, out=dist_xy)
    np.add(dist_xy, dot_y, out=dist_xy)
    np.add(dist_xx, dot_x, out=dist_xx)
    np.add(dist_xx, dot_x.T, out=dist_xx)
    np.add(dist_yy, dot_y.T, out=dist_yy)
    np.add(dist_yy, dot_y, out=dist_yy)
    np.fill_diagonal(dist_xx, 0.)
    np.fill_diagonal(dist_yy, 0.)
    return dist_xy, dist_xx, dist_yy


def mmd_loss(random_cells: np.ndarray, valid_cells: np.ndarray,
             sigma: float) -> float:
    """Compute mmd loss between random cells and valid cells.

    Args:
        random_cells (np.ndarray): Random generated cells.
        valid_cells (np.ndarray): Valid (decoded) cells.
        sigma (float): Precalculated Sigma value.

    Returns:
        float: MMD loss between random and valid cells.
    """
    # pylint: disable=too-many-locals
    random_cells = random_cells.astype(np.float32)
    valid_cells = valid_cells.astype(np.float32)

    dist_xy, dist_xx, dist_yy = _calculate_distances(random_cells, valid_cells)

    scales = np.linspace(0.8, 1.5, num=23, dtype=np.float32)

    sigma = np.float32(sigma)
    return _mmd_loop(dist_xy, dist_xx, dist_yy, scales, sigma)
