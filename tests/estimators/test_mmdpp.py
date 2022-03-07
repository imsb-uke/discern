"""Testing MMDPP."""
import numpy as np
import pytest
import tensorflow as tf

from discern.estimators import customlayers


def _distance_original(points):
    norms = tf.reduce_sum(tf.square(points), axis=1, keepdims=True)
    dotprods = tf.matmul(points, points, transpose_b=True)

    return norms, norms + tf.transpose(norms) - 2. * dotprods


def _block_diagonal_original(matrices, dtype=tf.float32):
    """Construct block-diagonal matrices from a list of batched 2D tensors.

    Taken from: https://stackoverflow.com/questions/42157781/block-diagonal-matrices-in-tensorflow
    Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of matrices with the
     same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [sum_i N_i, sum_i M_i].
    """
    matrices = [
        tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices
    ]
    blocked_rows = 0
    blocked_cols = 0
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(
            tf.pad(tensor=matrix,
                   paddings=tf.concat([
                       tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                       [(row_before_length, row_after_length)]
                   ],
                                      axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))

    return blocked


def _mmdpp_penalty_original(sample_qz, sample_pz, encoder_sigma,
                            total_number_cells, batch_size, latent_dim):
    # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    n = batch_size
    num_sample = 8

    eps = tf.random.normal((n * num_sample, latent_dim),
                           mean=0.,
                           stddev=1.,
                           dtype=tf.float32)

    block_var = tf.reshape(tf.tile(tf.exp(encoder_sigma), [1, num_sample]),
                           [-1, latent_dim])

    eps_q = tf.multiply(eps, block_var)

    block_means = tf.reshape(tf.tile(sample_qz, [1, num_sample]),
                             [-1, latent_dim])
    sample_qhat = block_means + eps_q
    norms_pz, distances_pz = _distance_original(sample_pz)
    norms_qhat, distances_qhat = _distance_original(sample_qhat)

    dotprods_pz_qhat = tf.matmul(sample_pz, sample_qhat, transpose_b=True)
    distances_pz_qhat = norms_pz + tf.transpose(
        norms_qhat) - 2. * dotprods_pz_qhat
    mask = _block_diagonal_original([
        np.ones((num_sample, num_sample), dtype=np.float32) for _ in range(n)
    ], tf.float32)

    cbase = 2. * latent_dim  # 2 Can documentation?

    stat = 0.

    for scale in [.1, .5, 1., 2., 10.]:
        c = cbase * scale
        res1 = c / (c + distances_pz)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (n * n - n)

        res2 = c / (c + distances_pz_qhat)
        res2 = tf.reduce_sum(res2) / (n * n) / num_sample

        res3 = c / (c + distances_qhat)
        res3 = tf.multiply(res3, 1. - mask)
        res3 = tf.reduce_sum(res3) / (n * (n - 1)) / (num_sample**2)

        res4 = c / (c + distances_qhat)
        res4 = tf.multiply(res4, mask - tf.eye(n * num_sample))
        res4 = tf.reduce_sum(res4) / n / num_sample / (
            num_sample - 1.) / total_number_cells  # Can?

        stat += (res1 - 2 * res2 + res3 + res4)

    return stat


@pytest.mark.parametrize("num_sample, batch_size", [(10, 2), (10, 10),
                                                    (10, 20)])
def test_blockdiagnoal(num_sample, batch_size):
    """Test creation of blockdiagonal matrix."""
    blocks = [
        np.ones((num_sample, num_sample), dtype=np.float32)
        for _ in range(batch_size)
    ]
    expected = _block_diagonal_original(blocks)
    got = customlayers._block_diagonal(batch_size, num_sample)  # pylint: disable=protected-access
    assert got.shape == expected.shape
    assert np.all(got == expected)


@pytest.mark.parametrize("shape", range(1, 1000, 125))
def test_eye(shape):
    """Test creation of eye matrix."""
    expected = tf.eye(shape, shape).numpy()
    got = customlayers._eye(shape, name="matrix")  # pylint: disable=protected-access
    assert expected.shape == got.shape
    np.testing.assert_allclose(expected, got)


@pytest.mark.parametrize("shape", [(150, 50), (200, 128), (200, 200)])
@pytest.mark.parametrize("loc", [0, 1, 5])
@pytest.mark.parametrize("scale", [1, 2, 5])
@pytest.mark.flaky(reruns=5)
def test_mmdpp_penalty(shape, loc, scale):
    """Test mmdpp_penalty, comparing to old version."""
    batch_size, latent_dim = shape
    total_number_cells = 100
    sample_qz = tf.random.normal(mean=loc,
                                 stddev=1.,
                                 shape=(batch_size, latent_dim))
    sample_pz = tf.random.normal(mean=0.,
                                 stddev=1.,
                                 shape=(batch_size, latent_dim))
    encoder_sigma = tf.random.normal(
        mean=scale, stddev=1., shape=(batch_size, latent_dim)) + 0.1
    encoder_sigma = encoder_sigma**2

    expected = _mmdpp_penalty_original(sample_qz, sample_pz, encoder_sigma,
                                       total_number_cells, batch_size,
                                       latent_dim).numpy()

    got = customlayers.mmdpp_penalty(sample_qz=sample_qz,
                                     sample_pz=sample_pz,
                                     encoder_sigma=encoder_sigma,
                                     total_number_cells=total_number_cells,
                                     latent_dim=latent_dim).numpy()
    # adjusting the tolerance based on the number of samples.
    rtol = 400 / (batch_size * latent_dim)
    np.testing.assert_allclose(got, expected, rtol=rtol)


@pytest.mark.parametrize("shape", [(150, 50), (200, 128), (200, 200)])
@pytest.mark.parametrize("loc", [0, 1, 5])
@pytest.mark.parametrize("scale", [1, 2, 5])
def test_distances(shape, loc, scale):
    """Test _distance function."""
    sample = tf.random.normal(mean=loc, stddev=scale, shape=shape)
    norms_original, distance_original = _distance_original(sample)
    norms_new, distance_new = customlayers._distance(sample)  # pylint: disable=protected-access
    np.testing.assert_array_equal(norms_new, norms_original)
    np.testing.assert_array_equal(distance_new, distance_original)
