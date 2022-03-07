"""Custom Keras Layers."""
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import linalg

from discern import functions


def getmembers() -> Dict[str, tf.keras.layers.Layer]:
    """Return a dictionary of all custom layers defined in this module.

    Returns:
        Dict[str, tf.keras.layers.Layer]: Name and class of custom layers.

    """
    return functions.getmembers(__name__)


def condlayernorm(
        input_cells: tf.Tensor,
        labels: tf.Tensor,
        size: int,
        regularization: Optional[Dict[str, Any]] = None) -> tf.Tensor:
    """Create a conditioning layer.

    Args:
        input_cells (tf.Tensor): Input to the laxer
        labels (tf.Tensor): Label for each sample.
        size (int): Size of the output/input

    Returns:
        tf.Tensor: The output of the conditioning layer, with the same size
            as the input and spezified in size.
    """

    if regularization:
        func = functions.get_function_by_name(regularization["name"])
        regularization = func(
            **{k: v
               for k, v in regularization.items() if k != "name"})

    beta = tf.keras.layers.Dense(size,
                                 kernel_initializer='zeros',
                                 bias_initializer=None,
                                 use_bias=False,
                                 activity_regularizer=regularization)(labels)

    return tf.keras.layers.Add()([input_cells, beta])


@tf.keras.utils.register_keras_serializable()
class GaussianReparametrization(tf.keras.layers.Layer):
    """Reparametrization layer using gaussians."""
    def build(self, input_shape: Tuple[tf.Tensor, tf.Tensor]):
        """Build the layer, usually automatically called at first call.

        Args:
            input_shape (Tuple[tf.Tensor, tf.Tensor]): Shape of the inputs. Both should have
                as last dimension the size of the latent space.
        """
        self.latent_dim = input_shape[0][1:]  # pylint: disable=attribute-defined-outside-init

    @staticmethod
    def call(inputs: Tuple[tf.Tensor, tf.Tensor],
             **kwargs: Dict[str, Any]) -> tf.Tensor:
        """Call the layer.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): latent codes and sigmas from encoder
            **kwargs (Dict[str,Any]): Additional attributes, should contain 'training'".

        Returns:
            tf.Tensor: Rescaled latent codes.
        """
        latent, sigmas_enc = inputs

        sampling_noise = tf.random.normal(shape=tf.shape(latent),
                                          name="sample_pz")

        if kwargs["training"]:
            return latent + tf.multiply(sampling_noise, tf.exp(sigmas_enc))
        return latent


@tf.keras.utils.register_keras_serializable()
class SigmaRegularization(tf.keras.layers.Layer):
    """Regularization term to push sigmas near to one."""
    def build(self, input_shape: tf.Tensor):
        """Build the layer, usually automatically called at first call.

        Args:
            input_shape (tf.Tensor): Shape of the input.

        """
        self.latent_dim = input_shape[-1]  # pylint: disable=attribute-defined-outside-init

    def call(self, inputs: tf.Tensor, **kwargs: Dict[str, Any]) -> tf.Tensor:
        # pylint: disable=unused-argument
        """Call the layer.

        Args:
            inputs (tf.Tensor): Inputs to layer consisting of sigma values.

        Returns:
            tf.Tensor: Regularization loss

        """
        abs_sigma = tf.abs(inputs)
        loss = self.latent_dim * tf.reduce_mean(abs_sigma, axis=1)
        return loss


@tf.keras.utils.register_keras_serializable()
class MMDPP(tf.keras.layers.Layer):
    """mmdpp penalty calculation in keras layer.

    Args:
        scale (float): Value used to scale the output.

    Attributes:
        scale (float): Value used to scale the output.

    """

    scale: float

    def __init__(self, scale: float, **kwargs):
        """Initialize the class and set scale value."""
        super().__init__(**kwargs)
        self.scale = scale

    def build(self, input_shape: Tuple[tf.Tensor, tf.Tensor]):
        """Build the layer, usually automatically called at first call.

        Args:
            input_shape (Tuple[tf.Tensor, tf.Tensor]): Shape of the inputs.
                Both shapes should have the size of the latent space as last dimension.

        """
        self.latent_dim = input_shape[0][-1]  # pylint: disable=attribute-defined-outside-init

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             **kwargs: Dict[str, Any]) -> tf.Tensor:
        # pylint: disable=unused-argument
        """Call the layer.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): The latent codes and sigma values from encoder.

        Returns:
            tf.Tensor: mmdpp penalty loss.

        """
        latent, sigma = inputs
        shape = tf.shape(latent)
        sample_pz = tf.random.normal(mean=0., stddev=1., shape=shape)
        loss = mmdpp_penalty(sample_qz=latent,
                             sample_pz=sample_pz,
                             encoder_sigma=sigma,
                             total_number_cells=self.scale,
                             latent_dim=self.latent_dim)
        return tf.repeat(loss, tf.shape(latent)[0])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration of the layer. Used for serialization.

        Returns:
            Dict[str,Any]: Configuration of the layer

        """
        config = {'scale': self.scale}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _block_diagonal(batch_size: tf.Tensor, num_sample: tf.Tensor) -> tf.Tensor:
    first_idx = tf.repeat(tf.range(num_sample * batch_size), num_sample)
    second_idx = tf.reshape(
        tf.repeat(tf.reshape(tf.range(num_sample * batch_size),
                             (-1, num_sample)),
                  num_sample,
                  axis=0), (-1, ))
    idx = tf.transpose(
        tf.reshape(tf.concat([first_idx, second_idx], axis=0), (2, -1)))

    updates = tf.ones(num_sample * num_sample * batch_size)
    return tf.scatter_nd(idx, updates,
                         (batch_size * num_sample, batch_size * num_sample))


def _eye(dimension: tf.Tensor, name: str) -> tf.Tensor:
    ones = tf.ones(dimension, dtype=tf.float32)
    diagonal_matrix = tf.zeros((dimension, dimension),
                               dtype=tf.float32,
                               name=name)
    return linalg.set_diag(diagonal_matrix, ones)


def _distance(points):
    norms = tf.reduce_sum(input_tensor=tf.square(points),
                          axis=1,
                          keepdims=True)
    dotprods = tf.matmul(points, points, transpose_b=True)
    return norms, norms + tf.transpose(a=norms) - 2. * dotprods


def mmdpp_penalty(sample_qz: tf.Tensor, sample_pz: tf.Tensor,
                  encoder_sigma: tf.Tensor, total_number_cells: float,
                  latent_dim: int) -> tf.Tensor:
    # pylint: disable=too-many-locals
    """Calculate the mmdpp penalty.

    Based on https://github.com/tolstikhin/wae/blob/master/improved_wae.py

    Args:
        sample_qz (tf.Tensor): Sample from the aggregated posterior.
        sample_pz (tf.Tensor): Sample from the prior.
        encoder_sigma (tf.Tensor): Sigma values from the random encoder.
        total_number_cells (int): Total number of samples for scaling.
        latent_dim (int): Dimension of the latent space.

    Returns:
        tf.Tensor: mmdpp penalty loss.

    """
    batch_size_dim = tf.squeeze(tf.gather(tf.shape(sample_qz), [0]))
    batch_size = tf.cast(batch_size_dim, tf.float32)

    num_sample_dim = 8
    num_sample = tf.cast(num_sample_dim, tf.float32)

    eps = tf.random.normal(shape=(batch_size_dim * num_sample_dim, latent_dim),
                           mean=0.,
                           stddev=1.,
                           dtype=tf.float32)

    block_var = tf.reshape(tf.tile(tf.exp(encoder_sigma), [1, num_sample_dim]),
                           [-1, latent_dim])
    eps_q = tf.multiply(eps, block_var)

    block_means = tf.reshape(tf.tile(sample_qz, [1, num_sample_dim]),
                             [-1, latent_dim])
    sample_qhat = block_means + eps_q

    norms_pz, distances_pz = _distance(sample_pz)
    norms_qhat, distances_qhat = _distance(sample_qhat)

    dotprods_pz_qhat = tf.matmul(sample_pz, sample_qhat, transpose_b=True)
    distances_pz_qhat = norms_pz + tf.transpose(
        a=norms_qhat) - 2. * dotprods_pz_qhat

    mask = _block_diagonal(batch_size_dim, num_sample_dim)

    cbase = 2. * latent_dim

    eye_matrix = tf.math.subtract(tf.constant(1.),
                                  _eye(batch_size_dim, "first_eye_mmdpp"))
    second_eye = tf.math.subtract(
        mask, _eye(batch_size_dim * num_sample_dim, "second_eye_mmdpp"))
    stat = 0.

    for scale in [.1, .5, 1., 2., 10.]:
        cval = cbase * scale
        res1 = cval / (cval + distances_pz)
        res1 = tf.multiply(res1, eye_matrix)
        res1 = tf.reduce_sum(res1) / (batch_size * batch_size - batch_size)

        res2 = cval / (cval + distances_pz_qhat)
        res2 = tf.reduce_sum(res2) / (batch_size * batch_size) / num_sample

        res3 = cval / (cval + distances_qhat)
        res3 = tf.reduce_sum(tf.multiply(res3, 1. - mask))
        res3 = res3 / (batch_size * (batch_size - 1)) / (num_sample**2)

        res4 = cval / (cval + distances_qhat)
        res4 = tf.multiply(res4, second_eye)
        res4 = tf.reduce_sum(input_tensor=res4) / batch_size / num_sample / (
            num_sample - 1.) / total_number_cells

        stat += (res1 - 2 * res2 + res3 + res4)

    return stat
