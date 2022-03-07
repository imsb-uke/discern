"""Module containing all losses."""
import importlib
from typing import Any, Dict, Union

import numpy as np
import tensorflow as tf

from discern import functions


def getmembers(
) -> Dict[str, Union[tf.keras.losses.Loss, tf.keras.metrics.Metric]]:
    """Return a dictionary of all custom losses and metrics defined in this module.

    Returns:
        Dict[str, Union[tf.keras.losses.Loss,tf.keras.metrics.Metric]]:
            Name and class of custom losses and metrics.

    """
    return functions.getmembers(__name__)


class Lnorm(tf.keras.losses.Loss):
    """Calculate the Lnorm of input and output.

    Args:
        p (int): Which Lnorm to calculate, for example p=1 means L1-Norm.
        name (str): Description of parameter `name`. Defaults to 'LNorm'.
        reduction (int): Reduction type to use. Defaults to tf.keras.losses.Reduction.AUTO.
        axis (int): Axis on which the norm is calculated. Defaults to 0.
        epsilon (float):Small value to add if (square)root is used.. Defaults to 1e-20.
        use_root (bool): Use (square)root. Defaults to False.

    Attributes:
        pnorm (int): Which Lnorm to calculate, for example p=1 means L1-Norm.
        epsilon (float): Small value to add if (square)root is used.
        axis (int): Axis on which the norm is calculated.
        use_root (bool):  Use (square)root.

    """

    pnorm: int
    epsilon: float
    axis: int
    use_root: bool

    def __init__(self,
                 p: int,
                 name: str = 'LNorm',
                 reduction: str = tf.keras.losses.Reduction.AUTO,
                 axis: int = 0,
                 epsilon: float = 1e-20,
                 use_root: bool = False):
        """Initialize the loss."""
        # pylint: disable=too-many-arguments, invalid-name

        super().__init__(reduction=reduction, name=name)
        self.pnorm = p
        self.epsilon = epsilon
        self.axis = axis
        self.use_root = use_root

    def call(self, y_true, y_pred):
        """Call and returns the loss."""
        norm = tf.abs(y_true - y_pred)
        if self.pnorm == 2:
            norm = tf.square(norm)
        elif self.pnorm > 2:
            norm = tf.math.pow(norm, self.pnorm)
        norm = tf.reduce_sum(norm, axis=self.axis)
        if self.use_root:
            if self.pnorm == 2:
                norm = tf.math.sqrt(norm + self.epsilon)
            elif self.pnorm > 2:
                norm = tf.math.pow(norm + self.epsilon, 1 / self.pnorm)
        return norm

    def get_config(self):
        """Serialize the loss."""
        config = {
            "p": self.pnorm,
            "epsilon": self.epsilon,
            "axis": self.axis,
            "use_root": self.use_root
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HuberLoss(tf.keras.losses.Huber):
    """Huber loss."""

    # pylint: disable=too-few-public-methods

    def call(self, y_true, y_pred):
        """Calculate Huber loss."""
        norm = super().call(y_true, y_pred)
        norm = tf.reduce_mean(norm, axis=-1)
        return norm


class MaskedCrossEntropy(tf.keras.losses.BinaryCrossentropy):
    """Categorical crossentropy Loss with creates mask in true data.

    Args:
        zeros (np.ndarray): Value(s) which represent values to be zeros.
        zeros_eps (float): Value to check for approximate matching to `zeros`.

    """
    _zeros: np.ndarray
    _zeros_eps: float

    def __init__(self,
                 zeros: np.ndarray,
                 zeros_eps: float = 1e-6,
                 lower_label_smoothing: float = 0.0,
                 **kwargs):
        self._zeros = np.array(zeros).reshape((1, -1)).astype(np.float32)
        self._zeros_eps = float(zeros_eps)
        self._masking_invert = 1.0 + lower_label_smoothing
        super().__init__(**kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Call of the loss."""
        upper_mask = y_true <= (self._zeros + self._zeros_eps)
        lower_mask = y_true >= (self._zeros - self._zeros_eps)
        mask = tf.cast(tf.logical_and(upper_mask, lower_mask), tf.float32)
        mask = self._masking_invert - mask
        mask = tf.math.minimum(mask, 1.0)
        return super().call(mask, y_pred)

    def get_config(self):
        """Return the configuration of the loss."""
        config = {
            "zeros": self._zeros.tolist(),
            "zeros_eps": self._zeros_eps,
            "lower_label_smoothing": self._masking_invert - 1
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def reconstruction_loss(loss_type: Dict[str, Any]) -> tf.Tensor:
    """Generate different loss classes based on dictionary.

    Args:
        loss_type (Dict[str, Any]): Dictionary with name as classname of the
            loss and all parameter to be set.

    Returns:
        tf.Tensor: Calculated loss (object)

    Raises:
        KeyError: When the loss name is not supported.

    """
    lossname = loss_type.pop('name')
    module_name, _, func_name = lossname.rpartition(".")
    if module_name:
        module = importlib.import_module(module_name)
        loss = getattr(module, func_name)
    loss = globals()[func_name]
    return loss(**loss_type)


class DummyLoss(tf.keras.losses.Loss):
    """Dummy loss simpy passing the input y_pred as loss output.

    Args:
        reduction (int): Reduction type to use. Defaults to tf.keras.losses.Reduction.AUTO.
        name (str): Name of the loss. Defaults to 'Dummy'.

    """

    # pylint: disable=too-few-public-methods
    def __init__(self,
                 reduction: int = tf.keras.losses.Reduction.AUTO,
                 name: str = 'Dummy'):
        """Initialize dummy loss."""
        super().__init__(reduction=reduction, name=name)

    @staticmethod
    def call(y_true, y_pred):
        # pylint: disable=unused-argument
        """Call the loss and returns the predicted value."""
        return y_pred
