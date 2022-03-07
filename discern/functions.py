"""Module containing diverse TensorFlow related functions."""
import importlib
import inspect
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow import config as tfconfig


def getmembers(name: str) -> Dict[str, Any]:
    """Return a dictionary of all classes defined in this module.

    Args:
        name (str): Name of the module. Usually __name__.

    Returns:
        Dict[str,Any]:
            Name and class of module.

    """
    clsmembers = {
        name: classobj
        for name, classobj in inspect.getmembers(sys.modules[name],
                                                 inspect.isclass)
        if issubclass(classobj, tf.keras.losses.Loss)
        or issubclass(classobj, tf.keras.metrics.Metric)
        or issubclass(classobj, tf.keras.layers.Layer)
        or issubclass(classobj, tf.keras.optimizers.Optimizer)
    }
    return clsmembers


def get_function_by_name(func_str: str) -> Callable[..., Any]:
    """Get a function by its name.

    Args:
        func_str (str): Name of function including module,
            like 'tensorflow.nn.softplus'.

    Returns:
        Callable[Any]: the function.

    Raises:
        KeyError: Function does not exists.

    """
    module_name, _, func_name = func_str.rpartition(".")
    if module_name:
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    return globals()[func_name]


def set_gpu_and_threads(n_threads: int, gpus: Optional[List[int]]):
    """Limits CPU and GPU usage.

    Args:
        n_threads (int): Number of threads to use (get splittet to inter- and intra-op threads).
            Can be disabled by feeding 0.
        gpus (List[int]): List of GPUs to use. Use all GPUs by passing None and no GPUs
            by passing an empty list.
    """
    if n_threads is not None:
        if n_threads <= 1:
            intra_op_threads = inter_op_threads = n_threads
        else:
            intra_op_threads = inter_op_threads = n_threads // 2
            if intra_op_threads + inter_op_threads < n_threads:
                intra_op_threads += 1
        tfconfig.threading.set_inter_op_parallelism_threads(inter_op_threads)
        tfconfig.threading.set_intra_op_parallelism_threads(intra_op_threads)
    if gpus is not None:
        physical_devices = tf.config.list_physical_devices('GPU')
        logical_devices = [physical_devices[i] for i in gpus]
        tfconfig.set_visible_devices(logical_devices, 'GPU')


def prepare_train_valid(
        input_tfr: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    """Get all filennames for train and validation files.

    Args:
        input_tfr (pathlib.Path): Name of input directory.

    Returns:
        Tuple[List[pathlib.Path], List[pathlib.Path]]: Add train and validation
            files to seperate lists.
    """
    train_files = input_tfr.joinpath('training.tfrecords_v2')
    valid_files = input_tfr.joinpath('validate.tfrecords_v2')
    return train_files, valid_files


# yapf: disable
def parse_mean_var(features: pd.DataFrame,
                    scalings: Dict[str, Union[str, float]]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Get mean and variance from anndata.var and scaling dict.


    Args:
        features (pd.DataFrame): anndata.var.
        scalings: (Dict[str, Union[str, float]]): Scalings dict.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and variance

    """
    # yapf: enable
    if scalings['mean'] == 'genes':
        mean: np.ndarray = features['mean_scaling'].values.copy()
    else:
        mean = np.array(scalings['mean'], dtype=float)
    if scalings['var'] == 'genes':
        var: np.ndarray = features['var_scaling'].values.copy()
    else:
        var = np.array(scalings['var'], dtype=float)

    var += float(scalings.get('add_to_var', 0.0))
    mean += float(scalings.get('add_to_mean', 0.0))
    return mean, var


def rescale_by_params(
        adata: anndata.AnnData,
        scalings: Dict[str, Union[str, float, int]]) -> anndata.AnnData:
    """Rescale counts by fixed mean and variance (inplace).

    Reverting function scale_by_params.

    Args:
        adata (anndata.AnnData):  Data to be rescaled.
        scalings (Dict[str, Union[str, float, int]]): Mean and scale values used for rescaling.
            Can be numeric or `genes`. `genes` means using precomputed values
            in anndata object like `adata.var['mean_scaling']`
            and `adata.var['var_scaling']`, respectively.

    Raises:
        ValueError: mean and variance not numeric or `genes`.

    Returns:
        anndata.AnnData: The rescaled AnnData object
    """
    mean, var = parse_mean_var(adata.var, scalings)
    adata = scale(adata, mean=-np.divide(mean, var), var=1 / var)
    adata.X[adata.X < 0.0] = 0.0
    return adata


def scale_by_params(adata: anndata.AnnData,
                    scalings: Dict[str, Union[str, float]]) -> anndata.AnnData:
    """Scale counts by fixed mean and variance (inplace).

    Args:
        adata (anndata.AnnData): Data to be scaled.
        scalings (Dict[str, Union[str, float]]): Mean and scale values used for scaling.
            Can be numeric or `genes`. `genes` means using precomputed values
            in anndata object like `adata.var['mean_scaling']`
            and `adata.var['var_scaling']`, respectively.

    Raises:
        ValueError: mean and variance not numeric or `genes`.

    Returns:
        anndata.AnnData: The scaled AnnData object
    """
    mean, var = parse_mean_var(adata.var, scalings)
    return scale(adata, var=var, mean=mean)


def scale(adata: anndata.AnnData,
          mean: Optional[Union[np.ndarray, float]] = None,
          var: Optional[Union[np.ndarray, float]] = None) -> anndata.AnnData:
    """Scale counts by fixed mean and variance.

    Args:
        adata (anndata.AnnData): Data to be scaled.
        mean (Optional[np.ndarray]): Mean for scaling (will be zero-centered). Defaults to None
        var (Optional[np.ndarray]): Variance for scaling (will be rescaled to 1). Defaults to None

    Returns:
        anndata.AnnData: The AnnData file.

    """
    if mean is None:
        mean = adata.var['mean_scaling']
    if var is None:
        var = adata.var['var_scaling']

    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.todense()

    np.divide(adata.X, var, out=adata.X)
    np.subtract(adata.X, np.divide(mean, var), out=adata.X)
    return adata


def sample_counts(counts: np.ndarray,
                  probabilities: np.ndarray,
                  var: Optional[pd.DataFrame] = None,
                  uns: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Sample counts using probabilities.

    Args:
        counts (np.ndarray): Count data.
        probabilities (np.ndarray): Probability of being non-zero.

    Returns:
        np.ndarray: Sampled count data.
    """
    sample = np.random.rand(*counts.shape)
    sampled_counts = counts.copy()
    if uns and "fixed_scaling" in uns:
        mean, var = parse_mean_var(var, uns["fixed_scaling"])
        sampled_counts *= var
        sampled_counts += mean
    sampled_counts[probabilities < sample] = 0.0
    if uns and "fixed_scaling" in uns:
        sampled_counts -= mean
        sampled_counts /= var
    return sampled_counts
