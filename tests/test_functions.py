"""Tests discern.functions."""
import inspect

import numpy as np
import pandas as pd
import pytest
import scipy
import tensorflow as tf
from tensorflow.python.eager import context as tfcontext  # pylint: disable=no-name-in-module

from discern import functions, preprocessing


def test_scale(anndata_file):
    """Test scaling."""
    anndata_file = anndata_file(2000)
    original_counts = anndata_file.X.copy()
    mean = original_counts.mean(axis=0)
    mean_sq = np.square(original_counts).mean(axis=0)
    var = np.sqrt(mean_sq - mean**2)
    var[var < 1e-8] = 1e-8
    anndata_file = functions.scale(anndata_file, mean=mean, var=var)
    np.testing.assert_allclose(anndata_file.X.mean(axis=0), 0., atol=3e-6)
    np.testing.assert_allclose(anndata_file.X.var(axis=0), 1., rtol=1e-4)


def test_rescale(anndata_file):
    """Test rescaling."""
    anndata_file = anndata_file(2000)
    original_counts = anndata_file.X.copy()
    mean = original_counts.mean(axis=0)
    mean_sq = np.square(original_counts).mean(axis=0)
    var = np.sqrt(mean_sq - mean**2)
    var[var < 1e-8] = 1e-8
    anndata_file = functions.scale(anndata_file, mean=mean, var=var)
    anndata_file.var['mean_scaling'] = mean
    anndata_file.var['var_scaling'] = var
    rescaled = functions.rescale_by_params(anndata_file, {
        'mean': 'genes',
        'var': 'genes'
    })
    rescaled_counts = rescaled.X
    np.testing.assert_allclose(rescaled_counts, original_counts, atol=1e-6)


def test_scale_rescale(anndata_file, monkeypatch):
    """Test scaling and rescaling pipeline."""
    anndata_file = anndata_file(1000)
    original_counts = anndata_file.X.copy()
    anndata_file.X = np.expm1(anndata_file.X)

    def _patched_init(self, sc_raw):
        self.sc_raw = sc_raw

    monkeypatch.setattr(preprocessing.WAERecipe, '__init__', _patched_init)
    pipeline = preprocessing.WAERecipe(anndata_file.copy())
    pipeline.projection_pca(pcs=32)
    np.testing.assert_allclose(pipeline.sc_raw.X, anndata_file.X)
    pipeline.sc_raw.X = original_counts.copy()
    pipeline.sc_raw = functions.scale(pipeline.sc_raw)
    rescaled = functions.rescale_by_params(pipeline.sc_raw.copy(), {
        'mean': 'genes',
        'var': 'genes'
    })
    np.testing.assert_allclose(rescaled.var['var_scaling'],
                               pipeline.sc_raw.var['var_scaling'])
    np.testing.assert_allclose(rescaled.var['mean_scaling'],
                               pipeline.sc_raw.var['mean_scaling'])

    rescaled_counts = rescaled.X

    np.testing.assert_allclose(rescaled_counts, original_counts, atol=1e-6)


@pytest.mark.parametrize("kind",
                         ["loss", 'metric', 'layer', "optimizer", None])
def test_getmembers(monkeypatch, kind):
    """Test getmembers function."""
    expected = {
        "loss": tf.keras.losses.Loss,
        "metric": tf.keras.metrics.Metric,
        "layer": tf.keras.layers.Layer,
        "optimizer": tf.keras.optimizers.Optimizer,
    }
    to_return = []
    if kind is not None:
        to_return = [(kind, expected[kind])]
    monkeypatch.setattr(inspect, "getmembers", lambda *_: to_return)
    got = functions.getmembers(__name__)
    assert len(to_return) == len(got)
    if kind:
        assert list(got.keys())[0] == kind
        assert got[kind] == expected[kind]


@pytest.mark.parametrize("n_cpus", range(0, 10))
@pytest.mark.forked
def test_set_gpu_and_threads_cpu_only(n_cpus):
    """Test set_gpu_and_threads function, but cpu only."""
    tf.keras.backend.clear_session()
    tfcontext._context = None  # pylint: disable=protected-access
    tfcontext._create_context()  # pylint: disable=protected-access
    functions.set_gpu_and_threads(n_threads=n_cpus, gpus=None)
    got = tf.config.threading.get_inter_op_parallelism_threads()
    got += tf.config.threading.get_intra_op_parallelism_threads()
    expected = n_cpus if n_cpus != 1 else 2
    assert got == expected
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)


@pytest.mark.parametrize("gpus", [[1], [0, 1, 3], [4], [6]])
@pytest.mark.forked
def test_set_gpu_and_threads_gpu_only(monkeypatch, gpus):
    """Test set_gpu_and_threads function, but gpu only."""
    gpu_list = list(range(5))
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda _: gpu_list)
    monkeypatch.setattr(tf.config, "set_visible_devices", lambda *_: None)
    if max(gpus) > len(gpu_list):
        with pytest.raises(IndexError):
            functions.set_gpu_and_threads(n_threads=None, gpus=gpus)
    else:
        functions.set_gpu_and_threads(n_threads=None, gpus=gpus)


def test_prepare_train_valid(tmp_path):
    """Test prepare_train_valid function."""
    trainfile = tmp_path.joinpath('training.tfrecords_v2')
    validfile = tmp_path.joinpath('validate.tfrecords_v2')
    trainfile.touch()
    validfile.touch()
    got_train, got_valid = functions.prepare_train_valid(tmp_path)
    assert got_train == trainfile
    assert got_valid == validfile


@pytest.mark.parametrize("mean", ['genes', 5.])
@pytest.mark.parametrize("var", ['genes', 5.])
def test_parse_mean_var(anndata_file, mean, var):
    """Test _parse_mean_var function."""
    anndata_obj = anndata_file(100)
    anndata_obj.var['mean_scaling'] = 100.
    anndata_obj.var['var_scaling'] = 100.
    expected_mean = 100. if mean == 'genes' else mean
    expected_var = 100. if var == 'genes' else var
    got_mean, got_var = functions.parse_mean_var(  # pylint: disable=protected-access
        anndata_obj.var,
        scalings=dict(mean=mean, var=var))
    np.testing.assert_equal(got_mean, expected_mean)
    np.testing.assert_equal(got_var, expected_var)
    got_mean, got_var = functions.parse_mean_var(  # pylint: disable=protected-access
        anndata_obj.var,
        scalings=dict(mean=mean, var=var, add_to_mean=2, add_to_var=3))
    np.testing.assert_equal(got_mean, expected_mean + 2)
    np.testing.assert_equal(got_var, expected_var + 3)
    anndata_obj.X = scipy.sparse.csr_matrix(anndata_obj.X)
    got_mean, got_var = functions.parse_mean_var(  # pylint: disable=protected-access
        anndata_obj.var,
        scalings=dict(mean=mean, var=var))
    np.testing.assert_equal(got_mean, expected_mean)
    np.testing.assert_equal(got_var, expected_var)


@pytest.mark.parametrize("mean,var", [
    ('genes', 'genes'),
    ('genes', 1.),
    (0., 'genes'),
    (0., 1.),
])
def test_scale_by_params(anndata_file, mean, var):
    """Test scale_by_params."""
    anndata_file = anndata_file(2000)
    original_counts = anndata_file.X
    mean_val = original_counts.mean(axis=0)
    mean_sq = np.square(original_counts).mean(axis=0)
    var_val = np.sqrt(mean_sq - mean_val**2)
    var_val[var_val < 1e-8] = 1e-8
    anndata_file.var['mean_scaling'] = mean_val
    anndata_file.var['var_scaling'] = var_val
    functions.scale_by_params(anndata_file, scalings=dict(mean=mean, var=var))
    if mean == "genes":
        np.testing.assert_allclose(anndata_file.X.mean(axis=0), 0., atol=3e-6)
    elif var == 'genes':
        np.testing.assert_allclose((anndata_file.X * var_val).mean(axis=0),
                                   mean_val - mean,
                                   atol=3e-6)
    else:
        np.testing.assert_allclose(anndata_file.X.mean(axis=0),
                                   mean_val,
                                   rtol=1.5e-6)
    if var == "genes":
        np.testing.assert_allclose(anndata_file.X.var(axis=0), 1., rtol=1e-4)
    elif mean == "genes":
        np.testing.assert_allclose((anndata_file.X + mean_val).var(axis=0),
                                   var_val**2 / var,
                                   rtol=1e-4)
    else:
        np.testing.assert_allclose(anndata_file.X.var(axis=0),
                                   var_val**2,
                                   rtol=8e-5)


def _compare_types(object1, object2):
    if isinstance(object1, type(object2)):
        return True
    type1 = object1.dtype if isinstance(object1,
                                        (pd.Series,
                                         np.ndarray)) else type(object1)
    type2 = object2.dtype if isinstance(object2,
                                        (pd.Series,
                                         np.ndarray)) else type(object2)
    return type1 == type2


def _compare_nested_dicts(got, expected):
    keys = set(got.keys()).union(set(expected.keys()))
    assert len(got.keys()) == len(keys)
    for key in keys:
        got_value = got[key]
        expected_value = expected[key]
        assert _compare_types(got_value, expected_value)
        if isinstance(got_value, dict):
            _compare_nested_dicts(got_value, expected_value)
        elif isinstance(got_value, (pd.Series, np.ndarray)):
            np.testing.assert_allclose(got_value, expected_value)
        else:
            assert got_value == expected_value
