"""Test functions for online learning."""
import itertools
from contextlib import ExitStack as no_raise
import json

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from discern import preprocessing, io
from discern.estimators import run_exp, batch_integration
import discern.online_learning as discern


@pytest.mark.parametrize("old_weights, new_weights, expected", [
    (([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
     ([[1, 1], [1, 1]], [[1, 1], [1, 1]]),
     ([[0, 0], [0, 0]], [[0, 0], [0, 0]])),
    (([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
     ([[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]),
     ([[0, 0], [0, 0], [1, 1]], [[0, 0], [0, 0], [1, 1]])),
])
def test_update_weights(old_weights, new_weights, expected):
    """Test weight update function."""

    old_weights = list(map(np.array, old_weights))
    new_weights = list(map(np.array, new_weights))

    changed = any(
        (x.shape != y.shape for x, y in zip(old_weights, new_weights)))

    got_weights, got_changed = discern._update_weights(  # pylint: disable=protected-access
        old_weights=old_weights,
        new_weights=new_weights)
    np.testing.assert_equal(got_weights, expected)
    assert got_changed == changed


def _create_model(shapes, inputs):
    inputs = tf.keras.Input(inputs)
    dense = inputs
    for shape in shapes:
        if isinstance(shape, int):
            dense = tf.keras.layers.Dense(shape)(dense)
        else:
            dense = _create_model(shape, dense.shape[1])(dense)
    return tf.keras.Model(inputs=inputs, outputs=dense)


def _chain_layers(model: tf.keras.Model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sublayer in _chain_layers(layer):
                yield sublayer
        else:
            yield layer


@pytest.mark.parametrize("old_model, new_model, freeze, expected", [
    ((10, 6, 1), (11, 6, 1), False, (True, True, True)),
    ((10, 6, 1), (11, 6, 1), True, (False, True, False)),
    ((10, (6, 5), 1), (11, (6, 5), 1), False, (True, True, True, True, True)),
    ((10, (6, 5), 1), (11, (6, 5), 1), True,
     (False, False, True, False, False)),
])
def test_update_model(old_model, new_model, freeze, expected):
    """ Test model weight update."""
    old_model = _create_model(old_model[1:], old_model[0])
    expected_model = _create_model(new_model[1:], new_model[0])
    new_model = _create_model(new_model[1:], new_model[0])

    got_model = discern.update_model(old_model,
                                   new_model,
                                   freeze_unchanged=freeze)
    got_layers = list(_chain_layers(got_model))
    try:
        assert len(got_layers) == len(expected)
        trainable = [layer.trainable for layer in got_layers]
        np.testing.assert_equal(trainable, expected)
        for expected_weight, got in zip(expected_model.get_weights(),
                                        got_model.get_weights()):
            assert got.shape == expected_weight.shape
    except AssertionError as exception:
        got_model.summary()
        raise AssertionError from exception


class TestOnlineWAERecipe:
    """Testclass for OnlineWAERecipe."""
    # pylint: disable=no-self-use
    @pytest.mark.parametrize(
        "kwargs, exception",
        [
            ({
                "params": {
                    "testparams": True
                }
            },
             pytest.raises(
                 AssertionError,
                 match="Either `inputs` or `input_files` need to be provided.")
             ),
            ({
                "input_files": ["somefile.h5ad"]
            },
             pytest.raises(
                 TypeError,
                 match=
                 r"__init__\(\) missing 1 required positional argument: 'params'"
             )),
            ({
                "input_files": ["somefile.h5ad"],
                "params": {
                    "testparams": True
                }
            }, no_raise()),
            ({
                "input_files": ["somefile.h5ad"],
                "params": {
                    "testparams": True
                },
                "n_jobs": 2
            }, no_raise()),
        ],
    )
    def test_init(self, anndata_file, tmp_path, kwargs, exception):
        """ Test init."""
        data = anndata_file(10)
        if "input_files" in kwargs:
            for i, file in enumerate(kwargs["input_files"]):
                file = tmp_path.joinpath(file)
                data.write(file)
                kwargs["input_files"][i] = str(file)

        with exception as excinfo:
            obj = discern.OnlineWAERecipe(data, **kwargs)

        if not isinstance(excinfo, no_raise):
            return
        assert id(obj._reference) == id(data)  # pylint: disable=protected-access
        assert obj._n_jobs == kwargs.get("n_jobs", -1)  # pylint: disable=protected-access
        assert obj.params == kwargs["params"]
        assert isinstance(obj.sc_raw, type(data))

    @pytest.mark.parametrize("min_genes", (10, 20, 100))
    def test_filtering(self, anndata_file, tmp_path, min_genes):
        """Test filtering."""
        reference = anndata_file(40)
        genes = reference.var.sample(frac=0.9).index.to_series()
        reference = reference[:, genes].copy()
        query = anndata_file(30)
        query.X += 1.0
        dropped_cells = np.random.randint(5, 20, 1)[0]
        dropped_genes = genes.sample(len(genes) - min_genes + 1)
        query[-dropped_cells:, dropped_genes].X = 0.0
        inputs = tmp_path.joinpath("testfile.h5ad")
        query.write(inputs)
        obj = discern.OnlineWAERecipe(reference,
                                    input_files=[inputs],
                                    n_jobs=1,
                                    params={})
        obj.filtering(min_genes=min_genes)
        pd.testing.assert_index_equal(obj.sc_raw.var_names,
                                      reference.var_names)
        assert obj.sc_raw.X.shape[0] == 30 - dropped_cells

    @pytest.mark.parametrize("n_pcs, exception", (
        (10, no_raise()),
        (20, pytest.raises(ValueError, match="expected 10")),
        (5, pytest.raises(ValueError, match="expected 10")),
    ))
    def test_projection_pca(self, anndata_file, tmp_path, n_pcs, exception):
        """Test projection_pca."""
        reference = anndata_file(10)
        inputs = tmp_path.joinpath("testfile.h5ad")
        reference.write(inputs)
        n_genes = reference.n_vars
        reference.var["mean_scaling"] = 15.0
        reference.var["var_scaling"] = 3.0
        reference.var['pca_genes'] = np.random.choice([True, False],
                                                      size=n_genes)
        reference.varm['PCs'] = np.zeros((n_genes, 10))
        reference.varm['PCs'][reference.var['pca_genes']] = 1.0
        obj = discern.OnlineWAERecipe(reference,
                                    input_files=[inputs],
                                    n_jobs=1,
                                    params={})
        assert obj.sc_raw.shape == reference.shape
        with exception as excinfo:
            obj.projection_pca(pcs=n_pcs)
        if not isinstance(excinfo, no_raise):
            return
        np.testing.assert_equal(obj.sc_raw.varm['PCs'], reference.varm['PCs'])
        pd.testing.assert_series_equal(obj.sc_raw.var['mean_scaling'],
                                       reference.var['mean_scaling'])
        pd.testing.assert_series_equal(obj.sc_raw.var['var_scaling'],
                                       reference.var['var_scaling'])
        pd.testing.assert_series_equal(obj.sc_raw.var['pca_genes'],
                                       reference.var['pca_genes'])

    @pytest.mark.parametrize("n_batches_new", (0, 1, 2))
    @pytest.mark.parametrize("n_batches_old", (0, 1, 2))
    def test_fix_batch_labels(self, n_batches_new, n_batches_old, anndata_file,
                              tmp_path):
        """Test fix_batch_labels."""
        if n_batches_new == 0 and n_batches_old == 0:
            pytest.skip("Cannot be run without any batch")
        reference = anndata_file(10)
        n_old_labels = len(reference.obs.batch.cat.categories)
        query = reference.copy()
        old_batches = np.random.choice(reference.obs.batch.cat.categories,
                                       size=n_batches_old,
                                       replace=False)
        batches = {"new_batch" + str(i) for i in range(1, n_batches_new + 1)}
        batches = list(batches.union(old_batches))
        query.obs.batch = pd.Categorical(
            itertools.islice(itertools.cycle(batches), query.obs_names.size))

        inputs = tmp_path.joinpath("testfile.h5ad")
        query.write(inputs)
        obj = discern.OnlineWAERecipe(reference,
                                    input_files=[inputs],
                                    n_jobs=1,
                                    params={})
        obj.fix_batch_labels()

        new_idx = obj.sc_raw.obs.batch.str.startswith("new_batch")
        old_batch_labels = obj.sc_raw.obs.batch[~new_idx].cat.codes
        new_batch_labels = obj.sc_raw.obs.batch[new_idx].cat.codes
        assert len(obj.sc_raw.obs.batch.cat.categories
                   ) == n_old_labels + n_batches_new
        assert len(np.unique(old_batch_labels)) == n_batches_old
        assert len(np.unique(new_batch_labels)) == n_batches_new
        if n_batches_old > 0:
            assert old_batch_labels.min() >= 0
            assert old_batch_labels.max() <= n_old_labels
        if n_batches_new > 0:
            assert new_batch_labels.min() >= n_old_labels

    @pytest.mark.parametrize("with_fixed_scaling", [True, False])
    @pytest.mark.parametrize("mean", [10.0])
    @pytest.mark.parametrize("var", [2.0])
    def test_mean_var_scaling(self, mean, var, anndata_file,
                              with_fixed_scaling, tmp_path):
        """Test mean var scaling."""
        reference = anndata_file(10)
        inputs = tmp_path.joinpath("testfile.h5ad")
        reference.uns.pop("fixed_scaling", None)
        reference.X = np.ones_like(reference.X)
        reference.var["mean_scaling"] = mean
        reference.var["var_scaling"] = var
        reference.write(inputs)

        if with_fixed_scaling:
            reference.uns['fixed_scaling'] = dict(var='genes', mean='genes')

        obj = discern.OnlineWAERecipe(reference,
                                    input_files=[inputs],
                                    n_jobs=1,
                                    params={})
        obj.mean_var_scaling()
        got = obj.sc_raw
        assert got.shape == reference.shape

        expected = 1.0
        if with_fixed_scaling:
            expected = (1.0 - mean) / var
            assert got.uns['fixed_scaling'] == dict(var='genes', mean='genes')
        np.testing.assert_allclose(got.X, expected)

    def test_call(self, anndata_file, tmp_path, monkeypatch):
        """Test call."""
        call_order = []
        adata = anndata_file(10)
        inputs = tmp_path.joinpath("testfile.h5ad")
        adata.write(inputs)

        monkeypatch.setattr(preprocessing.WAERecipe, "__call__",
                            lambda *_: call_order.append("call"))
        monkeypatch.setattr(discern.OnlineWAERecipe, "fix_batch_labels",
                            lambda *_: call_order.append("batch"))

        obj = discern.OnlineWAERecipe(adata,
                                    input_files=[inputs],
                                    n_jobs=1,
                                    params={})
        obj()
        assert call_order == ["call", "batch"]


@pytest.mark.parametrize("n_runs", [1, 3])
def test_save_data(tmp_path, anndata_file, n_runs):
    """Test save_data"""
    adata = io.DISCERNData(anndata_file(10), batch_size=10)
    path = tmp_path.joinpath("concatenated_data.h5ad")
    ref = anndata_file(100)
    ref.write(path)
    for _ in range(n_runs):
        old_data = io.DISCERNData.read_h5ad(path, batch_size=10)
        discern.save_data(file=path, data=adata, old_data=old_data)
    got = old_data = io.DISCERNData.read_h5ad(path, batch_size=10)
    pd.testing.assert_index_equal(ref.obs_names, got.obs_names[0:100])
    pd.testing.assert_index_equal(ref.var_names, got.var_names)
    pd.testing.assert_series_equal(ref.obs.batch, got.obs.batch.iloc[0:100])
    assert got.shape[0] == 100 + (n_runs * 10)


@pytest.mark.parametrize("expect_freeze", [True, False])
def test_online_training(parameters, monkeypatch, expect_freeze, default_model,
                         anndata_file):
    """Test online_learning."""
    expect_exp_folder = parameters.parent
    anndatafile = expect_exp_folder.joinpath("processed_data",
                                             "concatenated_data.h5ad")
    expect_exp_folder.joinpath("job").mkdir(exist_ok=True)
    anndatafile.parent.mkdir()
    anndata_file(120).write(anndatafile)
    expect_filename = expect_exp_folder.joinpath("file.h5ad")
    anndata_file(101).write(expect_filename)

    def _check_setup_experiment(exp_folder):
        assert exp_folder == expect_exp_folder
        params = json.loads(parameters.read_bytes())
        return default_model, params

    monkeypatch.setattr(run_exp, "setup_exp", _check_setup_experiment)

    monkeypatch.setattr(discern.OnlineWAERecipe, "projection_pca",
                        lambda *_, **unused: None)

    monkeypatch.setattr(discern, "save_data", lambda *_, **unused: None)

    monkeypatch.setattr(discern.OnlineWAERecipe, "kernel_mmd",
                        lambda *_, **unused: None)

    def _check_restore_model(self, folder):
        assert folder == expect_exp_folder.joinpath("job")
        default_model.build_model(15289, 2, 4)
        self.wae_model = default_model.wae_model
        default_model.wae_model = None
        default_model.recon_loss_type["name"] = "Lnorm"
        tf.keras.backend.clear_session()

    monkeypatch.setattr(batch_integration.DISCERN, "restore_model",
                        _check_restore_model)

    def _check_train(model, exp_folder, inputdata, early_stopping, max_steps):
        assert isinstance(model, batch_integration.DISCERN)
        assert exp_folder == expect_exp_folder
        assert isinstance(inputdata, io.DISCERNData)
        assert inputdata._batch_size == 128  # pylint: disable=protected-access
        assert early_stopping == {
            'patience': 20,
            'min_delta': 0.01,
            'mode': 'auto',
            'monitor': 'val_loss',
            'restore_best_weights': True
        }
        assert max_steps == 100

    monkeypatch.setattr(run_exp, "_train", _check_train)

    discern.online_training(exp_folder=expect_exp_folder,
                          filename=expect_filename,
                          freeze=expect_freeze)
    assert expect_exp_folder.joinpath("backup").exists()
    assert expect_exp_folder.joinpath("job").exists()
