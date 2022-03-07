"""Test callbacks."""
import pathlib

import anndata
import numpy as np
import pytest
import scanpy as sc
import scipy
import tensorflow as tf
from tensorflow import config as tfconfig
from tensorflow.keras import callbacks as tf_callbacks

from discern import io
from discern.estimators import callbacks

_MODEL_INPUT_SHAPE: int = 20


@pytest.fixture
def tensorflow_model():
    """Create a tensorflow model which has is similar to discern."""
    def _encoder():
        encoder_input1 = tf.keras.layers.Input(_MODEL_INPUT_SHAPE,
                                               name="encoder_input")
        encoder_input2 = tf.keras.layers.Input(2, name="encoder_labels")
        encoder_dense1 = tf.keras.layers.Dense(_MODEL_INPUT_SHAPE)(
            encoder_input1)
        encoder_dense2 = tf.keras.layers.Dense(_MODEL_INPUT_SHAPE)(
            encoder_input2)
        return tf.keras.Model(inputs={
            "encoder_input": encoder_input1,
            "encoder_labels": encoder_input2
        },
                              outputs=[encoder_dense1, encoder_dense2],
                              name="encoder")

    def _decoder():
        decoder_input1 = tf.keras.layers.Input(_MODEL_INPUT_SHAPE)
        decoder_input2 = tf.keras.layers.Input(_MODEL_INPUT_SHAPE)
        decoder_input3 = tf.keras.layers.Input(2)
        decoder_dense = tf.keras.layers.Dense(_MODEL_INPUT_SHAPE)(
            decoder_input3)
        decoder_output = tf.keras.layers.Add()(
            [decoder_input1, decoder_input2, decoder_dense])
        return tf.keras.Model(inputs={
            "dec1": decoder_input1,
            "dec2": decoder_input2,
            "dec3": decoder_input3
        },
                              outputs=decoder_output)

    input1 = tf.keras.layers.Input(_MODEL_INPUT_SHAPE, name="input_data")
    input2 = tf.keras.layers.Input(2, name="batch_input_enc")
    input3 = tf.keras.layers.Input(2, name="batch_input_dec")
    encoder = _encoder()
    decoder = _decoder()
    latent, sigma = encoder({
        "encoder_input": input1,
        "encoder_labels": input2
    })
    output = decoder({"dec1": latent, "dec2": sigma, "dec3": input3})
    return tf.keras.Model(inputs={
        'input_data': input1,
        'batch_input_enc': input2,
        'batch_input_dec': input3,
    },
                          outputs=[output, output],
                          name="decoder")


@pytest.mark.slow
@pytest.mark.parametrize("pca", [True, False])
def test_plot_umap(tmp_path, anndata_file, pca):
    """Test _plot_umap function."""
    anndata_file = anndata_file(100)
    for key in anndata_file.obs.columns[anndata_file.obs.dtypes == "category"]:
        anndata_file.obs[key] = anndata_file.obs[key].cat.codes
    anndata_file.obs['testcol'] = ['cond1'] * 50 + ['cond2'] * 50
    anndata_file.obs['testcol2'] = ['cond2'] * 50 + ['cond1'] * 50
    anndata_file.obs['testcol'] = anndata_file.obs['testcol'].astype(
        'category')
    anndata_file.obs['testcol2'] = anndata_file.obs['testcol2'].astype(
        'category')
    files_before = list(tmp_path.rglob('*.png'))
    sc.settings.figdir = tmp_path
    callbacks._plot_umap(  # pylint: disable=protected-access
        anndata_file,
        logdir=tmp_path,
        epoch=15,
        disable_pca=pca)
    files_after = list(tmp_path.rglob('*.png'))
    assert len(files_after) == len(files_before) + 2
    assert tmp_path.joinpath("umap_epoch_15_testcol.png").exists()
    assert tmp_path.joinpath("umap_epoch_15_testcol2.png").exists()


class TestVisualisationCallback:
    """Tests for VisualisationCallback."""

    # pylint: disable=no-self-use
    @pytest.mark.parametrize("batch_size", (10, 25))
    @pytest.mark.parametrize("freq", (None, 15))
    @pytest.mark.parametrize("threads", (0, 1, 2))
    def test_init(self, tmp_path, monkeypatch, anndata_file, batch_size, freq,
                  threads):
        """Test __init__ function."""
        # pylint: disable=too-many-arguments
        monkeypatch.setattr(tfconfig.threading,
                            "get_inter_op_parallelism_threads",
                            lambda: threads)
        monkeypatch.setattr(tfconfig.threading,
                            "get_intra_op_parallelism_threads",
                            lambda: threads)
        anndata_file = anndata_file(10)
        kwargs = dict(outdir=tmp_path,
                      data=anndata_file,
                      batch_size=batch_size)
        if freq:
            kwargs['freq'] = freq
        viz = callbacks.VisualisationCallback(**kwargs)
        assert tmp_path.joinpath('UMAP').exists()
        assert sc.settings.autosave
        if threads > 0:
            assert sc.settings.n_jobs == threads * 2
        # pylint: disable=protected-access
        assert viz._outdir == tmp_path.joinpath("UMAP")
        assert id(viz._initial_data) == id(anndata_file)
        assert viz._batch_size == batch_size
        assert viz._data == {}
        if freq:
            assert viz._freq == freq
        else:
            assert viz._freq == 10
        # pylint: enable=protected-access
    @pytest.mark.parametrize("sparse", (True, False))
    def test_on_train_begin(self, tmp_path, monkeypatch, anndata_file,
                            tensorflow_model, sparse):
        """Test on_train_begin function."""
        # pylint: disable=redefined-outer-name, too-many-arguments
        anndata_file = anndata_file(100)
        original_cells = anndata_file.X.copy()
        if sparse:
            anndata_file.X = scipy.sparse.csr_matrix(anndata_file.X)
        anndata_file.obs.batch = ["cond1"] * 50 + ["cond2"] * 50
        anndata_file.obs.batch = anndata_file.obs.batch.astype('category')
        kwargs = dict(outdir=tmp_path, data=anndata_file, batch_size=10)
        viz = callbacks.VisualisationCallback(**kwargs)
        viz.set_model(tensorflow_model)

        def _check_input(data, logdir, epoch, disable_pca=False):
            assert isinstance(data, anndata.AnnData)
            cells = data.X
            if sparse:
                cells = cells.todense()
            np.testing.assert_allclose(cells, original_cells)
            assert (data.obs == anndata_file.obs).all().all()
            assert isinstance(logdir, pathlib.Path)
            assert isinstance(epoch, int)
            assert isinstance(disable_pca, bool)

        monkeypatch.setattr(callbacks, "_plot_umap", _check_input)

        viz.on_train_begin(None)
        assert tmp_path.joinpath("UMAP", "projected_to_original").exists()
        # pylint: disable=protected-access
        assert set(viz._data.keys()) == set(
            ["cond1", "cond2", "original", "latent"])
        np.testing.assert_allclose(viz._data['original']["input_data"].numpy(),
                                   original_cells)
        np.testing.assert_equal(
            viz._data['original']["batch_input_enc"].numpy().argmax(axis=1),
            anndata_file.obs.batch.cat.codes)
        np.testing.assert_equal(
            viz._data['original']["batch_input_dec"].numpy().argmax(axis=1),
            anndata_file.obs.batch.cat.codes)

        np.testing.assert_allclose(
            viz._data['latent']["encoder_input"].numpy(), original_cells)
        np.testing.assert_equal(
            viz._data['latent']["encoder_labels"].numpy().argmax(axis=1),
            anndata_file.obs.batch.cat.codes)

        np.testing.assert_allclose(viz._data['cond1']["input_data"].numpy(),
                                   original_cells)
        np.testing.assert_equal(
            viz._data['cond1']["batch_input_enc"].numpy().argmax(axis=1),
            anndata_file.obs.batch.cat.codes)
        np.testing.assert_equal(
            viz._data['cond1']["batch_input_dec"].numpy().argmax(axis=1), 0.)

        np.testing.assert_allclose(viz._data['cond2']["input_data"].numpy(),
                                   original_cells)
        np.testing.assert_equal(
            viz._data['cond2']["batch_input_enc"].numpy().argmax(axis=1),
            anndata_file.obs.batch.cat.codes)
        np.testing.assert_equal(
            viz._data['cond2']["batch_input_dec"].numpy().argmax(axis=1), 1.)

        # pylint: enable=protected-access

    @pytest.mark.parametrize("with_celltype", (True, False))
    def test_do_prediction_and_plotting(self, tmp_path, monkeypatch,
                                        anndata_file, tensorflow_model,
                                        with_celltype):
        """Test _do_prediction_and_plotting function."""
        # pylint: disable=redefined-outer-name, too-many-arguments
        anndata_file = anndata_file(100)
        anndata_file = anndata_file[:, 0:_MODEL_INPUT_SHAPE]
        anndata_file.obs.batch = ["cond1"] * 50 + ["cond2"] * 50
        anndata_file.obs.batch = anndata_file.obs.batch.astype('category')
        if with_celltype:
            anndata_file.obs[
                'celltype'] = ["cell1"] * 33 + ["cell2"] * 33 + ["cell3"] * 34
        kwargs = dict(outdir=tmp_path, data=anndata_file, batch_size=10)
        viz = callbacks.VisualisationCallback(**kwargs)
        viz.set_model(tensorflow_model)
        viz.on_train_begin(None)

        def _check_input(data, logdir, epoch, disable_pca=False):
            assert isinstance(data, anndata.AnnData)
            assert isinstance(logdir, pathlib.Path)
            assert isinstance(epoch, int)
            assert isinstance(disable_pca, bool)

        monkeypatch.setattr(callbacks, "_plot_umap", _check_input)
        viz._do_prediction_and_plotting(epoch=10, batch_size=10)  # pylint: disable=protected-access

    @pytest.mark.parametrize("input_epoch", [0, 1, 2, 3, 4, 5, 6])
    def test_on_epoch_end(self, monkeypatch, tmp_path, anndata_file,
                          input_epoch):
        """Test on_epoch_end function."""
        def _check_input(epoch, _):
            assert epoch == input_epoch + 1
            if (input_epoch - 1) % 2 == 0:
                raise AssertionError(
                    "Should not be called due to frequency settings")

        anndata_file = anndata_file(10)
        kwargs = dict(outdir=tmp_path,
                      data=anndata_file,
                      batch_size=10,
                      freq=2)
        viz = callbacks.VisualisationCallback(**kwargs)
        monkeypatch.setattr(viz, "_do_prediction_and_plotting", _check_input)
        viz.on_epoch_end(input_epoch, None)

    def test_on_train_end(self, monkeypatch, tmp_path, anndata_file):
        """Test _on_train_end function."""
        def _check_input(epoch, _):
            assert epoch == "end"

        anndata_file = anndata_file(10)
        kwargs = dict(outdir=tmp_path,
                      data=anndata_file,
                      batch_size=10,
                      freq=2)
        viz = callbacks.VisualisationCallback(**kwargs)
        monkeypatch.setattr(viz, "_do_prediction_and_plotting", _check_input)
        viz.on_train_end(None)


@pytest.mark.parametrize("with_batchsize", [True, False])
@pytest.mark.parametrize("with_umap_cells", [True, False])
def test_create_callbacks(tmp_path, with_batchsize, with_umap_cells,
                          anndata_file):
    """Test create_callbacks function."""

    expected = [
        tf.keras.callbacks.TerminateOnNaN, callbacks.DelayedEarlyStopping,
        tf.keras.callbacks.TensorBoard
    ]

    kwargs = dict(early_stopping_limits=dict(patience=5, min_delta=0.5),
                  exp_folder=tmp_path,
                  profile_batch=2,
                  freq_of_viz=10)
    if with_umap_cells:
        kwargs['umap_cells_no'] = 1000

    if with_batchsize and with_umap_cells:
        kwargs["inputdata"] = io.DISCERNData(anndata_file(100), batch_size=10)
        expected += [callbacks.VisualisationCallback]
    got = callbacks.create_callbacks(**kwargs)
    assert len(got) == len(expected)
    for got_callback, expected_callback in zip(got, expected):
        assert isinstance(got_callback, expected_callback)


class TestDelayedEarlyStopping:
    """Test DelayedEarlyStopping."""
    # pylint: disable=no-self-use
    @pytest.mark.parametrize("delay", [0, 10, 100])
    def test_init(self, delay):
        """Test initalization and default arguments."""
        callback = callbacks.DelayedEarlyStopping(delay=delay)
        ref_callback = tf_callbacks.EarlyStopping()
        assert callback._delay == delay  # pylint: disable=protected-access
        got_attributes = callback.__dict__.copy()
        got_attributes.pop("_delay")
        assert ref_callback.__dict__ == got_attributes

    @pytest.mark.parametrize("delay", [0, 10, 100])
    @pytest.mark.parametrize("patience", [0, 10, 100])
    def test_on_epoch_end(self, delay, patience):
        """Test on_epoch_end function."""
        callback = callbacks.DelayedEarlyStopping(delay=delay,
                                                  patience=patience)
        callback.on_train_begin()
        model = tf.keras.Model()
        model.stop_training = False
        callback.set_model(model)
        for i in range((delay + patience) * 2 + 10):
            callback.on_epoch_end(epoch=i, logs={"val_loss": i})
            if callback.model.stop_training:
                break
        assert i == (delay + max(patience, 1))
