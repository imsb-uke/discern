"""Test discern.estimators.batch_integration."""
import json
import pathlib
from contextlib import ExitStack as no_raise

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tensorflow_addons

from discern import io
from discern.estimators import batch_integration
from discern.estimators import losses, utilities_wae


class TestDISCERN:
    """Testclass for DISCERN."""

    # pylint: disable=no-self-use
    def test_from_json(self, parameters):
        """Test model creation from json."""
        # pylint: disable=too-many-locals
        parameters_path = pathlib.Path(parameters)
        with parameters_path.open('r') as file:
            parameters = json.load(file)
        got = batch_integration.DISCERN.from_json(parameters)
        assert got.start_step == 0
        assert got.wae_model is None

    @pytest.mark.parametrize("with_build, with_model, exception",
                             [(True, False, pytest.raises(AttributeError)),
                              (False, True, pytest.raises(AttributeError)),
                              (True, True, no_raise())])
    def test_encoder(self, default_model, with_build, with_model, exception):
        """Test encoder property."""
        tf.keras.backend.clear_session()
        if with_build:
            default_model.build_model(n_genes=100, n_labels=2, scale=0)
        if not with_model:
            default_model.wae_model = None
        with exception:
            got = default_model.encoder
            assert isinstance(got, tf.keras.Model)
            assert got.name == 'encoder'

    @pytest.mark.parametrize("with_build, with_model, exception",
                             [(True, False, pytest.raises(AttributeError)),
                              (False, True, pytest.raises(AttributeError)),
                              (True, True, no_raise())])
    def test_decoder(self, default_model, with_build, with_model, exception):
        """Test decoder property."""
        tf.keras.backend.clear_session()
        if with_build:
            default_model.build_model(n_genes=100, n_labels=2, scale=0)
        if not with_model:
            default_model.wae_model = None
        with exception:
            got = default_model.decoder
            assert isinstance(got, tf.keras.Model)
            assert got.name == 'decoder'

    @pytest.mark.parametrize("is_compiled", [True, False])
    def test_restore_model(self, default_model, monkeypatch, is_compiled):
        """Test restoring of a model."""
        tf.keras.backend.clear_session()

        def patch_load_model_from_directory(directory):
            assert directory == "somedir"
            model = tf.keras.Model()
            if is_compiled:
                model.compile(optimizer='adam', loss='mse')
            return model, 0

        def patch_compile(self, optimizer):
            self.wae_model.compile(optimizer=optimizer, loss='mae')

        monkeypatch.setattr(utilities_wae, "load_model_from_directory",
                            patch_load_model_from_directory)
        monkeypatch.setattr(batch_integration.DISCERN, "get_optimizer",
                            lambda self: tf.keras.optimizers.Adagrad())
        monkeypatch.setattr(batch_integration.DISCERN, "compile", patch_compile)
        default_model.restore_model("somedir")
        model = default_model.wae_model
        if is_compiled:
            assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
            assert model.loss == "mse"
        else:
            assert isinstance(model.optimizer, tf.keras.optimizers.Adagrad)
            assert model.loss == "mae"

    def test_build_model(self, default_model, monkeypatch):
        """Test model building."""
        def patch_create_encoder(latent_dim, enc_layers, enc_norm_type,
                                 activation_fn, input_dim, n_labels,
                                 regularization, conditional_regularization):
            # pylint: disable=too-many-arguments
            assert latent_dim == default_model.latent_dim
            assert enc_layers == default_model.encoder_config.layers
            assert enc_norm_type == default_model.encoder_config.norm_type
            assert activation_fn == default_model.activation_fn
            assert input_dim == 100
            assert n_labels == 2
            assert regularization == default_model.encoder_config.regularization
            assert (conditional_regularization ==
                    default_model.decoder_config.conditional_regularization)
            return "Encoder"

        monkeypatch.setattr(utilities_wae, "create_encoder",
                            patch_create_encoder)

        def patch_create_decoder(latent_dim, output_cells_dim, dec_layers,
                                 dec_norm_type, output_lsn, activation_fn,
                                 output_fn, n_labels, regularization,
                                 conditional_regularization):
            # pylint: disable=too-many-arguments
            assert latent_dim == default_model.latent_dim
            assert output_cells_dim == 100
            assert n_labels == 2
            assert dec_layers == default_model.decoder_config.layers
            assert dec_norm_type == default_model.decoder_config.norm_type
            assert output_lsn == default_model.output_lsn
            assert activation_fn == default_model.activation_fn
            assert output_fn == default_model.output_fn
            assert regularization == default_model.decoder_config.regularization
            assert (conditional_regularization ==
                    default_model.decoder_config.conditional_regularization)
            return "Decoder"

        monkeypatch.setattr(utilities_wae, "create_decoder",
                            patch_create_decoder)

        def patch_create_model(encoder, decoder, total_cells):
            assert encoder == "Encoder"
            assert decoder == "Decoder"
            assert total_cells == 0
            return "Model"

        monkeypatch.setattr(utilities_wae, "create_model", patch_create_model)

        monkeypatch.setattr(batch_integration.DISCERN, "get_optimizer",
                            lambda self: "Optimizer")

        def patch_compile(_, optimizer, scale):
            assert scale == 15000
            assert optimizer == "Optimizer"

        monkeypatch.setattr(batch_integration.DISCERN, "compile", patch_compile)
        default_model.build_model(n_genes=100, n_labels=2, scale=0)
        assert default_model.wae_model == "Model"

    @pytest.mark.parametrize("with_decay", [True, False])
    @pytest.mark.parametrize("with_lookahead", [True, False])
    @pytest.mark.parametrize("algo", ["Adam", 'Adagrad'])
    def test_get_optimizer(self, default_model, with_decay, with_lookahead,
                           algo):
        """Test optimizer creation."""
        algo = "tensorflow.keras.optimizers." + algo
        config = {
            "learning_rate": 0.1,
            "algorithm": algo,
            "epsilon": 1e-08,
        }
        if with_decay:
            config["learning_decay"] = dict(
                name="tensorflow.keras.optimizers.schedules.ExponentialDecay",
                decay_steps=1,
                decay_rate=0.2)
        if with_lookahead:
            config["Lookahead"] = True

        default_model.optimizer_config = config
        got = default_model.get_optimizer()
        if with_lookahead:
            assert isinstance(got,
                              tensorflow_addons.optimizers.lookahead.Lookahead)
            got = got._optimizer  # pylint: disable=protected-access

        if algo.endswith('Adam'):
            assert isinstance(got, tf.keras.optimizers.Adam)
        elif algo.endswith('Adagrad'):
            assert isinstance(got, tf.keras.optimizers.Adagrad)
        else:
            raise AssertionError("Invalid config")
        got = got.get_config()
        assert got['epsilon'] == config['epsilon']

        if with_decay:
            assert got["learning_rate"] == {
                'class_name': 'ExponentialDecay',
                'config': {
                    'decay_rate': 0.2,
                    'decay_steps': 1,
                    'initial_learning_rate': 0.1,
                    'name': None,
                    'staircase': False
                }
            }
        else:
            assert got['learning_rate'] == config['learning_rate']

    def test_compile(self, default_model, monkeypatch):
        """Test compiling model."""
        def patch_reconstruction_loss(losstype):
            assert losstype == default_model.recon_loss_type
            return "mse"

        monkeypatch.setattr(losses, "reconstruction_loss",
                            patch_reconstruction_loss)
        default_model.build_model(n_genes=100, n_labels=2, scale=0)
        default_model.compile("Adam")
        model = default_model.wae_model
        assert model._is_compiled  # pylint: disable=protected-access
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
        assert len(model.loss) == 4
        assert isinstance(model.loss['decoder_dropouts'],
                          losses.MaskedCrossEntropy)
        assert model.loss['decoder_counts'] == "mse"
        assert isinstance(model.loss['sigma_regularization'], losses.DummyLoss)
        assert isinstance(model.loss['mmdpp'], losses.DummyLoss)
        assert model.loss_weights == {
            "decoder_counts": 15000.0,
            "decoder_dropouts": default_model.weighting_decoder_dropout,
            "sigma_regularization": default_model.weighting_random_encoder,
            "mmdpp": default_model.wae_lambda
        }
        assert len(model.metrics) == 0

    @pytest.mark.parametrize("savepath", (True, False))
    def test_training(self, default_model, monkeypatch, savepath):
        """Test training function without performing actual training."""
        exp_batchsize = 10
        exp_maxstep = 1
        monkeypatch.setattr(
            batch_integration._LOGGER,  # pylint: disable=protected-access
            'getEffectiveLevel',
            lambda: 20)

        default_model.build_model(n_genes=100, n_labels=2, scale=0)

        class _PatchDISCERNData:
            def __init__(self):
                traindataset = tf.data.Dataset.from_tensor_slices(np.zeros(10))
                validdataset = tf.data.Dataset.from_tensor_slices(np.ones(10))
                self.tfdata = traindataset, validdataset
                self.batch_size = exp_batchsize
                self.config = {"total_train_count": 10}

        def patch_fit(x, epochs, validation_data, verbose, callbacks,
                      initial_epoch):
            # pylint: disable=too-many-arguments, invalid-name
            assert isinstance(x, tf.data.Dataset)
            for val in x:
                assert val == 0.
            assert isinstance(validation_data, tf.data.Dataset)
            for val in validation_data:
                assert val == 1.
            assert epochs == exp_maxstep
            assert verbose == 1
            assert callbacks == "Callbacks"
            assert initial_epoch == 0.
            return "Result"

        def _check_save(*_, **unused_kwargs):
            assert savepath

        monkeypatch.setattr(default_model.wae_model, "fit", patch_fit)
        monkeypatch.setattr(default_model.wae_model, "save", _check_save)

        got = default_model.training(savepath=savepath if savepath else None,
                                     inputdata=_PatchDISCERNData(),
                                     max_steps=exp_maxstep,
                                     callbacks="Callbacks")
        assert got == "Result"

    def test_generate_latent_codes(self, default_model):
        """Test generation of latent codes."""
        exp_batchsize = 1
        counts = np.random.uniform(0, 4, 20).reshape(10, 2) - 2
        labels = np.ones(10)[:, np.newaxis]
        exp_counts = counts.copy()

        class PatchEncoder:
            """Patch for don't using real encoder."""

            # pylint: disable=too-few-public-methods
            def predict(self, dataset, batch_size):
                """Predict test."""
                assert batch_size == exp_batchsize
                assert len(dataset) == 2
                assert isinstance(dataset['encoder_labels'], tf.Tensor)
                for val in dataset['encoder_labels']:
                    np.testing.assert_allclose(val, np.ones((1, )))
                assert isinstance(dataset['encoder_input'], tf.Tensor)
                for i, val in enumerate(dataset['encoder_input']):
                    np.testing.assert_allclose(val, exp_counts[i])

        class PatchModel:
            """Patch for don't using real model."""

            # pylint: disable=too-few-public-methods
            def get_layer(self, layername):
                """Get layer patched."""
                if layername == "encoder":
                    return PatchEncoder()
                raise AssertionError('Invalid layer')

        default_model.wae_model = PatchModel()
        default_model.generate_latent_codes(counts=counts,
                                            batch_labels=labels,
                                            batch_size=exp_batchsize)

    def test_generate_cells_from_latent(self, default_model):
        """Test generation of cells."""
        exp_batchsize = 1
        latent = np.random.rand(10, 2)
        labels = np.ones(10)[:, np.newaxis]

        class PatchDecoder:
            """Patch for don't using real decoder."""

            # pylint: disable=too-few-public-methods
            def predict(self, dataset, batch_size):
                """Predict test."""
                assert batch_size == exp_batchsize
                assert len(dataset) == 2
                assert isinstance(dataset['decoder_labels'], tf.Tensor)
                for val in dataset['decoder_labels']:
                    np.testing.assert_allclose(val, np.ones((1, )))
                assert isinstance(dataset['decoder_input'], tf.Tensor)
                for i, val in enumerate(dataset['decoder_input']):
                    np.testing.assert_allclose(val, latent[i])

        class PatchModel:
            """Patch for don't using real model."""

            # pylint: disable=too-few-public-methods
            def get_layer(self, layername):
                """Get layer patched."""
                if layername == "decoder":
                    return PatchDecoder()
                raise AssertionError('Invalid layer')

        default_model.wae_model = PatchModel()
        default_model.generate_cells_from_latent(latent_codes=latent,
                                                 output_batch_labels=labels,
                                                 batch_size=exp_batchsize)

    @pytest.mark.parametrize("inputs",
                             [
                                 dict(metadata=[("batch", "batch1"),
                                                ("batch", "batch2"),
                                                ("batch", None),
                                                ("metadata", "type1"),
                                                ("metadata", "type2"),
                                                ("metadata", None)],
                                      is_scaled=False,
                                      exception=no_raise(),
                                      exp_frequencies=[
                                          {
                                              "batch1": [1., 0.],
                                              "batch2": [1., 0.],
                                          },
                                          {
                                              "batch1": [0., 1.],
                                              "batch2": [0., 1.],
                                          },
                                          {
                                              "batch1": [0.4, 0.6],
                                              "batch2": [0.4, 0.6],
                                          },
                                          {
                                              "type1": [1.0, 0.0],
                                              "type2": [1.0, 0.0],
                                          },
                                          {
                                              "type1": [0.25, 0.75],
                                              "type2": [0.25, 0.75],
                                          },
                                          {
                                              "type1": [1., 0.0],
                                              "type2": [0.25, 0.75],
                                          },
                                      ]),
                                 dict(metadata=[("batch", "batch1"),
                                                ("batch", "batch2"),
                                                ("batch", None),
                                                ("metadata", "type1"),
                                                ("metadata", "type2"),
                                                ("metadata", None)],
                                      is_scaled=True,
                                      exception=no_raise(),
                                      exp_frequencies=[
                                          {
                                              "batch1": [1., 0.],
                                              "batch2": [1., 0.],
                                          },
                                          {
                                              "batch1": [0., 1.],
                                              "batch2": [0., 1.],
                                          },
                                          {
                                              "batch1": [0.4, 0.6],
                                              "batch2": [0.4, 0.6],
                                          },
                                          {
                                              "type1": [1.0, 0.0],
                                              "type2": [1.0, 0.0],
                                          },
                                          {
                                              "type1": [0.25, 0.75],
                                              "type2": [0.25, 0.75],
                                          },
                                          {
                                              "type1": [1., 0.0],
                                              "type2": [0.25, 0.75],
                                          },
                                      ]),
                                 dict(metadata=[("batch", )],
                                      is_scaled=False,
                                      exception=pytest.raises(ValueError),
                                      exp_frequencies=[]),
                                 dict(metadata=[("invalid_column", "batch1")],
                                      is_scaled=False,
                                      exception=pytest.raises(KeyError),
                                      exp_frequencies=[]),
                                 dict(metadata=[("metadata", "invalid_value")],
                                      is_scaled=False,
                                      exception=pytest.raises(ValueError),
                                      exp_frequencies=[]),
                             ])
    def test_project_to_metadata(self, monkeypatch, tmp_path, anndata_file,
                                 default_model, inputs):
        """Test project_to_metadata."""
        # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
        exp_batchsize = 10
        anndata_file = io.DISCERNData(anndata_file(100), batch_size=10)
        anndata_file.uns.pop("fixed_scaling", None)
        exp_threshold = 0.0
        if inputs["is_scaled"]:
            exp_threshold = -np.inf
            anndata_file.uns["fixed_scaling"] = {}
        anndata_file.obs["metadata"] = ["type1"] * 20 + ["type2"] * 80
        batches = ["batch1"] * 20 + ["batch2"] * 30
        batches += ["batch1"] * 20 + ["batch2"] * 30
        anndata_file.obs["batch"] = batches
        anndata_file.obs["batch"] = anndata_file.obs.batch.astype("category")

        default_model.build_model(n_genes=100, n_labels=2, scale=0)

        def patch_generate_latent_codes(data, labels, batchsize):
            np.testing.assert_equal(data, anndata_file.X)
            assert labels.shape == (100, 2)
            labels = labels.argmax(axis=1)
            assert (labels == anndata_file.obs["batch"].cat.codes).all()
            assert batchsize == exp_batchsize
            return "latent", None

        monkeypatch.setattr(default_model, "generate_latent_codes",
                            patch_generate_latent_codes)

        metadata = inputs["metadata"].copy()
        second_metadata_check = metadata.copy()

        exp_frequencies = inputs.pop("exp_frequencies")

        def patch_generate_cells_from_latent(latent, labels, batchsize):
            assert latent == "latent"
            assert batchsize == exp_batchsize
            curr_col, curr_val = metadata.pop(0)
            if curr_val == "invalid_value":
                return (curr_col, curr_val)
            got_freq = pd.DataFrame(
                labels, columns=anndata_file.obs.batch.cat.categories)
            got_freq[curr_col] = anndata_file.obs[curr_col].reset_index(
                drop=True)
            got_freq.drop_duplicates(inplace=True)
            got_freq.set_index(curr_col, inplace=True)
            exp_freq = pd.DataFrame.from_dict(
                exp_frequencies.pop(0),
                orient="index",
                columns=anndata_file.obs.batch.cat.categories)
            pd.testing.assert_frame_equal(got_freq,
                                          exp_freq,
                                          check_index_type=False,
                                          check_column_type=False,
                                          check_categorical=False,
                                          check_dtype=False,
                                          check_names=False)
            return (curr_col, curr_val)

        monkeypatch.setattr(default_model, "generate_cells_from_latent",
                            patch_generate_cells_from_latent)

        def patch_generate_h5ad(counts, threshold, save_path, var, obs, uns,
                                obsm):
            # pylint: disable=too-many-arguments
            assert threshold == exp_threshold
            assert (var == anndata_file.var).all(axis=None)
            assert (obs == anndata_file.obs).all(axis=None)
            assert uns == anndata_file.uns
            assert obsm["X_DISCERN"] == "latent"
            curr_metadata = second_metadata_check.pop(0)
            assert counts == curr_metadata
            exp_save_path = str(
                pathlib.Path(tmp_path,
                             "projected_to_average_{}".format(counts[0])))
            if counts[1]:
                exp_save_path += "_{}".format(counts[1])
            exp_save_path += ".h5ad"
            assert save_path == pathlib.Path(exp_save_path)

        monkeypatch.setattr(io, "generate_h5ad", patch_generate_h5ad)

        with inputs["exception"]:
            default_model.project_to_metadata(input_data=anndata_file,
                                              metadata=inputs["metadata"],
                                              save_path=tmp_path)
            assert len(metadata) == 0
            assert len(second_metadata_check) == 0
