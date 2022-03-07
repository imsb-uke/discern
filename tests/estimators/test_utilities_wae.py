"""Test discern.estimatiors.utilities_wae.py."""
import collections
import json
import pathlib
from contextlib import ExitStack as no_raise

import h5py
import numpy as np
import pytest
import tensorflow as tf

from discern.estimators import customlayers, utilities_wae


@pytest.fixture(name='savedmodel')
def tensorflow_savedmodel(tmp_path, default_model):
    """Create and save a Model."""
    modelfile = pathlib.Path(tmp_path, 'model.h5')

    def _generate_model():
        default_model.build_model(n_genes=100, n_labels=2, scale=0)
        model = default_model.wae_model
        model.fit(x={
            'input_data':
            tf.zeros((10, model.input_shape["input_data"][1])),
            "batch_input_enc":
            tf.zeros((10, model.input_shape["batch_input_enc"][1])),
            "batch_input_dec":
            tf.zeros((10, model.input_shape["batch_input_dec"][1]))
        },
                  y=(tf.zeros((10, model.input_shape["input_data"][1])),
                     tf.zeros((10, model.input_shape["input_data"][1]))),
                  verbose=0,
                  epochs=1,
                  batch_size=5)
        model.save(str(modelfile))
        return model

    return _generate_model, modelfile


@pytest.mark.parametrize("case", ["normal", "failing", "wo_optimizer"])
def test_load_step_from_hd5file(savedmodel, case):
    """Test model step loading from hdf5-file."""
    model, modelfile = savedmodel
    model = model()
    if case == "failing":
        with h5py.File(modelfile, 'a') as file:
            del file["optimizer_weights"]
        expected = 0
    elif case == "wo_optimizer":
        with h5py.File(modelfile, 'a') as file:
            file["optimizer_weights"]["training"]["iter:0"] = file[
                "optimizer_weights"]["training"]["Adam"]["iter:0"]
            del file["optimizer_weights"]["training"]["Adam"]
        expected = model.optimizer.iterations

    else:
        expected = model.optimizer.iterations

    got = utilities_wae._load_step_from_hd5file(modelfile)  # pylint: disable=protected-access
    assert got == expected


@pytest.mark.parametrize("case", [
    pytest.param("savedmodel", marks=pytest.mark.slow), "dummymodel",
    "uncompiled", "no_model"
])
def test_load_model_from_directory(savedmodel, case):
    """Test model loading."""
    modelfile = savedmodel[1]
    expected_step = 0
    if case == "savedmodel":
        expected_model = savedmodel[0]()
        modelfile.rename(modelfile.parent.joinpath("best_model.hdf5"))
        expected_step = expected_model.optimizer.iterations
    elif case in ("dummymodel", "uncompiled"):
        inputs = tf.keras.Input(shape=(3, ))
        outputs = tf.keras.layers.Dense(5)(inputs)
        expected_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        if case == "dummymodel":
            expected_model.compile(optimizer='adam', loss='mse')
            expected_model.fit(x=tf.zeros((10, 3)),
                               y=tf.zeros((10, 5)),
                               batch_size=5,
                               epochs=1,
                               verbose=0)
            expected_step = expected_model.optimizer.iterations
        expected_model.save(str(modelfile.parent.joinpath("best_model.hdf5")))

    got_model, got_step = utilities_wae.load_model_from_directory(
        modelfile.parent)
    assert got_step == expected_step
    if case == "no_model":
        assert got_model is None
    else:
        assert got_model.name == expected_model.name
        assert got_model.input_shape == expected_model.input_shape
        assert got_model.output_shape == expected_model.output_shape


@pytest.mark.parametrize("name, expected, exception", [
    ("batchnorm", tf.keras.layers.BatchNormalization, no_raise()),
    ("layernorm", tf.keras.layers.LayerNormalization, no_raise()),
    ("invalid_layer", None,
     pytest.raises(NotImplementedError,
                   match="Normtype invalid_layer not implemented")),
])
def test_get_normtype_by_name(name, expected, exception):
    """Test _get_normtype_by_name."""
    with exception:
        got = utilities_wae._get_normtype_by_name(name)  # pylint: disable=protected-access
        assert isinstance(got, expected)


def test_create_model(monkeypatch):
    """Test model creation."""
    class _PatchEncoderDecoder:  # pylint: disable=too-few-public-methods
        def __init__(self, typ):
            self.typ = typ
            if typ == 'encoder':
                self.input_shape = {
                    "encoder_labels": [None, 2],
                    "encoder_input": [None, 10]
                }
                self.expected_args = {
                    "encoder_input": "input_data:0",
                    "encoder_labels": "batch_input_enc:0"
                }
            else:
                self.input_shape = {
                    "decoder_input": [None, 4],
                    "decoder_labels": [None, 3]
                }
                self.expected_args = {
                    "decoder_input": "gaussian",
                    "decoder_labels": "batch_input_dec:0"
                }

        def __call__(self, inputs):
            assert len(inputs) == len(self.input_shape)
            for key, layer in inputs.items():
                assert self.input_shape[key][1] == layer.shape[1]
                assert self.expected_args[key] == layer.name
            if self.typ == 'encoder':
                return 'latent', 'sigma'
            return ('decoder_output', "decoder_output_1")

    def patch_gaussian():
        def _func(inputs):
            assert inputs[0] == 'latent'
            assert inputs[1] == 'sigma'
            return collections.namedtuple('Layer',
                                          ['shape', 'name'])(name='gaussian',
                                                             shape=[None, 4])

        return _func

    monkeypatch.setattr(customlayers, 'GaussianReparametrization',
                        patch_gaussian)

    def patch_sigma():
        def _func(inputs):
            assert inputs == 'sigma'
            return 'sigma_reg'

        return _func

    monkeypatch.setattr(customlayers, 'SigmaRegularization', patch_sigma)

    total_number_cells = 100

    def patch_mmdpp(n_total):
        assert n_total == total_number_cells

        def _func(inputs):
            assert inputs[0] == 'latent'
            assert inputs[1] == 'sigma'
            return 'mmdpp'

        return _func

    monkeypatch.setattr(customlayers, 'MMDPP', patch_mmdpp)

    encoder = _PatchEncoderDecoder('encoder')
    decoder = _PatchEncoderDecoder('decoder')

    def patch_model(inputs, outputs, name):
        assert name == "WAE"
        assert outputs == {
            "decoder_dropouts": "decoder_output_1",
            "decoder_counts": "decoder_output",
            "mmdpp": "mmdpp",
            "sigma_regularization": "sigma_reg",
        }
        assert set(inputs.keys()) == set(
            ("input_data", "batch_input_enc", "batch_input_dec"))
        for key, layer in inputs.items():
            assert isinstance(layer, tf.Tensor)
            assert layer.name == key + ":0"

    monkeypatch.setattr(tf.keras, "Model", patch_model)

    utilities_wae.create_model(encoder=encoder,
                               decoder=decoder,
                               total_number_cells=total_number_cells)


def test_create_encoder():
    """Test create_encoder."""
    tf.keras.backend.clear_session()
    latent_dim = 20
    enc_layers = [100, 40]
    enc_norm_type = ['conditionallayernorm'] * len(enc_layers)
    activation_fn = tf.keras.activations.relu
    input_dim = 200
    n_labels = 2
    regularization = 0.5
    got = utilities_wae.create_encoder(latent_dim=latent_dim,
                                       enc_layers=enc_layers,
                                       enc_norm_type=enc_norm_type,
                                       activation_fn=activation_fn,
                                       input_dim=input_dim,
                                       n_labels=n_labels,
                                       regularization=regularization)
    assert got.name == "encoder"
    assert got.output_shape == [(None, latent_dim), (None, latent_dim)]
    assert got.input_shape == {
        'encoder_input': (None, input_dim),
        'encoder_labels': (None, n_labels)
    }

    got_layers = json.loads(got.to_json())["config"]
    with pathlib.Path(__file__).parent.joinpath(
            'test_utilities_wae.json').resolve().open('r') as file:
        expected = json.load(file)['encoder']
    assert got_layers == expected


@pytest.mark.parametrize("output_fn, expected", [
    ('LSN', [[1.609438, 1.94591], [1.694596, 1.880313]]),
    ("sigsoftmax", [[1.263619, 2.135562], [1.299372, 2.120224]]),
    ('softmax', [[1.305468, 2.11753], [1.305468, 2.11753]]),
    ('sigmoid', [[0.880797, 0.952574], [0.982014, 0.993307]]),
    ('softplus', [[2.126928, 3.048587], [4.01815, 5.006715]]),
    ('tensorflow.keras.activations.softplus', [[2.126928, 3.048587],
                                               [4.01815, 5.006715]]),
    ('tensorflow.keras.activations.relu', [[2, 3], [4, 5]]),
    (None, [[2, 3], [4, 5]]),
])
def test_rescale_cells(output_fn, expected):
    """Test _rescale_cells."""
    output_lsn = 10
    inputs = np.array([[2, 3], [4, 5]], dtype=np.float32)
    got = utilities_wae._rescale_cells(  # pylint: disable=protected-access
        output_cells=inputs,
        output_fn=output_fn,
        output_lsn=output_lsn)
    expected = np.array(expected, dtype=np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_create_decoder():
    """Test creation of the decoder."""
    tf.keras.backend.clear_session()
    latent_dim = 20
    output_cells_dim = 200
    dec_layers = [40, 100]
    dec_norm_type = ['conditionallayernorm'] * len(dec_layers)
    activation_fn = tf.keras.activations.relu
    output_fn, output_lsn = (None, None)
    n_labels = 2
    regularization = 0.5

    got = utilities_wae.create_decoder(latent_dim=latent_dim,
                                       output_cells_dim=output_cells_dim,
                                       dec_layers=dec_layers,
                                       dec_norm_type=dec_norm_type,
                                       activation_fn=activation_fn,
                                       output_fn=output_fn,
                                       n_labels=n_labels,
                                       regularization=regularization,
                                       output_lsn=output_lsn)
    assert got.name == "decoder"
    assert got.output_shape == [(None, output_cells_dim),
                                (None, output_cells_dim)]
    assert got.input_shape == {
        'decoder_input': (None, latent_dim),
        'decoder_labels': (None, n_labels)
    }

    got_layers = json.loads(got.to_json())["config"]
    with pathlib.Path(__file__).parent.joinpath(
            'test_utilities_wae.json').resolve().open('r') as file:
        expected = json.load(file)['decoder']
    assert got_layers == expected
