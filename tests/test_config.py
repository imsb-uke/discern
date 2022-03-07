"""Test DISCERN configuration."""
import json
import pathlib
from contextlib import ExitStack as no_raise
import pytest

import tensorflow as tf

import discern


class TestDISCERNConfig:
    """Test DISCERNConfig."""

    # pylint: disable=no-self-use, too-few-public-methods
    def test_from_json(self, parameters):
        """Test creation from json."""
        # pylint: disable=too-many-locals
        parameters_path = pathlib.Path(parameters)
        with parameters_path.open('r') as file:
            parameters = json.load(file)
        got = discern.DISCERNConfig.from_json(parameters)
        model_params = parameters['model']
        attributes_to_check = {
            "latent_dim": int(model_params['latent_dim']),
            "output_lsn":
            parameters["input_ds"]["scale"].get("LSN", float("nan")),
            "encoder_config": model_params['encoder'],
            "decoder_config": model_params['decoder'],
            "wae_lambda": model_params['wae_lambda'],
            "weighting_random_encoder":
            model_params['weighting_random_encoder'],
            "optimizer_config": parameters['training']['optimizer'],
        }
        for attribute, expected in attributes_to_check.items():
            assert got.__getattribute__(attribute) == expected
        assert got.activation_fn is tf.keras.activations.relu
        assert dict(got.encoder_config) == model_params['encoder']
        assert dict(got.decoder_config) == model_params['decoder']

        expected_recon_loss_type = model_params['reconstruction_loss'].copy()
        got_recon_loss_type = got.recon_loss_type.copy()
        assert got_recon_loss_type == expected_recon_loss_type


@pytest.mark.parametrize("testdata, exception", [
    ({
        "layers": "dummy"
    }, no_raise()),
    ({
        "encoder": {
            "layers": "dummy"
        },
        "decoder": {
            "layers": "dummy"
        }
    }, no_raise()),
    ({
        "decoder": {
            "layers": "dummy"
        }
    }, pytest.raises(AssertionError)),
    ({
        "layers": "dummy",
        "decoder": {
            "layers": "dummy"
        },
        "encoder": {
            "layers": "dummy"
        }
    }, pytest.raises(KeyError)),
])
def test_make_symmetric(testdata, exception):
    with exception:
        results = discern._config._make_symmetric(testdata)  # pylint: disable=protected-access
        assert "layers" not in results
        assert "layers" in results["encoder"]
        assert "layers" in results["decoder"]
