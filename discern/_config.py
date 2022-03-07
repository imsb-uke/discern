"""Module for configuring a DISCERN model."""
import warnings
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np

from discern import functions

DISCERNConfigType = TypeVar('DISCERNConfigType', bound='DISCERNConfig')


class EncoderDecoderConfig:
    """Class for handling Encoder-Decoder config"""
    def __init__(self,
                 layers: List[int],
                 norm_type: List[str],
                 regularization: float = 0.0,
                 conditional_regularization: Optional[Dict[str, Any]] = None):

        self.layers = [int(i) for i in layers]
        self.norm_type = [str(i) for i in norm_type]

        self.regularization = 0.0 if not regularization else float(
            regularization)

        self.conditional_regularization = conditional_regularization

    def __eq__(self, other) -> bool:
        return dict(self) == dict(other)

    def __iter__(self):
        yield "layers", self.layers
        yield "norm_type", self.norm_type
        yield "regularization", self.regularization
        if self.conditional_regularization is not None:
            yield "conditional_regularization", self.conditional_regularization


class DISCERNConfig:
    """Basic DISCERN configuration.

    Args:
        latent_dim (int): Dimension of the latent space.
        layers (Dict[str,List[int]]): Number and size of the layers.
        output_lsn (float): Scaling factor.
        norm_type (Dict[str,List[str]]): Batchnorm type (batch vs. layer).
        activation_fn (str): Name of the activation function between layers.
        output_fn (str): Name of the output function producing cells.
        recon_loss_type (Dict[str,Any]): Reconstruction loss type.
        wae_lambda (float): Lambda for regularization loss.
        weighting_random_encoder (float): Weighting of the sigma regularization.
        optimizer (Dict[str,Any]): Configuration for Optimizer including learning rate.

    Attributes:
        optimizer_wae (Union[tf.keras.optimizers.Optimizer, None]): Optimizer
        latent_dim (int): Dimension of the latent space.
        layers (Dict[str,List[int]]): Number and size of the layers.
        output_lsn (float): Scaling factor.
        norm_type (Dict[str,List[str]]): Batchnorm type (batch vs. layer).
        activation_fn (Callable[...,tf.keras.layers.Layer]): Activation function between layers.
        output_fn (str): Name of the output function producing cells.
        recon_loss_type (Dict[str,Any]): Reconstruction loss type.
        wae_lambda (float): Lambda for regularization loss.
        weighting_random_encoder (float): Weighting of the sigma regularization.
        optimizer_config (Dict[str,Any]): Configuration for Optimizer including learning rate
        total_number_cells (int): Total number of samples for scaling.

    """
    # pylint: disable=too-many-instance-attributes, too-few-public-methods

    latent_dim: int
    encoder_config: EncoderDecoderConfig
    decoder_config: EncoderDecoderConfig
    output_lsn: float
    output_fn: str
    recon_loss_type: Dict[str, Any]
    wae_lambda: float
    weighting_random_encoder: float
    optimizer_config: Dict[str, Any]
    weighting_decoder_dropout: float
    zeros: Union[np.ndarray, float]
    crossentropy_kwargs: Dict[str, Any]

    def __init__(self,
                 latent_dim: int,
                 encoder: Dict[str, Any],
                 decoder: Dict[str, Any],
                 output_lsn: float,
                 activation_fn: str,
                 output_fn: str,
                 recon_loss_type: Dict[str, Any],
                 wae_lambda: float,
                 weighting_random_encoder: float,
                 optimizer: Dict[str, Any],
                 weighting_decoder_dropout: float = 0.0,
                 zeros: Union[np.ndarray, float] = 0.0,
                 crossentropy_kwargs: Optional[Dict[str, Any]] = None,
                 regulatrization_conditional: Optional[Dict[str, Any]] = None):
        """Initialize the class."""
        # pylint: disable=too-many-arguments,too-many-locals

        self.latent_dim = latent_dim
        self.encoder_config = EncoderDecoderConfig(**encoder)
        self.decoder_config = EncoderDecoderConfig(**decoder)
        self.output_lsn = output_lsn
        self.regulatrization_conditional = regulatrization_conditional
        self.activation_fn = functions.get_function_by_name(  # type: ignore
            activation_fn)
        self.output_fn = output_fn
        self.recon_loss_type = recon_loss_type
        self.wae_lambda = wae_lambda
        self.weighting_random_encoder = weighting_random_encoder
        self.optimizer_config = optimizer
        self.weighting_decoder_dropout = weighting_decoder_dropout
        self.zeros = zeros
        self.crossentropy_kwargs = crossentropy_kwargs or {}

    @classmethod
    def from_json(cls: Type[DISCERNConfigType],
                  jsondata: Dict[str, Any]) -> DISCERNConfigType:
        """Create an DISCERNConfig instance form hyperparameter json dictionary.

        Args:
            jsondata (Dict[str, Any]): Hyperparameters for this model.

        Returns:
            "DISCERNConfig": An initialized DISCERNConfig instance

        Raises:
            KeyError: If required key not found in hyperparameter json.

        """
        rand_encoder = jsondata['model']['weighting_random_encoder']
        recon_loss_type = jsondata['model']['reconstruction_loss']

        jsondata['model'] = _fix_inconsistent_naming(jsondata['model'])
        jsondata['model'] = _make_symmetric(jsondata['model'])

        return cls(
            latent_dim=int(jsondata['model']['latent_dim']),
            output_lsn=jsondata["input_ds"]["scale"].get("LSN", float("nan")),
            encoder=jsondata['model']['encoder'],
            decoder=jsondata["model"]["decoder"],
            recon_loss_type=recon_loss_type,
            wae_lambda=jsondata['model']['wae_lambda'],
            weighting_random_encoder=rand_encoder,
            activation_fn=jsondata['model']['activation_fn'],
            output_fn=jsondata['model']['output_fn'],
            zeros=0.0,
            weighting_decoder_dropout=jsondata['model'].get(
                "weighting_decoder_dropout", 0.0),
            crossentropy_kwargs=jsondata["model"].get("crossentropy", None),
            optimizer=jsondata['training']['optimizer'])


def _make_symmetric(modeldata: Dict[str, Any]) -> Dict[str, Any]:
    modeldata.setdefault("encoder", {})
    modeldata.setdefault("decoder", {})
    for key in ("layers", "norm_type", "regularization",
                "conditional_regularization"):
        if key in modeldata:
            values = modeldata.pop(key)
            if key in modeldata["encoder"]:
                raise KeyError(f"{key} specified twice.")
            modeldata["encoder"][key] = values
            if isinstance(values, list):
                values = values[::-1]
            if key in modeldata["decoder"]:
                raise KeyError(f"{key} specified twice.")
            modeldata["decoder"][key] = values
        if key in modeldata["encoder"] or key in modeldata["decoder"]:
            if key not in modeldata["encoder"]:
                raise AssertionError(f"{key} missing in encoder config.")
            if key not in modeldata["decoder"]:
                raise AssertionError(f"{key} missing in decoder config.")
    return modeldata


def _fix_inconsistent_naming(
        modeldata: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
    warning = ("Defining '{encoderkey}' is deprecated. "
               "Please define them under '{layer}' using this key '{newkey}'.")
    keys = list(modeldata.keys())
    modeldata.setdefault("encoder", {})
    modeldata.setdefault("decoder", {})
    for key in keys:
        for name in ("encoder", "decoder"):
            if key.startswith(name[:3] + "_"):
                newname = key[4:]
            elif key.endswith("_" + name[:3]):
                newname = key[:-4]
            else:
                continue
            warnings.warn(
                warning.format(encoderkey=key, layer=name, newkey=newname),
                DeprecationWarning)
            if newname in modeldata[name]:
                raise KeyError(f"{newname} specified twice.")

            modeldata[name][newname] = modeldata.pop(key)
    return modeldata
