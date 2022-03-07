#!/usr/bin/python
# -*- coding: utf-8 -*-
"""A number of classes and functions used across all types of models."""
import importlib
import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models as keras_models

from discern import functions
from discern.estimators import customlayers
from discern.estimators import losses as customlosses

_LOGGER = logging.getLogger(__name__)


def _load_step_from_hd5file(filepath: pathlib.Path):
    with h5py.File(filepath, 'r') as file:
        training_config = file.get('optimizer_weights', None)
        if training_config:
            training_weights = training_config["training"]
            if "iter:0" in training_weights.keys():
                return training_weights["iter:0"][()]
            for weights in training_weights.values():
                if "iter:0" in weights.keys():
                    return weights["iter:0"][()]
        return 0


def load_model_from_directory(
        directory: pathlib.Path) -> Tuple[Union[None, tf.keras.Model], int]:
    """Load model from latest checkpoint using its hdf5 file.

    Args:
        directory (pathlib.Path): Name of the directory with hdf5 files.

    Returns:
        Tuple[Union[None, tf.keras.Model], int]: Full model and last step.
            None and zero if no models could be loaded.

    """
    custom_objects = functions.getmembers("tensorflow_addons.optimizers")
    custom_objects.update(**customlayers.getmembers())
    custom_objects.update(**customlosses.getmembers())
    modelfile = directory.joinpath("best_model.hdf5")
    if modelfile.exists():
        try:
            model = keras_models.load_model(modelfile,
                                            compile=True,
                                            custom_objects=custom_objects)
            try:
                step = model.optimizer.iterations.numpy()
            except AttributeError:
                _LOGGER.warning("Failed to load compiled model,"
                                "optimizer and step is not restored")
                step = 0

        except (TypeError, ValueError):
            _LOGGER.warning("Failed to load compiled model,"
                            "optimizer state is not restored")
            model = keras_models.load_model(modelfile,
                                            compile=False,
                                            custom_objects=custom_objects)
            step = _load_step_from_hd5file(modelfile)
        return model, step
    return None, 0


def _get_normtype_by_name(name: str) -> tf.keras.layers.Layer:
    if "batchnorm" in name:
        return layers.BatchNormalization(center=False,
                                         scale=False,
                                         trainable=False)

    if "layernorm" in name:
        return layers.LayerNormalization(center=False,
                                         scale=False,
                                         trainable=False)

    raise NotImplementedError("Normtype {} not implemented".format(name))


def create_model(encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 total_number_cells: float,
                 name: str = "WAE") -> tf.keras.Model:
    """Generate a model from encoder and decoder, adding gaussian noise (reparametrization).

    Args:
        encoder (tf.keras.Model): The encoder.
        decoder (tf.keras.Model): The decoder.
        total_number_cells (int): Total number of cells used for scaling MMDPP.
        name (str): Name of the model. Defaults to "WAE".

    Returns:
        tf.keras.Model: The created model including SigmaRegularization and MMDPP loss.

    """

    inputs = {
        "input_data":
        keras.Input(shape=(encoder.input_shape["encoder_input"][1], ),
                    name='input_data'),
        "batch_input_enc":
        keras.Input(shape=(encoder.input_shape["encoder_labels"][1], ),
                    name='batch_input_enc',
                    dtype='float32'),
        "batch_input_dec":
        keras.Input(shape=(decoder.input_shape["decoder_labels"][1], ),
                    name='batch_input_dec',
                    dtype='float32'),
    }

    latent, sigma = encoder({
        "encoder_input": inputs["input_data"],
        "encoder_labels": inputs["batch_input_enc"],
    })
    input_latent = customlayers.GaussianReparametrization()((latent, sigma))
    loss_random_encoder = customlayers.SigmaRegularization()(sigma)
    regularization_loss = customlayers.MMDPP(total_number_cells)(
        (latent, sigma))

    counts, dropout = decoder({
        "decoder_input": input_latent,
        "decoder_labels": inputs["batch_input_dec"]
    })
    counts = tf.keras.layers.Layer(name="decoder_counts")(counts)
    dropout = tf.keras.layers.Layer(name="decoder_dropouts")(dropout)

    outputs = {
        "decoder_counts": counts,
        "decoder_dropouts": dropout,
        "mmdpp": regularization_loss,
        "sigma_regularization": loss_random_encoder,
    }

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def create_encoder(
    latent_dim: int,
    enc_layers: List[int],
    enc_norm_type: List[str],
    activation_fn: Callable[[tf.Tensor], tf.Tensor],
    input_dim: int,
    n_labels: int,
    regularization: float,
    conditional_regularization: Optional[Dict[str,
                                              Any]] = None) -> tf.keras.Model:
    """Create an Encoder.

    Args:
        latent_dim (int):Dimension of the latent space.
        enc_layers (List[int]):Dimension of the encoding layers.
        enc_norm_type (List[str]): Normalization type, eg. BatchNormalization.
        activation_fn (Callable[[tf.Tensor], tf.Tensor]): Activation function in the model.
        input_dim (int): Dimension of the input.
        n_labels (int): Number of labels for the batch labels.
        regularization (float): Rate of dropout.

    Returns:
        tf.keras.Model: The encoder.

    Raises:
        NotImplementedError: If enc_norm_type is not understood.

    """
    # pylint: disable=too-many-arguments
    inputs = {
        "encoder_input":
        keras.Input(shape=(input_dim, ), name="encoder_input"),
        "encoder_labels":
        keras.Input(shape=(n_labels, ), dtype='float32', name="encoder_labels")
    }
    batch_input_enc = inputs["encoder_labels"]
    latent = inputs["encoder_input"]

    for size, normtype in zip(enc_layers, enc_norm_type):
        latent = layers.Dense(size,
                              kernel_initializer='he_normal',
                              bias_initializer=None,
                              use_bias=False)(latent)
        latent = _get_normtype_by_name(normtype)(latent)
        if normtype.startswith("conditional"):
            latent = customlayers.condlayernorm(
                latent,
                batch_input_enc,
                size,
                regularization=conditional_regularization)
        latent = activation_fn(latent)
        latent = layers.Dropout(regularization)(latent)

    latent_codes = layers.Dense(latent_dim,
                                name='latent',
                                kernel_initializer='he_normal',
                                bias_initializer='zeros')(latent)

    # A Gaussian random encoder is used as proposed in https://arxiv.org/pdf/1802.03761.pdf.
    sigmas_enc = layers.Dense(
        latent_dim,
        name='sigma_enc',
        kernel_initializer='he_normal',
        bias_initializer=tf.keras.initializers.Constant(-10))(latent)

    # instantiate encoder model
    return keras.Model(inputs=inputs,
                       outputs=[latent_codes, sigmas_enc],
                       name='encoder')


def _rescale_cells(output_cells: tf.Tensor, output_fn: Optional[str],
                   output_lsn: Optional[float]) -> tf.Tensor:

    if output_fn == "sigsoftmax":
        output_cells = tf.nn.sigmoid(output_cells) * tf.math.exp(output_cells)
        scale = tf.reduce_sum(output_cells, axis=1) + 1e-8

        return tf.math.log1p(output_lsn * output_cells /
                             tf.reshape(scale, (-1, 1)))

    if output_fn == "LSN":
        output_cells = tf.nn.relu(output_cells)
        scale = tf.reduce_sum(output_cells, axis=1) + 1e-8

        return tf.math.log1p(output_lsn * output_cells /
                             tf.reshape(scale, (-1, 1)))

    if output_fn == "softmax":
        return tf.math.log1p(output_lsn * tf.nn.softmax(output_cells))

    if output_fn:
        *mname, fname = output_fn.split(".")
        module = keras.activations
        if mname:
            module = importlib.import_module(".".join(mname))
        return getattr(module, fname)(output_cells)

    return output_cells


def create_decoder(
    latent_dim: int,
    output_cells_dim: int,
    dec_layers: List[int],
    dec_norm_type: List[str],
    activation_fn: Callable[[tf.Tensor], tf.Tensor],
    output_fn: Optional[str],
    n_labels: int,
    regularization: float,
    output_lsn: Optional[float] = None,
    conditional_regularization: Optional[Dict[str,
                                              Any]] = None) -> tf.keras.Model:
    """Create a decoder.

    Args:
        latent_dim (int): Dimension of the latent space.
        output_cells_dim (int): Dimension of the output.
        dec_layers (List[int]): Dimensions for the decoder layers.
        dec_norm_type (List[str]):  Normalization type, eg. BatchNormalization.
        activation_fn (Callable[[tf.Tensor], tf.Tensor]): Activation function in the model.
        output_fn (str): Function to produce gene counts.
        n_labels (int): Number of labels for the batch labels.
        regularization (float): Dropout rate.
        output_lsn (Optional[float]): Scaling parameter, used for softmax and LSN.


    Returns:
        tf.keras.Model: The decoder.

    """
    # pylint: disable=too-many-arguments, too-many-locals
    inputs = {
        "decoder_input":
        keras.Input(shape=(latent_dim, ),
                    name='decoder_input',
                    dtype='float32'),
        "decoder_labels":
        keras.Input(shape=(n_labels, ), dtype='float32',
                    name="decoder_labels"),
    }

    z_input = inputs["decoder_input"]

    for size, normtype in zip(dec_layers, dec_norm_type):
        z_input = layers.Dense(size,
                               kernel_initializer='he_normal',
                               bias_initializer=None,
                               use_bias=False)(z_input)
        z_input = _get_normtype_by_name(normtype)(z_input)
        if normtype.startswith("conditional"):
            z_input = customlayers.condlayernorm(
                z_input,
                inputs["decoder_labels"],
                size,
                regularization=conditional_regularization)
        z_input = activation_fn(z_input)
        z_input = layers.Dropout(regularization)(z_input)

    output_cells = layers.Dense(output_cells_dim,
                                kernel_initializer='he_normal',
                                bias_initializer='zeros')(z_input)
    output_cells = _rescale_cells(output_cells, output_fn, output_lsn)

    dropouts = layers.Dense(output_cells_dim,
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            activation="sigmoid")(z_input)

    return keras.Model(inputs=inputs,
                       outputs=[output_cells, dropouts],
                       name='decoder')
