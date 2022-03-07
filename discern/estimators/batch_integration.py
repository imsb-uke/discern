"""Basic DISCERN architecture."""
import logging
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow_addons import optimizers as tfa_opt

import discern
from discern import io, functions
from discern.estimators import losses, utilities_wae

_LOGGER = logging.getLogger(__name__)


class DISCERN(discern.DISCERNConfig):  # pylint: disable=too-many-instance-attributes
    """Basic DISCERN model holding a lot of configuration.

    Args:
        **kwargs: discern.DISCERNConfig init args.

    Attributes:
        wae_model (Union[None, tf.keras.Model]): Keras model.
        start_step (int): Epoch to start training from

    """
    wae_model: tf.keras.Model
    start_step: int

    def __init__(self, **kwargs):
        """Initialize the class."""
        # pylint: disable=too-many-arguments,too-many-locals

        super().__init__(**kwargs)

        self.wae_model = None
        self.start_step = 0

    @property
    def encoder(self) -> tf.keras.Model:
        """Return the encoder.

        Returns:
            tf.keras.Model: The encoder model.

        Raises:
            ValueError: If the encoder is not present.
            AttributeError: If the model is not build.

        """
        return self.wae_model.get_layer("encoder")

    @property
    def decoder(self) -> tf.keras.Model:
        """Return the decoder.

        Returns:
            tf.keras.Model: The decoder model.

        Raises:
            ValueError: If the decoder is not present.
            AttributeError: If the model is not build.

        """
        return self.wae_model.get_layer("decoder")

    def restore_model(self, directory: pathlib.Path):
        """Restores model from hdf5 checkpoint and compiles it.

        Args:
            directory (pathlib.Path): checkpoint directory.

        """
        _LOGGER.debug("Try to restore model form checkpoint...")
        self.wae_model, self.start_step = utilities_wae.load_model_from_directory(
            directory)

        if self.wae_model and not self.wae_model.optimizer:
            optimizer = self.get_optimizer()
            self.compile(optimizer)

    def build_model(self, n_genes: int, n_labels: int, scale: float):
        """Initialize the auto-encoder model and defining the loss and optimizer."""

        _LOGGER.debug("Building encoder...")
        encoder = utilities_wae.create_encoder(
            latent_dim=self.latent_dim,
            enc_layers=self.encoder_config.layers,
            enc_norm_type=self.encoder_config.norm_type,
            activation_fn=self.activation_fn,
            input_dim=n_genes,
            n_labels=n_labels,
            regularization=self.encoder_config.regularization,
            conditional_regularization=self.encoder_config.
            conditional_regularization)

        _LOGGER.debug("Building decoder...")
        decoder = utilities_wae.create_decoder(
            latent_dim=self.latent_dim,
            output_cells_dim=n_genes,
            dec_layers=self.decoder_config.layers,
            dec_norm_type=self.decoder_config.norm_type,
            output_lsn=self.output_lsn,
            activation_fn=self.activation_fn,
            output_fn=self.output_fn,
            n_labels=n_labels,
            regularization=self.decoder_config.regularization,
            conditional_regularization=self.decoder_config.
            conditional_regularization)

        _LOGGER.debug("Combining autoencoder model...")
        self.wae_model = utilities_wae.create_model(encoder, decoder, scale)
        _LOGGER.debug("Set optimizer and compile model...")
        optimizer = self.get_optimizer()
        self.compile(optimizer, scale=15000.0)

        _LOGGER.info("Model building finished")

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create an Optimizer instance.

        Returns:
            tf.keras.optimizers.Optimizer: The created optimizer.

        """
        config = self.optimizer_config.copy()
        initial_learning_rate = config.pop("learning_rate")
        decay = config.pop("learning_decay", None)
        lookahead = config.pop("Lookahead", None)
        algorithm = functions.get_function_by_name(config.pop("algorithm"))

        if decay:
            decay_algo = functions.get_function_by_name(decay.pop("name"))
            learning_rate = decay_algo(
                initial_learning_rate=initial_learning_rate, **decay)

        else:
            learning_rate = initial_learning_rate

        opt_algorithm = algorithm(learning_rate=learning_rate, **config)
        if lookahead:
            opt_algorithm = tfa_opt.lookahead.Lookahead(opt_algorithm)

        return opt_algorithm

    def compile(self,
                optimizer: tf.keras.optimizers.Optimizer,
                scale: float = 15000.0):
        """Compile the model and sets losses and metrics.

        Args:
            optimizer (tf.keras.optimizers.Optimizer): Optimizer to use.
            scale (float): Numeric scaling factor for the losses. Defaults to 15000.


        """
        if "zeros" in self.recon_loss_type:
            self.recon_loss_type["zeros"] = self.zeros

        reconstruction_loss = losses.reconstruction_loss(
            self.recon_loss_type.copy())

        dummy = tf.constant(0.0, shape=(1, 1))

        self.wae_model.compile(
            optimizer=optimizer,
            loss={
                "decoder_counts":
                reconstruction_loss,
                "decoder_dropouts":
                losses.MaskedCrossEntropy(zeros=self.zeros,
                                          **self.crossentropy_kwargs),
                "sigma_regularization":
                losses.DummyLoss(),
                "mmdpp":
                losses.DummyLoss(),
            },
            loss_weights={
                "decoder_counts": scale,
                "decoder_dropouts": self.weighting_decoder_dropout,
                "sigma_regularization": self.weighting_random_encoder,
                "mmdpp": self.wae_lambda,
            },
            target_tensors={
                "sigma_regularization": dummy,
                "mmdpp": dummy,
            })

    def training(self,
                 inputdata: io.DISCERNData,
                 callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
                 savepath: Optional[pathlib.Path] = None,
                 max_steps: int = 25) -> Dict[str, float]:
        """Train the network `max_steps` times.

        Args:
            inputdata (io.DISCERNData): Training data.
            max_steps (int): Maximum number of epochs to train. Defaults to 25.
            callbacks: (List[tf.keras.callbacks.Callback], optional):
                List of keras callbacks to use. Defaults to None.
            savepath (pathlib.Path, optional):
                Filename to save model. Defaults to None.

        Returns:
            Dict[str,float]: Metrics from fit method.

        """
        # pylint: disable=too-many-arguments

        verbose = 1 if (_LOGGER.getEffectiveLevel() <= 20) else 2

        train_data, valid_data = inputdata.tfdata

        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
        valid_data = valid_data.prefetch(tf.data.experimental.AUTOTUNE)

        start_epoch = self.start_step // (
            inputdata.config["total_train_count"] // inputdata.batch_size)

        _LOGGER.debug(
            "Finished dataset and callbacks creation, start training...")
        if start_epoch > 0:
            _LOGGER.warning("Resuming training from epoch %d.", start_epoch)

        result = self.wae_model.fit(x=train_data,
                                    epochs=max_steps,
                                    validation_data=valid_data,
                                    verbose=verbose,
                                    callbacks=callbacks,
                                    initial_epoch=start_epoch)

        if savepath is not None:
            self.wae_model.save(savepath, overwrite=True)
        return result

    def generate_latent_codes(
            self, counts: Union[tf.Tensor,
                                np.ndarray], batch_labels: Union[tf.Tensor,
                                                                 np.ndarray],
            batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate latent codes from count and batch labels.

        Args:
            counts (Union[tf.Tensor, np.ndarray]): Count data.
            batch_labels (Union[tf.Tensor, np.ndarray]): (One Hot) Encoded batch labels.
                Can also be continous for fuzzy batch association.
            batch_size (int): Size of one batch.

        Returns:
            Tuple[np.ndarray, np.ndarray]: latent codes and sigma values.

        """
        dataset = {
            'encoder_input': tf.cast(counts, tf.float32),
            'encoder_labels': tf.cast(batch_labels, tf.float32),
        }
        return self.encoder.predict(dataset, batch_size=batch_size)

    def generate_cells_from_latent(
            self, latent_codes: Union[tf.Tensor, np.ndarray],
            output_batch_labels: Union[tf.Tensor, np.ndarray],
            batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate counts from latent codes and batch labels.

        Args:
            latent_codes (Union[tf.Tensor, np.ndarray]): Latent codes produced by encoder.
            output_batch_labels (Union[tf.Tensor, np.ndarray]):  (One Hot) Encoded batch labels
                for the output. Can also be continous for fuzzy batch association.
            batch_size (int): Size of one batch.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the generated count data and dropout probabilities.
        """
        dataset = {
            'decoder_input': tf.cast(latent_codes, tf.float32),
            'decoder_labels': tf.cast(output_batch_labels, tf.float32),
        }
        return self.decoder.predict(dataset, batch_size=batch_size)

    def project_to_metadata(self,
                            input_data: io.DISCERNData,
                            metadata: List[Tuple[str, str]],
                            save_path: pathlib.Path,
                            store_sigmas: bool = False):
        """Project to average batch with filtering for certain metadata.

        Args:
            input_data (io.DISCERNData): Input cells.
            metadata (List[Tuple[str, str]]): Column-value-Pair used for filerting the cells.
                Column should match to name in input_data.obs and
                value to a key in this column.
            save_path (pathlib.Path): Path for saving the created AnnData objects.
            store_sigmas (bool, optional): Save sigmas in obsm. Defaults to False.

        """
        # pylint: disable=too-many-locals, too-many-arguments
        save_path = pathlib.Path(save_path).resolve()
        n_labels = self.encoder.input_shape["encoder_labels"][1]
        input_labels = input_data.obs.batch.cat.codes.values.astype(np.int32)
        input_labels_one_hot = tf.one_hot(input_labels,
                                          depth=n_labels,
                                          dtype=tf.float32).numpy()
        latent, sigma = self.generate_latent_codes(input_data.X,
                                                   input_labels_one_hot,
                                                   input_data.batch_size)

        is_scaled = 'fixed_scaling' in input_data.uns

        generate_h5ad_kwargs = dict(var=input_data.var,
                                    obs=input_data.obs,
                                    uns=input_data.uns,
                                    obsm={"X_DISCERN": latent},
                                    threshold=-np.inf if is_scaled else 0.0)
        if store_sigmas:
            generate_h5ad_kwargs["obsm"]["X_DISCERN_sigma"] = sigma

        _LOGGER.debug('Generation of latent code finished')

        if not metadata:
            projected = self.generate_cells_from_latent(
                latent, input_labels_one_hot, input_data.batch_size)
            _LOGGER.debug('Generation of unprojected cell finished')
            io.generate_h5ad(counts=projected,
                             save_path=save_path.joinpath("unprojected.h5ad"),
                             **generate_h5ad_kwargs)
            _LOGGER.debug('Saving unprojected cells finished')
            return

        for column, value in metadata:
            tmp_labels = np.zeros_like(input_labels_one_hot)
            filename = f"projected_to_average_{column}"
            if value:
                idx = input_data.obs[column] == value
                if idx.sum() == 0.0:
                    raise ValueError(f"Value `{value}´ not in `{column}´.")
                freq = input_labels_one_hot[idx].sum(axis=0) / idx.sum()
                freq = np.nan_to_num(freq, posinf=0.0, nan=0.0, neginf=0.0)
                tmp_labels += freq
                filename += f"_{value}"
            elif column == "batch":
                summed = input_labels_one_hot.sum(axis=0)
                tmp_labels += summed / summed.sum()
            else:
                for col_value in input_data.obs[column].unique():
                    idx = input_data.obs[column] == col_value
                    freq = input_labels_one_hot[idx].sum(axis=0) / idx.sum()
                    tmp_labels[idx, :] = freq

            projected = self.generate_cells_from_latent(
                latent, tmp_labels, input_data.batch_size)
            _LOGGER.debug('Projection of %s:%s finished, saving files...',
                          column, value)
            io.generate_h5ad(
                counts=projected,
                save_path=save_path.joinpath(filename).with_suffix(".h5ad"),
                **generate_h5ad_kwargs)
            _LOGGER.debug('Saving files finished at %s.h5ad', filename)
        _LOGGER.debug('All projections finished successfully')
