"""Module for supporting online learning."""
import collections
import logging
import pathlib
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from discern import functions, io, preprocessing
from discern.estimators import batch_integration, run_exp

_LOGGER = logging.getLogger(__name__)


class OnlineDISCERNRunner(run_exp.DISCERNRunner):
    """DISCERNRunner supporting online learning."""

    # pylint: disable=too-few-public-methods
    def __init__(self, debug: bool, gpus: List[int]):
        super().__init__(debug=debug, gpus=gpus)
        self._run_options = {"train": online_training}


def save_data(file: pathlib.Path, data: io.DISCERNData,
              old_data: io.DISCERNData):
    """Save data by concatenating to reference."""
    adata = old_data.concatenate(data,
                                 join="inner",
                                 batch_key=None,
                                 uns_merge="first",
                                 index_unique=None)
    adata.varm = old_data.varm.copy()
    adata.obs.batch = adata.obs.batch.astype("category")
    categories = collections.OrderedDict.fromkeys(
        old_data.obs.batch.cat.categories, True)
    new_categories = collections.OrderedDict.fromkeys(
        data.obs.batch.cat.categories, True)
    categories.update(new_categories)
    adata.obs.batch.cat.reorder_categories(categories.keys(), inplace=True)
    adata.write(file)


def _prepare_inputdata(exp_folder: pathlib.Path, hparams: Dict[str, Any],
                       filename: pathlib.Path):

    reference_data = io.DISCERNData.from_folder(
        folder=exp_folder, batch_size=hparams['training']['batch_size'])

    inputdata = OnlineWAERecipe(reference_adata=reference_data,
                                params=hparams,
                                input_files=[filename])().sc_raw
    inputdata.batch_size = hparams['training']['batch_size']

    save_data(file=exp_folder.joinpath("processed_data",
                                       "concatenated_data.h5ad"),
              data=inputdata,
              old_data=reference_data)
    return inputdata


def online_training(exp_folder: pathlib.Path, filename: pathlib.Path,
                    freeze: bool):
    """Continue running an experiment.

    Args:
        exp_folder (pathlib.Path): Experiments folders.
        filename (pathlib.Path): Input path for new data set.
        freeze (bool): Freeze non conditional layers.
    """
    batch_model, hparams = run_exp.setup_exp(exp_folder)
    reference_model = batch_integration.DISCERN.from_json(hparams)

    inputdata = _prepare_inputdata(exp_folder=exp_folder,
                                   hparams=hparams,
                                   filename=filename)

    batch_model.zeros = inputdata.zeros

    if exp_folder.joinpath("backup").exists():
        shutil.rmtree(exp_folder.joinpath("backup"))
    shutil.copytree(exp_folder.joinpath("job"), exp_folder.joinpath("backup"))

    reference_model.restore_model(exp_folder.joinpath("job"))

    batch_model.build_model(n_genes=inputdata.var_names.size,
                            n_labels=inputdata.obs.batch.cat.categories.size,
                            scale=inputdata.config["total_train_count"])

    batch_model.wae_model = update_model(old_model=reference_model.wae_model,
                                         new_model=batch_model.wae_model,
                                         freeze_unchanged=freeze)

    _LOGGER.debug("Recompile Model to apply freezing")

    batch_model.compile(optimizer=batch_model.get_optimizer(), scale=15000.0)

    _LOGGER.debug("Starting online training of %s", exp_folder)

    run_exp._train(  # pylint: disable=protected-access
        model=batch_model,
        exp_folder=exp_folder.resolve(),
        inputdata=inputdata,
        early_stopping=hparams['training']["early_stopping"],
        max_steps=hparams['training']['max_steps'])

    _LOGGER.info('%s has finished online training', exp_folder)


def _update_weights(
        old_weights: List[np.ndarray],
        new_weights: List[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
    weights = []
    changed = False
    for w_old, w_new in zip(old_weights, new_weights):
        if w_old.shape != w_new.shape:
            w_new[:w_old.shape[0]] = w_old
            weights.append(w_new)
            changed = True
        else:
            weights.append(w_old)
    return weights, changed


def update_model(old_model: tf.keras.Model,
                 new_model: tf.keras.Model,
                 freeze_unchanged: bool = False) -> tf.keras.Model:
    """Update the weights from an old model to a new model.

    New model can have a bigger weight size for the layers in the first dimension.

    Args:
        old_model (tf.keras.Model): Old, possibly trained model.
        new_model (tf.keras.Model): New model, for which the weights
            should be set (inplace).
        freeze_unchanged (bool, optional): Freeze layers in the new model,
            which weights didn't changed in size compared to the old model.
            Defaults to False.

    Returns:
        tf.keras.Model: The updated new model.
    """
    for layer_old, layer_new in zip(old_model.layers, new_model.layers):
        if isinstance(layer_old, tf.keras.Model):
            update_model(layer_old, layer_new, freeze_unchanged)
        else:
            new_weights, changed = _update_weights(layer_old.get_weights(),
                                                   layer_new.get_weights())
            layer_new.set_weights(new_weights)
            if not changed:
                trainable = not freeze_unchanged if layer_new.trainable else layer_new.trainable
                layer_new.trainable = trainable
    return new_model


class OnlineWAERecipe(preprocessing.WAERecipe):
    """WAERecipe for the online setting.

       This class allows the same preprocessing as done for reference data,
       which allows further processing in using DISCERN online learning.
    """
    def __init__(self, reference_adata: io.DISCERNData, *args, **kwargs):
        """Initialize the class."""
        self._reference = reference_adata
        super().__init__(*args, **kwargs)

    def filtering(self, min_genes: int, *unused_args, **unused_kwargs):
        """Apply filtering in-place.

        Args:
            min_genes (int): Minimum number of genes to be present for cell to be considered.

        """
        self.sc_raw = self.sc_raw[:, self._reference.var_names]
        self._filter_cells(min_genes=min_genes)

    def projection_pca(self, pcs: int = 25):
        """Apply PCA projection from reference.

        Args:
            pcs (int): Number of principle components. Defaults to 32.

        """
        if pcs != self._reference.varm['PCs'].shape[1]:
            raise ValueError(
                "Cannot use different number of PCs than reference."
                " Got %d but expected %d" %
                (pcs, self._reference.varm['PCs'].shape[1]))
        self.sc_raw.var['mean_scaling'] = self._reference.var["mean_scaling"]
        self.sc_raw.var['var_scaling'] = self._reference.var["var_scaling"]
        self.sc_raw.var['pca_genes'] = self._reference.var['pca_genes']
        self.sc_raw.varm['PCs'] = self._reference.varm['PCs']

    def mean_var_scaling(self):
        """Apply Mean-Variance scaling if 'fixed_scaling' is present."""
        scale_dict = self._reference.uns.get('fixed_scaling', None)
        if scale_dict:
            _LOGGER.warning(
                "Found 'fixed_scaling' in reference, cells will be scaled.")
            mean, var = functions.parse_mean_var(self._reference.var,
                                                 scale_dict)
            functions.scale(self.sc_raw, var=var, mean=mean)
            self.sc_raw.uns['fixed_scaling'] = scale_dict
            _LOGGER.debug("Scaling of cells finished.")

    def fix_batch_labels(self):
        """Fix batch labels codes by including old categories."""
        categories = collections.OrderedDict.fromkeys(
            self._reference.obs.batch.cat.categories, True)
        new_categories = collections.OrderedDict.fromkeys(
            self.sc_raw.obs.batch.cat.categories, True)
        categories.update(new_categories)
        self.sc_raw.obs.batch = pd.Categorical(self.sc_raw.obs.batch,
                                               categories=categories.keys())

    def __call__(self) -> "OnlineWAERecipe":
        """Apply the recipe."""
        super().__call__()
        self.fix_batch_labels()
        return self
