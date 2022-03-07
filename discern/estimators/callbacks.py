"""Module for custom callbacks, especially visualization(UMAP)."""
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import scanpy as sc
import tensorflow as tf
from scipy import sparse as scsparse
from tensorflow import config as tfconfig
from tensorflow.keras import callbacks

from discern import functions, io


def _plot_umap(cells: anndata.AnnData,
               logdir: pathlib.Path,
               epoch: Union[int, str],
               disable_pca: bool = False):
    n_comps = min(20, min(cells.shape) - 1)
    if not disable_pca:
        sc.tl.pca(cells, svd_solver='arpack', n_comps=n_comps)
    sc.pp.neighbors(cells, use_rep='X' if disable_pca else "X_pca")
    sc.tl.umap(cells)
    for key in cells.obs.columns[cells.obs.dtypes == "category"]:
        sc.pl.umap(cells,
                   color=key,
                   title=str(logdir),
                   show=False,
                   size=50.0,
                   sort_order=False,
                   save='_epoch_{}_{}.png'.format(epoch, key))


class VisualisationCallback(callbacks.Callback):  # pylint: disable=too-few-public-methods
    """Redo prediction on datasets and visualize via UMAP.

    Args:
        outdir (pathlib.Path): Output directory for the figures.
        data (anndata.AnnData): Input cells.
        batch_size (int): Numer of cells to visualize.
        freq (int): Frequency for computing visualisations in epochs. Defaults 10.

    """

    _outdir: pathlib.Path
    _initial_data: anndata.AnnData
    _labels: np.ndarray
    _data: Dict[Union[str, int], tf.data.Dataset]
    _batch_size: int
    _freq: int

    def __init__(self,
                 outdir: Union[str, pathlib.Path],
                 data: anndata.AnnData,
                 batch_size: int,
                 freq: int = 10):
        """Initialize the callback and do one UMAP plot with original data."""
        #pylint: disable=too-many-arguments
        super().__init__()

        self._outdir = pathlib.Path(outdir).joinpath("UMAP")
        self._initial_data = data
        self._batch_size = batch_size
        self._data = dict()
        self._freq = freq
        n_threads = tfconfig.threading.get_inter_op_parallelism_threads()
        n_threads += tfconfig.threading.get_intra_op_parallelism_threads()
        if n_threads > 0:
            sc.settings.n_jobs = n_threads
        sc.settings.autosave = True
        self._outdir.mkdir(exist_ok=True, parents=True)

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None):  # pylint: disable=unused-argument
        """Run on training start.

        Args:
            logs (Optional[Dict[str, float]]): logs, not used only for compatibility reasons.
        """
        n_labels = self.model.input_shape["batch_input_enc"][-1]
        input_enc = self._initial_data.obs.batch.cat.codes.values.astype(
            np.int32)
        input_enc = tf.one_hot(input_enc, depth=n_labels, dtype=tf.float32)
        cells = self._initial_data.X
        if scsparse.issparse(cells):
            cells = cells.todense()
        cells = tf.cast(cells, tf.float32)
        self._data["original"] = {
            'input_data': cells,
            'batch_input_enc': input_enc,
            'batch_input_dec': input_enc,
        }
        self._data["latent"] = {
            'encoder_input': cells,
            'encoder_labels': input_enc,
        }
        name_to_code = {
            name: code
            for code, name in enumerate(
                self._initial_data.obs.batch.cat.categories)
        }
        labels = self._initial_data.obs.batch.value_counts(
            sort=False, dropna=True).index.values

        for name in labels:
            tmp = np.zeros_like(input_enc)
            tmp[:, name_to_code[name]] = 1
            self._data[name] = {
                'input_data': cells,
                'batch_input_enc': input_enc,
                'batch_input_dec': tf.cast(tmp, tf.float32),
            }

        logdir = self._outdir.joinpath("projected_to_original")
        logdir.mkdir(exist_ok=True, parents=True)
        sc.settings.figdir = logdir
        _plot_umap(self._initial_data, logdir, 0)

    def _do_prediction_and_plotting(self, epoch: Union[str, int],
                                    batch_size: int):
        loglevel = logging.getLogger(__name__).getEffectiveLevel()
        logging.getLogger("anndata").setLevel(loglevel)
        for dataset, dataiterator in self._data.items():
            if dataiterator.keys() != self._data["original"].keys():
                continue
            predictions = self.model.predict(dataiterator,
                                             batch_size=batch_size)[:2]
            predictions = functions.sample_counts(counts=predictions[0],
                                                  probabilities=predictions[1],
                                                  var=self._initial_data.var,
                                                  uns=self._initial_data.uns)
            logdir = self._outdir.joinpath("projected_to_{}".format(dataset))
            logdir.mkdir(parents=True, exist_ok=True)
            sc.settings.figdir = logdir
            predictions = anndata.AnnData(predictions,
                                          obs=self._initial_data.obs,
                                          var=self._initial_data.var)
            merged = predictions.concatenate(
                self._initial_data[self._initial_data.obs.batch ==
                                   dataset].copy(),
                join='inner',
                batch_categories=['_autoencoded', '_valid'],
                batch_key='origin')

            merged.obs.batch = merged.obs.apply(
                lambda row: row.batch + row.origin, axis=1).astype("category")
            if "celltype" in merged.obs.columns:
                merged.obs.celltype = merged.obs.celltype.astype("category")
            merged.obs = merged.obs.drop(columns=["origin"], errors="ignore")

            _plot_umap(merged, logdir, epoch)

        encoder = self.model.get_layer("encoder")
        logdir = self._outdir.joinpath("latent_codes")
        logdir.mkdir(exist_ok=True, parents=True)
        sc.settings.figdir = logdir
        latent = encoder.predict(self._data["latent"],
                                 batch_size=batch_size)[0]
        latent = anndata.AnnData(latent, obs=self._initial_data.obs)
        _plot_umap(latent, logdir, epoch, disable_pca=True)

    def on_epoch_end(self,
                     epoch: int,
                     logs: Optional[Dict[str, float]] = None):  # pylint: disable=unused-argument
        """Run on epoch end. Executes only at specified frequency.

        Args:
            epoch (int): Epochnumber.
            logs (Optional[Dict[str, float]]): losses and metrics passed by tensorflow fit .
                    Defaults to None.
        """
        if epoch > 0 and epoch % self._freq == 0:
            self._do_prediction_and_plotting(epoch + 1, self._batch_size)

    def on_train_end(self, logs: Optional[Dict[str, float]] = None):  # pylint: disable=unused-argument
        """Run on training end.

        Args:
            logs (Optional[Dict[str, float]]): losses and metrics passed by tensorflow fit .
                    Defaults to None.
        """
        self._do_prediction_and_plotting("end", self._batch_size)


def create_callbacks(early_stopping_limits: Dict[str, Any],
                     exp_folder: pathlib.Path,
                     inputdata: Optional[io.DISCERNData] = None,
                     umap_cells_no: Optional[int] = None,
                     profile_batch: int = 2,
                     freq_of_viz: int = 30) -> List[callbacks.Callback]:
    """Generate list of callbacks used by tensorflow model.fit.

    Args:
        early_stopping_limits ( Dict[str,Any):
            Patience, min_delta, and delay for early stopping.
        exp_folder (str):  Folder where everything is saved.
        inputdata (io.DISCERNData, optional): Input data to use. Defaults to None
        umap_cells_no (int): Number of cells for UMAP.
        profile_batch (int): Number of the batch to do extensive profiling.
            Defaults to 2. (see tf.keras.callbacks.Tensorboard)
        freq_of_viz (int): Frequency of visualization callback in epochs. Defaults to 30.

    Returns:
        List[callbacks.Callback]: callbacks used by tensorflow model.fit.

    """
    # pylint: disable=too-many-arguments
    logdir = pathlib.Path(exp_folder).joinpath("job")

    used_callbacks = list()

    used_callbacks.append(callbacks.TerminateOnNaN())
    used_callbacks.append(DelayedEarlyStopping(**early_stopping_limits))
    used_callbacks.append(
        callbacks.TensorBoard(log_dir=str(logdir),
                              histogram_freq=20,
                              profile_batch=profile_batch,
                              update_freq='epoch'))

    if inputdata is not None:
        batch_size = inputdata.batch_size
        data = inputdata[inputdata.obs.split == "valid"].copy()
        data.obs = data.obs.drop(columns=["split", "barcodes"],
                                 errors="ignore")
        labels = data.obs.batch.value_counts(sort=True, dropna=True)
        labels = labels[:10].index.values
        data = data[data.obs.batch.isin(labels)]
        umap_cells_no = min(umap_cells_no, data.X.shape[0])
        idx = np.random.choice(np.arange(data.X.shape[0]),
                               umap_cells_no,
                               replace=False)
        used_callbacks.append(
            VisualisationCallback(logdir, data[idx], batch_size, freq_of_viz))

    return used_callbacks


class DelayedEarlyStopping(tf.keras.callbacks.EarlyStopping):
    """Stop when a monitored quantity has stopped improving after some delay time.
    Args:
        delay (int): Number of epochs to wait until applying early stopping.
        Defaults to 0, which means standard early stopping.
        monitor (str): Quantity to be monitored.
        min_delta (float): Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement. Defaults to `val_loss`.
        patience (int): Number of epochs with no improvement
        after which training will be stopped. Defaults to 0.
        verbose (int): verbosity mode. Defaults to 0.
        mode (str): One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `max`
        mode it will stop when the quantity
        monitored has stopped increasing; in `auto`
        mode, the direction is automatically inferred
        from the name of the monitored quantity. Defaults to `auto`.
        baseline (float, optional): Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline. Defaults to None.
        restore_best_weights (bool): Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used. Defaults to False.
    """
    # pylint: disable=too-few-public-methods
    _delay: int

    def __init__(self,
                 delay: int = 0,
                 monitor: str = 'val_loss',
                 min_delta: float = 0.,
                 patience: int = 0,
                 verbose: int = 0,
                 mode: str = 'auto',
                 baseline: Optional[float] = None,
                 restore_best_weights: bool = False):
        """Initialize the callback."""
        # pylint: disable=too-many-arguments
        self._delay = int(delay)
        super().__init__(min_delta=min_delta,
                         monitor=monitor,
                         patience=patience,
                         verbose=verbose,
                         mode=mode,
                         baseline=baseline,
                         restore_best_weights=restore_best_weights)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on epoch end to check for early stopping."""
        if epoch < self._delay:
            return
        super().on_epoch_end(epoch=epoch, logs=logs)
        return
