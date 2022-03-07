"""Basic module for running an experiment."""
import json
import logging
import multiprocessing as mp
import pathlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import tensorflow as tf

from discern import functions, io
from discern.estimators import batch_integration
from discern.estimators import callbacks as customcallbacks

_LOGGER = logging.getLogger(__name__)


class DISCERNRunner:
    """Run DISCERN training or project."""

    # pylint: disable=too-few-public-methods
    def __init__(self, debug: bool = False, gpus: List[int] = None):
        """SetUp the runner.

        Args:
            debug (bool, optional): Run in debug mode. Defaults to False.
            gpus (List[int], optional): Available GPUs. Defaults to None.
        """
        self.available_gpus = gpus if gpus else []
        if not self.available_gpus:
            logging.warning(
                "No GPUs available, running only on CPU is not recommended."
                " Going to run experiments sequentially...")
        elif len(self.available_gpus) == 1:
            logging.info("Only one GPU available, running sequentially...")
        else:
            logging.debug("Using GPUs %s", self.available_gpus)

        self.debug = debug
        self._run_options: Dict[str, Callable[..., None]] = {
            "train": run_train,
            "project": run_projection
        }

    def _run_debug(self, mode: str, experiments: List[pathlib.Path], **kwargs):
        self._run_options[mode](experiments[0], **kwargs)

    def _run_single_threaded(self, mode: str, experiments: List[pathlib.Path],
                             **kwargs):
        for exp_folder in experiments:
            self._run_options[mode](exp_folder, **kwargs)

    def _run_multiprocessed(self, mode: str, experiments: List[pathlib.Path],
                            **kwargs):
        manager = mp.Manager()
        avail_gpus = manager.list(self.available_gpus)
        run_args = ((exp_folder, avail_gpus, self._run_options[mode], kwargs)
                    for exp_folder in experiments)
        with mp.Pool(len(avail_gpus), maxtasksperchild=1) as pool:
            results = pool.starmap_async(run_exp_multiprocess,
                                         run_args,
                                         chunksize=1)
            results.get()

    def __call__(self, mode: str, experiments: List[pathlib.Path], **kwargs):
        func = self._run_multiprocessed
        if len(self.available_gpus) <= 1:
            func = self._run_single_threaded
        if self.debug:
            func = self._run_debug
        func(mode=mode, experiments=experiments, **kwargs)


def run_exp_multiprocess(exp_folder: pathlib.Path,
                         available_gpus: List[int],
                         func: Callable[..., None],
                         kwargs: Optional[Dict[str, Any]] = None) -> int:
    """Run an experiment with forced GPU setting (suitable for python mp).

    Args:
        exp_folder (pathlib.Path): Path to the experiement.
        available_gpus (List[int]): List of available GPUs.
        func (Callable[..., None]): Train or eval function.
        kwargs (Optional[Dict[str, Any]]): Additional arguments
            passed to the called functions. Defaults to None.

    Returns:
        int: Status code, 0 is success, 1 is failure.

    """

    if kwargs is None:
        kwargs = dict()

    hparams_file = exp_folder.joinpath('parameters.json')

    with hparams_file.open('r') as file:
        hparams = json.load(file)
    hparams['training']['GPU'] = [available_gpus.pop()]

    filehandler = logging.FileHandler(pathlib.Path(exp_folder, "discern.log"))
    filehandler.setFormatter(
        logging.Formatter(
            '%(asctime)s %(levelname)s:%(name)s: (pid=%(process)d) %(message)s'
        ))
    _LOGGER.addHandler(filehandler)

    with hparams_file.open('w') as file:
        file.write(json.dumps(hparams, sort_keys=False, indent=2))
    try:
        func(exp_folder, **kwargs)
    except Exception as excp:  # pylint: disable=broad-except
        _LOGGER.exception('Exception %s: %s occurred in %s',
                          type(excp).__name__, excp, exp_folder)
        return_code = 1
    else:
        return_code = 0
    finally:
        available_gpus.append(hparams['training']['GPU'][0])
    return return_code


def run_train(exp_folder: pathlib.Path,
              input_path: Optional[pathlib.Path] = None):
    """Run an experiment.

    Args:
        exp_folder (pathlib.Path): Experiments folders.
        input_path (Optional[pathlib.Path]): Input path for the TFRecords, if None the
            experiments folder is used. Defaults to None.

    """
    batch_model, hparams = setup_exp(exp_folder)

    input_path = input_path if input_path else exp_folder
    inputdata = io.DISCERNData.from_folder(
        folder=input_path, batch_size=hparams['training']['batch_size'])

    batch_model.zeros = inputdata.zeros
    if exp_folder.joinpath("job").exists():
        batch_model.restore_model(exp_folder.joinpath("job"))
        if batch_model.wae_model is None:
            _LOGGER.warning(
                "Restoring model from checkpoint failed, building new model")

    if batch_model.wae_model is None:
        batch_model.build_model(
            n_genes=inputdata.var_names.size,
            n_labels=inputdata.obs.batch.cat.categories.size,
            scale=inputdata.config["total_train_count"])

    _LOGGER.debug("Starting training of %s", exp_folder)

    _train(model=batch_model,
           exp_folder=exp_folder.resolve(),
           inputdata=inputdata,
           early_stopping=hparams['training']["early_stopping"],
           max_steps=hparams['training']['max_steps'])

    _LOGGER.info('%s has finished training', exp_folder)


def _train(model: batch_integration.DISCERN, exp_folder: pathlib.Path,
           inputdata: io.DISCERNData, max_steps: int,
           early_stopping: Dict[str, Any]):
    """Helper function to run training."""
    callbacks = customcallbacks.create_callbacks(
        inputdata=inputdata,
        early_stopping_limits=early_stopping,
        exp_folder=exp_folder,
        umap_cells_no=inputdata.config['total_valid_count'],
        profile_batch=2 if (_LOGGER.getEffectiveLevel() <= 30) else 0)

    model.training(savepath=exp_folder.joinpath("job", "best_model.hdf5"),
                   inputdata=inputdata,
                   callbacks=callbacks,
                   max_steps=max_steps)


def setup_exp(
    exp_folder: pathlib.Path
) -> Tuple[batch_integration.DISCERN, Dict[str, Any]]:
    """Setup experiment, by assigning the GPU and parsing the model.

    Args:
        exp_folder (pathlib.Path): Experiment folder.

    Returns:
        Tuple[batch_integration.DISCERN, pathlib.Path, Dict[str, Any]]:
            The model, the output path for training and all parameters.
    """
    tf.keras.backend.clear_session()

    with exp_folder.joinpath('parameters.json').open('r') as file:
        hparams = json.load(file)

    functions.set_gpu_and_threads(hparams['training']['parallel_pc'],
                                  hparams['training']['GPU'])
    tf.config.optimizer.set_jit(hparams['training']['XLA'])

    batch_model = batch_integration.DISCERN.from_json(hparams)

    return batch_model, hparams


def run_projection(exp_folder: pathlib.Path, metadata: List[str],
                   infile: Optional[Union[str, pathlib.Path]],
                   all_batches: bool, store_sigmas: bool):
    """Run projection to metadata on trained model.

    Args:
        exp_folder (pathlib.Path): Folder/ Experiment name to the trained model.
        metadata (List[str]): Metadata to use for integration.
            Should be like List[`column name:value`,...]
        infile (Optional[Union[str, pathlib.Path]]): Alternative input file.
        all_batches: (bool): Project to all batches.
        store_sigmas: (bool): Store sigmas after projection.
    """

    job_folder = exp_folder.joinpath("job")
    if not job_folder.joinpath("best_model.hdf5").exists():
        raise RuntimeError(
            "Could not find saved model in {}".format(job_folder))

    batch_model, hparams = setup_exp(exp_folder)
    _LOGGER.info("The trained model will be utilized on all cells")
    batch_model.restore_model(job_folder)

    if not infile:
        infile = exp_folder.joinpath("processed_data",
                                     "concatenated_data.h5ad")

    inputdata = io.DISCERNData.read_h5ad(
        pathlib.Path(infile), batch_size=hparams['training']['batch_size'])

    if not inputdata.config:
        _LOGGER.warning(
            "Input data does not seem to be preprocessed."
            " Make sure that you use DISCERN preprocessed data only.")

    save_path = exp_folder.joinpath("projected")
    save_path.mkdir(exist_ok=True, parents=True)

    checker = CheckMetaData(inputdata.obs)

    metadata_tuples = [
        checker.check(meta.split(':', maxsplit=1)) for meta in metadata
    ]

    if all_batches:
        for batch in inputdata.obs.batch.cat.categories:
            metadata_tuples.append(("batch", str(batch)))

    batch_model.project_to_metadata(input_data=inputdata,
                                    metadata=metadata_tuples,
                                    save_path=save_path,
                                    store_sigmas=store_sigmas)
    _LOGGER.info("Projection of %s finished", exp_folder)


class CheckMetaData:
    """Check MetaData column value pair in dataframe lazy."""
    # pylint: disable=too-few-public-methods
    _values: Dict[str, Set[str]]
    _template: str
    _data: pd.DataFrame

    def __init__(self, dataframe: pd.DataFrame):
        """Create dict of unique column and value information in dataframe.

        Args:
            dataframe (pd.DataFrame): Input data.
        """
        self._data = dataframe
        self._values = {}
        self._template = "Could not find {type} `{value}´ in {data}, only {available} available."

    def _check_column(self, column: str) -> bool:
        if column in self._values:
            return True
        if column in set(self._data.columns):
            self._values[column] = set(self._data[column].unique())
            return True
        return False

    def check(self, metadata_tuple: List[str]) -> Tuple[str, str]:
        """Check if column value pair is present.

        Args:
            metadata_tuple (List[str]): Column value pair

        Returns:
            Tuple[str, str]: Input column value pair.
        """
        column, value = metadata_tuple
        if not self._check_column(column):
            raise KeyError(
                self._template.format(type="column",
                                      value=column,
                                      data="adata.obs.columns",
                                      available=list(self._values.keys())))
        if not value:
            return column, ""
        if value not in self._values[column]:
            raise ValueError(
                self._template.format(type="value",
                                      value=value,
                                      data=f'adata.obs["{column}"]',
                                      available=self._values[column]))
        return column, value
