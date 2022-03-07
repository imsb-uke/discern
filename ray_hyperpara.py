"""Main modul for hyperparameter search."""
import collections
import importlib
import inspect
import json
import pathlib
from typing import Any, Dict, Union

import anndata
import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from ray import tune
from ray.tune import progress_reporter as tune_reporters
from sklearn import ensemble, metrics, model_selection
from tensorflow.keras import callbacks
from tensorflow.keras import models as keras_models

from discern import functions, io
from discern import mmd as sc_mmd
from discern.estimators import batch_integration
from discern.estimators import callbacks as custom_callbacks
from discern.estimators import customlayers


def _recursive_update(data, updates):
    if not isinstance(data, collections.abc.Mapping) and isinstance(
            updates, collections.abc.Mapping):
        data = {}
    for key, value in updates.items():
        if isinstance(value, collections.abc.Mapping):
            data[key] = _recursive_update(data.get(key, {}), value)
        else:
            data[key] = value
    return data


def parse_json_search_space(space: Dict[str, Any]) -> Dict[str, Any]:
    """Parse search space from JSON (dict-like) filecontent.

    Arguments for functions containing ``log`` are automatically multiplied with np.log(10).

    Args:
        space (Dict[str, Any]): Parsable-search space.

    Returns:
        Dict[str, Any]: Parsed search space, can now directly used for search.

    """
    if "name" in space and "type" in space and "args" in space:
        mod, name = space["type"].rsplit(".", 1)
        func = getattr(importlib.import_module(mod), name)
        args = space["args"]
        if "log" in space["type"]:
            args = [np.log(10) * x for x in space["args"]]
        if mod.startswith("ray.tune"):
            return func(*args)
        return func(space["name"], *args)
    return {
        key: parse_json_search_space(values)
        for key, values in space.items()
    }


def _generate_cells(model: tf.keras.Model, cells: anndata.AnnData,
                    output_labels: np.ndarray,
                    batch_size: int) -> anndata.AnnData:
    n_labels_enc = model.wae_model.input_shape['batch_input_enc'][1]
    n_labels_dec = model.wae_model.input_shape['batch_input_dec'][1]
    labels = tf.one_hot(cells.obs.batch.cat.codes.values.astype(np.int32),
                        depth=n_labels_enc)

    latent, _ = model.generate_latent_codes(counts=cells.X,
                                            batch_labels=labels,
                                            batch_size=batch_size)
    olabels = tf.one_hot(output_labels.astype(np.int32), depth=n_labels_dec)
    decoded = model.generate_cells_from_latent(latent_codes=latent,
                                               output_batch_labels=olabels,
                                               batch_size=batch_size)
    decoded = functions.sample_counts(counts=decoded[0],
                                      probabilities=decoded[1],
                                      var=cells.var,
                                      uns=cells.uns)
    decoded = anndata.AnnData(decoded,
                              obs=cells.obs,
                              var=cells.var,
                              uns=cells.uns)
    return decoded


class HyperParameterEarlyStopping(custom_callbacks.DelayedEarlyStopping):
    """Tensorflow EarlyStopping without resetting metrics on begin of training.

    Args:
        See documentation.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, *args, **kwargs):
        """Initialize the class."""
        super().__init__(*args, **kwargs)
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf  # pylint: disable=comparison-with-callable

    def on_train_begin(self, logs=None):
        """Drop the original on_train_begin function."""
        pass  # pylint: disable=unnecessary-pass


class RAYHYPER(ray.tune.Trainable):
    """Main object for hyperparameter optimization. Trainable for Ray[Tune].

    Contains train methods.

    """

    # pylint: disable=abstract-method
    def __init__(self, *args, **kwargs):
        self.model = None
        self.data = None
        self.batch_to_int = None
        self.train_data = None
        self.valid_data = None
        self.iterations = None
        self.callbacks = None
        super().__init__(*args, **kwargs)

    def setup(self, config):
        self.config['training']["GPU"] = [
            i for i, _ in enumerate(ray.get_gpu_ids())
        ]
        expfolder = pathlib.Path(self.config["exp_folder"]).resolve()
        self.config['logdir'] = self.logdir

        self.config['model']['latent_dim'] = int(
            self.config['model']['latent_dim'])

        self.config["training"]["seed"] = np.random.randint(1000)

        # Writes configuration of the hyperparameters to a parameters.json file.
        with pathlib.Path(self.logdir, 'parameters.json').open('w') as file:
            json.dump(self.config, file, sort_keys=False, indent=4)

        tf.keras.backend.clear_session()
        tf.random.set_seed(self.config["training"]["seed"])

        functions.set_gpu_and_threads(self.config['training']['parallel_pc'],
                                      self.config['training']["GPU"])
        tf.config.optimizer.set_jit(self.config['training']['XLA'])

        self.model = batch_integration.DISCERN.from_json(self.config)

        self.data = io.DISCERNData.from_folder(
            expfolder, batch_size=int(self.config["training"]["batch_size"]))

        self.batch_to_int = {
            val: key
            for key, val in enumerate(self.data.obs.batch.cat.categories)
        }

        self.model.zeros = self.data.zeros

        self.model.build_model(
            n_genes=self.data.var_names.size,
            n_labels=self.data.obs.batch.cat.categories.size,
            scale=self.data.config["total_train_count"])

        self.train_data, self.valid_data = self.data.tfdata
        self.train_data = self.train_data.repeat().prefetch(
            tf.data.experimental.AUTOTUNE)
        self.valid_data = self.valid_data.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.iterations = 0

        self.callbacks = [
            HyperParameterEarlyStopping(
                **self.config['training']["early_stopping"]),
            callbacks.TerminateOnNaN()
        ]

    def step(self):
        batch_size = int(self.config["training"]["batch_size"])
        total = self.data.config["total_train_count"]

        res = self.model.wae_model.fit(x=self.train_data,
                                       epochs=self.iterations +
                                       self.config["epochs_per_timestep"],
                                       validation_data=self.valid_data,
                                       callbacks=self.callbacks,
                                       steps_per_epoch=total // batch_size,
                                       initial_epoch=self.iterations,
                                       verbose=0)

        self.iterations += self.config["epochs_per_timestep"]
        reached_max = self.iterations >= self.config["training"]["max_steps"]
        final_results = {
            "early_stopped": self.model.wae_model.stop_training,
            "reached_max_step": reached_max,
            'loss': np.min(res.history.get('val_loss', 500000)),
            'mmd': 0.0,
            "auc": 0.0,
            "mmd_autoencoded": 0.0,
            "auc_autoencoded": 0.0,
        }
        if not np.isfinite(res.history['loss']).all():
            final_results.update(dict(mmd=500, loss=500000, auc=100))
            return final_results
        return self._loss(final_results)

    def _loss(self, results):
        batch_size = int(self.config["training"]["batch_size"])
        sigma = self.data.config["mmd_kernel"]
        mmd_data = self.data[self.data.obs.for_mmd]
        for i in self.config["batch_to_check"]:
            i = self.batch_to_int[i]
            decoded_cells = _generate_cells(
                self.model, mmd_data,
                np.full_like(mmd_data.obs.batch.cat.codes.values, i),
                batch_size)

            recent = loss(decoded_cells, mmd_data, sigma["all"], i)
            results["mmd"] += recent.mmd
            results["auc"] += recent.auc
            results["mmd_autoencoded"] += recent.mmd_autoencoded
            results["auc_autoencoded"] += recent.auc_autoencoded
        return results

    def save_checkpoint(self, tmp_checkpoint_dir: str):
        savedir = pathlib.Path(tmp_checkpoint_dir, "model.hdf5").resolve()
        self.model.wae_model.save(str(savedir))
        return str(savedir)

    def load_checkpoint(self, checkpoint: str):
        savedir = pathlib.Path(checkpoint, "model.hdf5").resolve()
        self.model.wae_model = keras_models.load_model(
            str(savedir),
            compile=False,
            custom_objects=customlayers.getmembers())
        optimizer = self.model.get_optimizer()
        self.model.compile(optimizer)

    def cleanup(self):
        self.data = None
        self.train_data = None
        self.valid_data = None
        self.callbacks = None
        self.model = None
        tf.keras.backend.clear_session()
        importlib.import_module("gc").collect()


def _generate_trainspec(exp_folder: pathlib.Path, train_spec: Dict[str, Any],
                        batchname_to_check: pd.Series) -> Dict[str, Any]:

    with exp_folder.joinpath('parameters.json').open('r') as file:
        hparams = json.load(file)

    train_spec["config"] = _recursive_update(train_spec["config"], hparams)
    train_spec["config"]["exp_folder"] = str(exp_folder)
    train_spec["config"]["json_file"] = str(
        exp_folder.joinpath('parameters.json'))

    n_batches = min(len(batchname_to_check),
                    train_spec['config'].get('validation_batches', 10))

    train_spec["config"]["batch_to_check"] = batchname_to_check.iloc[
        0:n_batches].index.tolist()
    train_spec["local_dir"] = str(exp_folder)
    return train_spec


def _create_progressreporter(
        *columns: str,
        max_progress_rows: int = 20) -> tune_reporters.CLIReporter:
    """Add a custom metric column, in addition to the default metrics.

    Args:
        *columns (str): Columns to report.
        Note that this must be a metric that is returned in your training results.
        max_progress_rows (int, optional): Row limit for command line report.
            Defaults to 20.

    Returns:
        tune_reporters.CLIReporter: Reporter to be used with `tune.run`.
    """

    reporter = tune_reporters.CLIReporter(max_progress_rows=max_progress_rows)
    for column in columns:
        if column not in reporter._metric_columns:  # pylint: disable=protected-access
            reporter.add_metric_column(column)
    return reporter


def create_scheduler(schedulername: str, **initargs: Any) -> Any:
    """Create a scheduler based on the module string.

    Args:
        schedulername (str): Name of the scheduler,
            like `ray.tune.schedulers.AsyncHyperBandScheduler`.

    Returns:
        Any: A Scheduler.
    """

    mod, classname = schedulername.rsplit(".", 1)
    scheduler_cls = getattr(importlib.import_module(mod), classname)
    valid_args = inspect.getfullargspec(scheduler_cls).args
    args = {k: v for k, v in initargs.items() if k in valid_args}
    return scheduler_cls(**args)


def create_search_algo(name: str, space: Dict[str, Any],
                       **initargs: Any) -> Any:
    """Create search algo from string.

    Randomsearch requires that all parameters
    in the search space are defined as attributes from `ray.tune`.
    If using ray.tune.suggest.hyperopt.HyperOptSearch´, your
    parameter types should start with `hyperopt.hp.`.

    Args:
        name (str): Full name of the search algorithm.
            Like `Gridsearch` or `ray.tune.suggest.hyperopt.HyperOptSearch`
        space (Dict[str, Any]): Search space to be used.

    Returns:
        Any: The search algorithm (or None for Randomsearch)
    """

    if name == "Randomsearch":
        return None, space
    mod, classname = name.rsplit(".", 1)
    initargs.setdefault("space", {})
    initargs["space"] = _recursive_update(initargs["space"], space)
    searchalgo = getattr(importlib.import_module(mod), classname)
    valid_args = inspect.getfullargspec(searchalgo).args
    args = {k: v for k, v in initargs.items() if k in valid_args}
    return searchalgo(**args), {}


def create_experiment(name: str, searchspace: Dict[str, Any],
                      **initargs: Any) -> tune.Experiment:
    """Create a Experiment.

    Args:
        name (str): Name of the experiment.
        searchspace (Dict[str, Any]): Search space to use.
        **initargs (Any): Initialization arguments.

    Returns:
        tune.Experiment: [description]
    """
    initargs.setdefault("config", dict())
    initargs["config"] = _recursive_update(initargs["config"], searchspace)
    return tune.Experiment(name, run=RAYHYPER, **initargs)


def tune_hyperparameters(exp_folder: Union[str, pathlib.Path]):
    """Man function for tuning hyperparameters.

    Args:
        exp_folder (str): Path to the experiments folder.

    """
    exp_folder = pathlib.Path(exp_folder).resolve()
    with exp_folder.joinpath('hyperparameter_search.json').open('r') as file:
        parameter = json.load(file)

    ray.init(local_mode=False, **parameter["setup"])

    eval_data = anndata.read_h5ad(
        exp_folder.joinpath('processed_data', 'concatenated_data.h5ad')).obs

    space = parse_json_search_space(parameter["search_space"])
    train_spec = _generate_trainspec(
        exp_folder, parameter["train_spec"],
        eval_data[eval_data['for_mmd']]['batch'].value_counts())

    algo, searchspace = create_search_algo(parameter["searcher"].pop("name"),
                                           space=space,
                                           **parameter["searcher"])

    scheduler = create_scheduler(parameter["scheduler"].pop("name"),
                                 **parameter["scheduler"])

    reporter = _create_progressreporter("early_stopped", "loss", "mmd", "auc")
    experiment = create_experiment(train_spec.pop("name"), searchspace,
                                   **train_spec)

    tune.run(
        experiment,  # type: ignore
        scheduler=scheduler,
        search_alg=algo,
        verbose=True,
        queue_trials=True,
        raise_on_failed_trial=False,
        resume=False,
        with_server=False,
        progress_reporter=reporter)

    if "analysis" in parameter:
        folder = exp_folder.joinpath(experiment.name)
        analysis = tune.Analysis(str(folder))
        data = analysis.dataframe(metric=parameter["analysis"]["metric"],
                                  mode=parameter["analysis"]["mode"])
        data = data.to_dict(orient="index")
        with folder.joinpath("analysis.json").open("w") as file:
            json.dump(data, file)

    ray.shutdown()


def _scale_cells(adata: anndata.AnnData) -> anndata.AnnData:
    if 'fixed_scaling' in adata.uns:
        adata = functions.rescale_by_params(adata, adata.uns['fixed_scaling'])
    return functions.scale(adata)


LossResults = collections.namedtuple(
    "LossResults", ["auc", "mmd", "mmd_autoencoded", "auc_autoencoded"])


def loss(random_cells: anndata.AnnData, valid_cells: anndata.AnnData,
         sigma: float, batch_no: int) -> LossResults:
    """Compute several losses.

    Losses between random cells and valid cells and calculates the AUC using
    random forest classifier and IDR and performs DE gene tests.

    Args:
        random_cells (anndata.AnnData): Generated cells.
        valid_cells (anndata.AnnData):Valid cells.
        sigma (float): Sigma value.
        batch_no (int): Batch number of projected batch.

    Returns:
        Dict[str, float]: MMD loss between projected, autoencoded and valid cells.

    """
    mmd_valid_cells = valid_cells[:, valid_cells.var['pca_genes']].copy()
    mmd_valid_cells = _scale_cells(mmd_valid_cells)
    mmd_valid_cells_pca_space = np.matmul(mmd_valid_cells.X,
                                          mmd_valid_cells.varm['PCs'])

    mmd_random_cells = random_cells[:, valid_cells.var['pca_genes']].copy()
    mmd_random_cells = _scale_cells(mmd_random_cells)
    mmd_random_cells_pca_space = np.matmul(mmd_random_cells.X,
                                           mmd_valid_cells.varm['PCs'])

    valid_cells_idx = valid_cells.obs.batch.cat.codes == batch_no

    mmd = sc_mmd.mmd_loss(
        valid_cells=mmd_valid_cells_pca_space[valid_cells_idx],
        random_cells=mmd_random_cells_pca_space[~valid_cells_idx],
        sigma=sigma)

    auc = _classify_rfc(mmd_random_cells_pca_space[~valid_cells_idx],
                        mmd_valid_cells_pca_space[valid_cells_idx])

    mmd_autoencoded = sc_mmd.mmd_loss(
        valid_cells=mmd_valid_cells_pca_space[valid_cells_idx],
        random_cells=mmd_random_cells_pca_space[valid_cells_idx],
        sigma=sigma)

    auc_autoencoded = _classify_rfc(
        mmd_random_cells_pca_space[valid_cells_idx],
        mmd_valid_cells_pca_space[valid_cells_idx])

    return LossResults(mmd=mmd,
                       auc=auc,
                       mmd_autoencoded=mmd_autoencoded,
                       auc_autoencoded=auc_autoencoded)


def _classify_rfc(projected_random: np.ndarray,
                  projected_valid: np.ndarray,
                  n_jobs: int = 1) -> float:
    """Trains a random forest classifier and tries to distinguish datasets."""
    data = np.concatenate((projected_random, projected_valid))
    target = np.zeros(projected_random.shape[0] + projected_valid.shape[0])
    target[projected_valid.shape[0]:] = 1

    rf_classifier = ensemble.RandomForestClassifier(n_estimators=100,
                                                    min_samples_leaf=0.1,
                                                    max_depth=3,
                                                    max_features=5,
                                                    n_jobs=n_jobs,
                                                    class_weight="balanced")
    n_splits = 5
    cvsplit = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)

    auc_accum = 0
    for train, test in cvsplit.split(data, target):
        pred_target = rf_classifier.fit(data[train],
                                        target[train]).predict(data[test])
        auc = metrics.roc_auc_score(target[test], pred_target)

        auc_accum += auc / n_splits
    return auc_accum
