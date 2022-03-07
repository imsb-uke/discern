#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Contains the GeneMatrix class, used to represent the scRNA-seq data."""
import json
import logging
import pathlib
from typing import Any, Dict, List, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp_sparse
from numpy import random as nprandom
from sklearn import neighbors as sk_neighbors

from discern import functions, io

_LOGGER = logging.getLogger(__name__)


def read_raw_input(file_path: pathlib.Path) -> anndata.AnnData:
    """Read input and converts it to anndata.AnnData object.

    Currently h5, h5ad, loom, txt and a directory with matrix.mtx. genes.tsv and
    optional barcodes.tsv is supported.

    Args:
        file_path (pathlib.Path): (File-) Path to the input data.

    Returns:
        anndata.AnnData: The read AnnData object.

    Raises:
        ValueError: Datatype of input could not be interfered.

    """
    # 1) read the data set containing one batch cells
    file_path = pathlib.Path(file_path).resolve()
    file_format = file_path.suffix

    if file_format == '.h5':
        andata = sc.read_10x_h5(file_path)

    elif file_format in ('.h5ad', '.loom'):
        andata = sc.read(file_path)

    elif file_format == '.txt':
        andata = sc.read_text(file_path, delimiter='\t').transpose()

    elif file_path.is_dir():
        andata = sc.read(file_path.joinpath('matrix.mtx'),
                         cache=True).T  # transpose the data
        andata.var_names = pd.read_csv(file_path.joinpath('genes.tsv'),
                                       header=None,
                                       sep='\t')[1]
        if file_path.joinpath('barcodes.tsv').exists():
            andata.obs_names = pd.read_csv(file_path.joinpath('barcodes.tsv'),
                                           header=None)[0]
    else:
        raise ValueError(
            'Reading [ %s ] failed, the inferred file format [ %s ]'
            ' is not supported. Please convert your file to either '
            ' h5 or h5ad format.' % (str(file_path), file_format))

    # 2) Make genes unique
    andata.var_names_make_unique()

    # 3) convert to dense data if it is sparse
    if sp_sparse.issparse(andata.X):
        andata.X = andata.X.toarray()

    return andata


# yapf: disable
def merge_data_sets(
        raw_inputs: Dict[str, anndata.AnnData],
        batch_keys: Dict[str,str],
) -> Tuple[anndata.AnnData, Dict[int, str]]:
    # yapf: enable
    """Merge a dictionary of AnnData files to a single AnnData object.

    Args:
        raw_inputs (Dict[str, anndata.AnnData]): Names and AnnData objects.

    Returns:
        Tuple[anndata.AnnData, Dict[int, str]]: Merged AnnData and
        mapping from codes to names.

    """
    for key, adata in raw_inputs.items():
        if "batch" not in adata.obs.columns:
            adata.obs["batch"] = key
        adata.obs.batch = adata.obs[batch_keys[key]]
        if "dataset" in adata.obs.columns:
            _LOGGER.warning("Overwriting `dataset` column with filename,"
                            " if you need this column, rename it beforehand.")
        adata.obs["dataset"] = key

    sc_raw = anndata.AnnData.concatenate(*raw_inputs.values(),
                                         join='inner',
                                         batch_key=None,
                                         index_unique=None)
    sc_raw.obs_names_make_unique()
    sc_raw.obs.batch = sc_raw.obs.batch.astype("category")

    return sc_raw, dict(enumerate(sc_raw.obs.batch.cat.categories))


def _integer_to_float(value: Union[int, float], total: int, key: str) -> float:
    ratio = float(value)
    if int(value) == ratio and ratio > 1.0:
        ratio = value / total
        _LOGGER.warning(
            "Specifying the number of %s cells as "
            "integer is deprecated, please use a fraction instead.", key)
    asrt_msg = "Ratio of {} cells should be in [0,1], got {}"
    if not 0 < ratio <= 1.0:
        raise ValueError(asrt_msg.format(key, ratio))
    return ratio


class WAERecipe:
    """For storing and processing data.

    Can apply filtering, clustering. merging and splitting.

    Args:
        params (Dict[str,Any]): Default parameters for preprocessing.
        inputs (Dict[str,anndata.AnnData]): Input AnnData with batchname as
            dict-key. Defaults to None.
        input_files (List[pathlib.Path]): Paths to raw input data.
        Defaults to None.
        n_jobs (int): Number of jobs/processes to use. Defaults to -1.

    Attributes:
        sc_raw (io.DISCERNData): Read and concatenated input data.
        config (Dict[str, Any]): Parameters calculated during preprocessing.
        params (Dict[str,Any]): Default parameters for preprocessing.

    """

    sc_raw: io.DISCERNData
    params: Dict[str, Any]

    def __init__(self,
                 params: Dict[str, Any],
                 inputs: Dict[str, anndata.AnnData] = None,
                 input_files: Union[Dict[pathlib.Path, str],
                                    List[pathlib.Path]] = None,
                 n_jobs: int = -1):
        """Initialize the class and set threading values."""
        if inputs is None and input_files is None:
            raise AssertionError("Either `inputs` or `input_files` need"
                                 " to be provided.")
        raw_input_ds = inputs if inputs is not None else {}
        if isinstance(input_files, list):
            input_files = {file: "batch" for file in input_files}
        files: Dict[pathlib.Path,
                    str] = input_files if input_files is not None else {}
        batch_columns: Dict[str, str] = {key: "batch" for key in raw_input_ds}
        for ds_path, batch in files.items():
            ds_path = pathlib.Path(ds_path).resolve()
            ds_name = ds_path.name if ds_path.is_dir() else ds_path.stem
            raw_input_ds[ds_name] = read_raw_input(ds_path)
            batch_columns[ds_name] = batch if batch else "batch"

        sc_raw, batch_integer_dict = merge_data_sets(raw_input_ds,
                                                     batch_columns)

        self.sc_raw = io.DISCERNData(sc_raw, batch_size=-1)

        self.params = params

        self.sc_raw.config = {
            'scale': {},
            'mmd_kernel': {},
            'batch_ratios': {},
            'valid_cells_no': {},
            'train_cells_no': {},
            'batch_key': batch_integer_dict,
        }
        self._n_jobs = n_jobs
        sc.settings.n_jobs = n_jobs

    @property
    def config(self):
        """Configuration from preprocessing."""
        return self.sc_raw.config

    def _filter_cells(self, min_genes: int):
        shape = self.sc_raw.X.shape
        sc.pp.filter_cells(self.sc_raw, min_genes=min_genes, inplace=True)
        _LOGGER.debug(
            "Filtering of the raw data is done with minimum %d genes per cell",
            min_genes)
        if self.sc_raw.X.shape[0] < shape[0]:
            _LOGGER.info("Filtering removed %d cells",
                         shape[0] - self.sc_raw.X.shape[0])

    def filtering(self, min_genes: int, min_cells: int):
        """Apply filtering in-place.

        Args:
            min_genes (int): Minimum number of genes to be present for cell to be considered.
            min_cells (int):  Minimum number of cells to be present for gene to be considered.

        """
        self._filter_cells(min_genes=min_genes)
        shape = self.sc_raw.X.shape

        sc.pp.filter_genes(self.sc_raw, min_cells=min_cells, inplace=True)
        _LOGGER.debug(
            "Filtering of the raw data is done with minimum %d cells per gene",
            min_cells)

        if self.sc_raw.X.shape[1] < shape[1]:
            _LOGGER.info("Filtering removed %d genes",
                         shape[1] - self.sc_raw.X.shape[1])

    def scaling(self, scale: int):
        """Apply scaling in-place.

        Args:
            scale (int): Value use to scale with LSN.

        """
        scale = int(scale)
        total = np.sum(self.sc_raw.X, axis=1, keepdims=True)
        self.sc_raw.obs['n_counts'] = total
        np.multiply(self.sc_raw.X, scale / total, out=self.sc_raw.X)
        self.config["scale"]['LSN'] = scale

    def celltypes(self):
        """Aggregate celltype information."""

        if "celltype" not in self.sc_raw.obs_keys():
            columns = ["cluster_names", "cluster"]
            for col in columns:
                if col in self.sc_raw.obs_keys():
                    self.sc_raw.obs.rename(columns={col: "celltype"},
                                           inplace=True)
                    break

        if "celltype" not in self.sc_raw.obs_keys():
            return
        celltypes = self.sc_raw.obs.celltype.astype('category')
        self.sc_raw.obs.celltype = celltypes
        return

    def kernel_mmd(self, neighbors_mmd: int = 50, no_cells_mmd: int = 2000):
        """Apply kernel mmd metrics based on nearest neighbors in-place.

        Args:
            neighbors_mmd (int): Number of neighbors Defaults to 50.
            no_cells_mmd (int): Number of cells used for calculation of mmd. Defaults to 2000.
            projector(Optional[np.ndarray]): PCA-Projector to compute distancs
                in precomputed PCA space. Defaults to None.

        """
        mmd_obj = self.sc_raw[:, self.sc_raw.var['pca_genes']].copy()
        mmd_obj = functions.scale(mmd_obj)

        mmd_cells = nprandom.choice(mmd_obj.shape[0],
                                    min(mmd_obj.shape[0], no_cells_mmd),
                                    replace=False)
        real_mmd = mmd_obj.X[mmd_cells, :]
        real_mmd = np.matmul(real_mmd, mmd_obj.varm['PCs'])

        neighbors = int(np.ceil(min(real_mmd.shape[0] / 2, neighbors_mmd)))

        nbrs = sk_neighbors.NearestNeighbors(
            n_neighbors=neighbors,
            n_jobs=self._n_jobs,
            metric='euclidean',
            algorithm='ball_tree').fit(real_mmd)
        distances, _ = nbrs.kneighbors(real_mmd)

        # nearest neighbor is the point so we need to exclude it
        self.config["mmd_kernel"]["all"] = np.median(
            np.mean(distances[:, 1:neighbors_mmd], axis=1))**2

        _LOGGER.info('Computed nearest neighbour distances. Value: %f',
                     self.config["mmd_kernel"]['all'])

    def split(self,
              split_seed: int,
              valid_cells_ratio: Union[int, float],
              mmd_cells_ratio: Union[int, float] = 1.):
        """Split cells to train and validation set.

        Args:
            split_seed (int): Seed used with numpy.
            valid_cells_ratio (Union[int,float]): Number or ratio of cells in
                the validation set.
            mmd_cells_ratio (Optional[Union[int, float]]): Number of validation
            cells to use for mmd calculation during hyperparameter optimization.
                Defaults to 1. which is valid_cells_no.

        """
        valid_cells_ratio = _integer_to_float(valid_cells_ratio,
                                              self.sc_raw.shape[0],
                                              "validation")

        mmd_cells_ratio = _integer_to_float(
            mmd_cells_ratio, self.sc_raw.shape[0] * valid_cells_ratio, "MMD")

        nprandom.seed(split_seed)

        self.sc_raw.obs['split'] = pd.Categorical(
            values=np.repeat('train', self.sc_raw.shape[0]),
            categories=sorted(['valid', 'train']))
        # go through every batch and take percentage of it
        for batch in np.unique(self.sc_raw.obs.batch):

            # set the training and validation cells first
            batch_indices = self.sc_raw[self.sc_raw.obs['batch'] ==
                                        batch].obs.index
            valid_indices = nprandom.choice(batch_indices,
                                            int(valid_cells_ratio *
                                                batch_indices.shape[0]),
                                            replace=False)
            try:
                self.sc_raw.obs.at[valid_indices, 'split'] = 'valid'
            except AttributeError as exception:
                cols = set(self.sc_raw.obs.columns[
                    self.sc_raw.obs.columns.duplicated()])
                raise NotImplementedError(
                    "DISCERN preprocessing currently does not support"
                    " duplicated column names, currently %s are duplicated" %
                    cols) from exception

            self.config["valid_cells_no"][batch] = len(valid_indices)
            self.config["train_cells_no"][batch] = len(batch_indices) - len(
                valid_indices)
            self.config["batch_ratios"][batch] = len(
                batch_indices) / self.sc_raw.shape[0]

        self.config["total_count"] = self.sc_raw.shape[0]
        self.config["common_genes_count"] = self.sc_raw.shape[1]
        counts_by_split = self.sc_raw.obs.split.value_counts(sort=False)
        self.config["total_train_count"] = int(counts_by_split['train'])
        self.config["total_valid_count"] = int(counts_by_split['valid'])

        self.sc_raw.obs['for_mmd'] = False

        valid_cells = self.sc_raw.obs[self.sc_raw.obs.split == 'valid']

        stratif_data = valid_cells['batch'].astype(str)
        if "celltype" in valid_cells.columns:
            stratif_data += valid_cells['celltype'].astype(str)

        freq = stratif_data.value_counts(normalize=True, sort=False).to_frame()
        freq.columns = ['frequency']
        stratif_data = stratif_data.to_frame()
        stratif_data = pd.merge(stratif_data,
                                freq,
                                left_on=stratif_data.columns[0],
                                right_index=True)
        indices = stratif_data.sample(frac=mmd_cells_ratio,
                                      random_state=split_seed,
                                      replace=False,
                                      weights="frequency").index
        self.sc_raw.obs.at[indices, 'for_mmd'] = True

    def projection_pca(self, pcs: int = 25):
        """Apply PCA projection.

        Args:
            pcs (int): Number of principle components. Defaults to 32.

        """
        log_values = np.log1p(self.sc_raw.X)
        mean = np.mean(log_values, axis=0, dtype=np.float64)
        mean_sq = np.multiply(log_values, log_values).mean(axis=0,
                                                           dtype=np.float64)
        var_2 = mean_sq - mean**2
        self.sc_raw.var['mean_scaling'] = mean
        var = np.sqrt(var_2)
        var[var < 1e-8] = 1e-8
        self.sc_raw.var['var_scaling'] = var

        pc_array = self.sc_raw.copy()

        sc.pp.filter_genes_dispersion(pc_array,
                                      flavor='cell_ranger',
                                      n_top_genes=1000,
                                      log=False,
                                      subset=False)

        self.sc_raw.var['pca_genes'] = pc_array.var['highly_variable']
        pc_array = pc_array[:, pc_array.var['highly_variable']].copy()

        sc.pp.log1p(pc_array)
        functions.scale(pc_array)

        sc.tl.pca(pc_array, n_comps=pcs, svd_solver='arpack')

        pc_output = np.zeros(shape=[self.sc_raw.n_vars, pcs])

        pc_output[self.sc_raw.var['pca_genes']] = pc_array.varm['PCs']
        self.sc_raw.varm['PCs'] = pc_output

    @classmethod
    def from_path(cls, job_dir: pathlib.Path) -> "WAERecipe":
        """Create WAERecipe from DISCERN directory.

        Returns:
            WAERecipe: The initalized object.
        """
        param_path = job_dir.joinpath('parameters.json')

        with param_path.open('r') as file:
            hparam = json.load(file)

        functions.set_gpu_and_threads(hparam['training']['parallel_pc'], [])

        scd = cls(input_files=hparam['input_ds']['raw_input'],
                  n_jobs=hparam['input_ds']['n_cpus_preprocessing'],
                  params=hparam)
        return scd

    def mean_var_scaling(self):
        """Apply Mean-Variance scaling if 'fixed_scaling' is present in params."""
        if 'fixed_scaling' in self.params['input_ds']['scale']:
            _LOGGER.info(
                "Found 'fixed_scaling' in parameters, cells will be scaled.")
            scale_dict = self.params['input_ds']['scale']['fixed_scaling']
            self.sc_raw = functions.scale_by_params(self.sc_raw, scale_dict)
            self.sc_raw.uns['fixed_scaling'] = scale_dict
            _LOGGER.debug("Scaling of cells finished.")

    def __call__(self) -> "WAERecipe":
        """Run the Recipe.

        Returns:
            WAERecipe: The applied Recipe.
        """
        self.filtering(
            min_genes=self.params['input_ds']['filtering']['min_genes'],
            min_cells=self.params['input_ds']['filtering']['min_cells'])

        self.celltypes()
        _LOGGER.debug("Celltype processing finished.")
        if "LSN" in self.params['input_ds']['scale']:
            self.scaling(scale=self.params['input_ds']['scale']["LSN"])
            _LOGGER.info("Scaling of the data is done using LSN with %s",
                         self.params['input_ds']['scale']["LSN"])
        else:
            _LOGGER.info("Continues without library size normalization")

        _LOGGER.debug("Splitting cells to train and valid...")
        self.split(
            split_seed=self.params['input_ds']['split']['split_seed'],
            valid_cells_ratio=self.params['input_ds']['split']['valid_cells'],
            mmd_cells_ratio=self.params['input_ds']['split'].get(
                "mmd_cells", 1.0))

        _LOGGER.debug("Computing PCA...")
        self.projection_pca(pcs=32)

        _LOGGER.debug("Log1p of cells...")
        sc.pp.log1p(self.sc_raw)

        _LOGGER.debug("Calculating kernel mmd of cells...")
        self.kernel_mmd(neighbors_mmd=50, no_cells_mmd=10000)
        _LOGGER.debug("Kernel mmd finished.")

        sc.pp.filter_genes_dispersion(self.sc_raw, log=False, subset=False)
        _LOGGER.debug("Filtering gene dispersions finished.")

        self.mean_var_scaling()

        _LOGGER.debug("WAE recipe done")
        return self

    def dump(self, job_dir: pathlib.Path):
        """Dump recipe results to directory.

        Args:
            job_dir (pathlib.Path): The directory to save the results at.
        """
        file_path = job_dir.joinpath('processed_data')
        file_path.mkdir(exist_ok=True, parents=True)
        file_name = file_path.joinpath('concatenated_data.h5ad')

        _LOGGER.info("Writing data to %s.", file_name)
        self.sc_raw.write(file_name)
        _LOGGER.debug("Writing data finished.")

        param_path = job_dir.joinpath('parameters.json')

        with param_path.open('w') as file:
            json.dump(self.params, file, sort_keys=False, indent=4)

    def dump_tf_records(self, path: pathlib.Path):
        """Dump the TFRecords to disk.

        Args:
            path (pathlib.Path): Folder to save the TFrecords in.
        """
        tfdatasets = io.make_dataset_from_anndata(self.sc_raw,
                                                  for_tfrecord=True)

        with io.TFRecordsWriter(path) as writer:
            for split, tfdataset in zip(('train', 'valid'), tfdatasets):
                _LOGGER.debug("Start serializing dataset of split ``%s``",
                              split)
                writer.write_dataset(tfdataset, split)
        _LOGGER.info(' Writing TF records in: %s is completed. ', path)


def read_process_serialize(job_path: pathlib.Path,
                           with_tfrecords: bool = True):
    """Read data, preprocesses it and write output as anndata.AnnData and TFRecords.

    Args:
        job_path (pathlib.Path): Path of the experiments folder.
        with_tfrecords (bool): write tfrecord files. Defaults to True.

    """
    job_path = pathlib.Path(job_path).resolve()
    logging.getLogger("anndata").setLevel(_LOGGER.getEffectiveLevel())
    recipe = WAERecipe.from_path(job_path)
    recipe = recipe()
    recipe.dump(job_dir=job_path)
    if with_tfrecords:
        recipe.dump_tf_records(path=job_path.joinpath('TF_records'))
