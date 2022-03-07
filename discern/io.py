"""discern i/o operations."""
import contextlib
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse as sp_sparse

from discern import functions

_LOGGER = logging.getLogger(__name__)
_COMPRESSION_TYPE: str = 'GZIP'
_TFRECORD_EXTENSION = re.compile(r"\.tfrecords(_v2)?$")


def estimate_csr_nbytes(mat: np.ndarray) -> int:
    """Estimates the size of a sparse matrix generated from numpy array.

    Args:
        mat (np.ndarray): Input array.

    Returns:
        int: Estimated size of the sparse matrix.

    """
    dim1 = mat.shape[0] + 1
    non_zero = np.count_nonzero(mat)
    int32_size = np.dtype(np.int32).itemsize
    return mat.itemsize * non_zero + non_zero * int32_size + dim1 * int32_size


def generate_h5ad(counts: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                  var: pd.DataFrame,
                  obs: pd.DataFrame,
                  save_path: Optional[pathlib.Path] = None,
                  threshold: float = 0.1,
                  **kwargs) -> anndata.AnnData:
    """Generate AnnData format and can save it to file.

    Args:
        counts (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): Count data (X in AnnData).
        var (pd.DataFrame): Variables dataframe.
        obs (pd.DataFrame): Observations dataframe.
        save_path (Optional[pathlib.Path]): Save path for the AnnData in h5py file.
            Defaults to None.
        threshold (float): Set values lower than `threshold` to zero.
            Defaults to 0.1.
        kwargs: Keyword arguments passed to anndata.AnnData.

    Returns:
        anndata.AnnData: The AnnData file.

    """
    if isinstance(counts, np.ndarray):
        estimated_dropouts = None
    else:
        counts, estimated_dropouts = counts

    data = anndata.AnnData(X=counts, var=var, obs=obs, **kwargs)

    if "fixed_scaling" in data.uns:
        data = functions.rescale_by_params(data, data.uns["fixed_scaling"])
        data.uns = {k: v for k, v in data.uns.items() if k != "fixed_scaling"}

    if estimated_dropouts is not None:
        data.layers["estimated_counts"] = data.X.copy()
        data.layers["estimated_dropouts"] = estimated_dropouts
        data.X = functions.sample_counts(data.X,
                                         estimated_dropouts,
                                         var=var,
                                         uns=data.uns)
    data.X[data.X < threshold] = 0
    if data.X.nbytes > estimate_csr_nbytes(data.X):
        data.X = sp_sparse.csr_matrix(data.X)
    if save_path:
        data.write(save_path)
    return data


class DISCERNData(anndata.AnnData):
    """DISCERNData for storing and reading inputs."""

    BUFFER_SIZE = 10000

    def __init__(self,
                 adata: anndata.AnnData,
                 batch_size: int,
                 cachefile: Optional[Union[str, pathlib.Path]] = ''):
        """Create DISCERNData from anndata.

        Args:
            adata (anndata.AnnData): Preprocessed single cell data.
            batch_size (int): Batch size for model.
            cachefile (Optional[Union[str, pathlib.Path]], optional):
                Cachefile for tf.data.Dataset. Defaults to ''.
                '' means cache in memory, None disables cache usage.
        """
        super().__init__(X=adata)
        self._tfdata: Tuple[tf.data.Dataset, tf.data.Dataset] = (None, None)
        self.batch_size = batch_size
        self.cachefile = cachefile

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        batch_size = self._batch_size
        if "split" in self.obs.columns:
            batch_size = self.obs["split"].value_counts()["valid"]
            batch_size = min(self._batch_size, batch_size)
            if batch_size != self._batch_size:
                _LOGGER.info(
                    "Specified batch size %d is bigger than the number of validation cells %d",
                    self._batch_size, batch_size)
                _LOGGER.warning("Adjusting batch size to %d", batch_size)
        return batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int):
        self._batch_size = new_batch_size

    @classmethod
    def from_folder(
            cls,
            folder: pathlib.Path,
            batch_size: int,
            cachefile: Optional[Union[str,
                                      pathlib.Path]] = '') -> "DISCERNData":
        """Read data from DISCERN folder.

        Returns:
            DISCERNData: The data including AnnData and TFRecords.
        """
        adatafile = folder.joinpath('processed_data', 'concatenated_data.h5ad')
        obj = cls.read_h5ad(filename=adatafile,
                            batch_size=batch_size,
                            cachefile=cachefile)

        trainfile = folder.joinpath('TF_records', "training.tfrecords_v2")
        validfile = folder.joinpath('TF_records', "validate.tfrecords_v2")
        if trainfile.exists() and validfile.exists():
            train_data = parse_tfrecords(
                tfr_files=trainfile,
                genes_no=obj.var_names.size,
                n_labels=obj.obs.batch.cat.categories.size)
            valid_data = parse_tfrecords(
                tfr_files=validfile,
                genes_no=obj.var_names.size,
                n_labels=obj.obs.batch.cat.categories.size)
            obj.tfdata = (train_data, valid_data)
        return obj

    @property
    def config(self) -> Dict[str, Any]:
        """Get DISCERN data dependent configuration."""
        self.uns.setdefault("DISCERN", dict())
        return self.uns["DISCERN"]

    @config.setter
    def config(self, data: Dict[str, Any]):
        self.uns.setdefault("DISCERN", dict())
        self.uns["DISCERN"] = data.copy()

    @property
    def zeros(self):
        "Get Zero representation in current data."
        if "fixed_scaling" not in self.uns:
            return 0.0
        mean, var = functions.parse_mean_var(self.var,
                                             self.uns["fixed_scaling"])
        return -mean / var

    @classmethod
    def read_h5ad(
            cls,
            filename: pathlib.Path,
            batch_size: int,
            cachefile: Optional[Union[str,
                                      pathlib.Path]] = '') -> "DISCERNData":
        """Create DISCERNData from anndata H5AD file.

        Returns:
            DISCERNData: The single cell data.
        """
        adata = anndata.read_h5ad(filename)
        return cls(adata=adata, cachefile=cachefile, batch_size=batch_size)

    @property
    def tfdata(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """The accociated tf.data.Datasets.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation data.
        """
        if self._tfdata[0] is None:
            self.tfdata = make_dataset_from_anndata(self, for_tfrecord=False)
        train_data: tf.data.Dataset = self._tfdata[0]
        valid_data: tf.data.Dataset = self._tfdata[1]

        if self.batch_size <= 0:
            raise RuntimeError("Please set batchsize first")

        if self.cachefile is not None:
            _LOGGER.info("Use caching to %s",
                         self.cachefile if self.cachefile else "memory")
            train_data = train_data.cache(
                filename=(str(self.cachefile) +
                          '.train') if self.cachefile else '')
            valid_data = valid_data.cache(
                filename=(str(self.cachefile) +
                          '.valid') if self.cachefile else '')

        train_data = train_data.shuffle(buffer_size=self.BUFFER_SIZE,
                                        reshuffle_each_iteration=True)
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        valid_data = valid_data.shuffle(buffer_size=self.BUFFER_SIZE,
                                        reshuffle_each_iteration=True)
        valid_data = valid_data.batch(self.batch_size, drop_remainder=True)
        return train_data, valid_data

    @tfdata.setter
    def tfdata(self, tfdata: Tuple[tf.data.Dataset, tf.data.Dataset]):
        n_genes = self.var_names.size
        n_labels = self.obs.batch.cat.categories.size
        error = ("Shape ({name}) does not match for {dataset} data:"
                 " Got {got_shape}, expected {expected_shape}")
        for name, data in zip(("training", "validation"), tfdata):
            inputdata = data.element_spec[0]
            outputdata = data.element_spec[1]
            if inputdata["batch_input_dec"].shape != n_labels:
                raise RuntimeError(
                    error.format(name="batch_input_dec",
                                 dataset=name,
                                 got_shape=inputdata["batch_input_dec"].shape,
                                 expected_shape=n_labels))
            if inputdata["batch_input_enc"].shape != n_labels:
                raise RuntimeError(
                    error.format(name="batch_input_enc",
                                 dataset=name,
                                 got_shape=inputdata["batch_input_enc"].shape,
                                 expected_shape=n_labels))
            if inputdata["input_data"].shape != n_genes:
                raise RuntimeError(
                    error.format(name="input_data",
                                 dataset=name,
                                 got_shape=inputdata["input_data"].shape,
                                 expected_shape=n_genes))
            if outputdata[0].shape != n_genes:
                raise RuntimeError(
                    error.format(name="expected1",
                                 dataset=name,
                                 got_shape=outputdata[0].shape,
                                 expected_shape=n_genes))

            if outputdata[1].shape != n_genes:
                raise RuntimeError(
                    error.format(name="expected2",
                                 dataset=name,
                                 got_shape=outputdata[1].shape,
                                 expected_shape=n_genes))
        self._tfdata = (tfdata[0], tfdata[1])

    def __eq__(self, unused_other: object):
        raise NotImplementedError()

    def __getitem__(self, index) -> "DISCERNData":
        adata = super().__getitem__(index=index).copy()
        return DISCERNData(adata=adata,
                           batch_size=self.batch_size,
                           cachefile=self.cachefile)


def parse_tfrecords(tfr_files: Union[pathlib.Path, List[pathlib.Path]],
                    genes_no: int, n_labels: int) -> tf.data.Dataset:
    """Generate TensorFlow dataset from TensorFlow records file(s).

    Args:
        tfr_files (Union[pathlib.Path, List[pathlib.Path]]): TFRecord file(s).
        genes_no (int): Number of genes in the TFRecords.
        n_labels (int): Number of batch labels
        batch_size (int): Size of one batch

    Returns:
         tf.data.Dataset: Dataset containing 'input_data',
            'batch_input_enc' and 'batch_input_dec'

    """
    def parser(serialized_example):  # pragma no cover
        outtype = tf.float32
        total_length = outtype.size * (genes_no + n_labels)
        values = tf.io.decode_raw(tf.reshape(serialized_example, (-1, 1)),
                                  out_type=outtype,
                                  fixed_length=total_length)
        values = tf.reshape(values, (1, -1))
        dense, batch_no = tf.split(values, [genes_no, n_labels], 1)
        dense = tf.squeeze(dense)
        batch_no = tf.reshape(batch_no, shape=(-1, ))

        return {
            'input_data': dense,
            'batch_input_enc': batch_no,
            'batch_input_dec': batch_no
        }, (dense, dense)

    tfr = [str(file) for file in tfr_files] if isinstance(
        tfr_files, list) else str(tfr_files)
    file_paths = tf.data.Dataset.list_files(tfr)
    dataset = tf.data.TFRecordDataset(file_paths,
                                      compression_type='GZIP',
                                      buffer_size=DISCERNData.BUFFER_SIZE,
                                      num_parallel_reads=20)
    dataset = dataset.map(map_func=parser,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def _serialize_numpy(counts: np.ndarray, batch_no: np.ndarray) -> np.ndarray:
    """Serialize numpy arrays to array of bytes."""
    counts = counts.astype(np.float32)
    batch_no = batch_no.astype(np.float32)
    border = np.ones((counts.shape[0], 1), dtype=np.float32)
    data = np.hstack([counts, batch_no, border])
    serialized_data = np.apply_along_axis(lambda x: x.tobytes('C'), 1, data)
    return serialized_data


def make_dataset_from_anndata(
        adata: anndata.AnnData,
        for_tfrecord: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Generate TensorFlow Dataset from AnnData object.

    Args:
        adata (anndata.AnnData): Input cells
        for_tfrecord (bool): make output for writing TFrecords.
            Defaults to False.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]:
            The training and validation datasets
    """
    batch_no = np_one_hot(adata.obs.batch.cat)
    tfdatasets = dict()
    for split in ('train', 'valid'):
        idx = adata.obs.split == split
        currdata = adata[idx]
        curr_batch_no = batch_no[idx]
        if for_tfrecord:
            mapping: np.ndarray = _serialize_numpy(currdata.X, curr_batch_no)
        else:
            mapping = (dict(input_data=currdata.X,
                            batch_input_enc=curr_batch_no,
                            batch_input_dec=curr_batch_no), (currdata.X,
                                                             currdata.X))
        tfdatasets[split] = tf.data.Dataset.from_tensor_slices(mapping)
    return tfdatasets['train'], tfdatasets['valid']


def np_one_hot(labels: pd.Categorical) -> np.ndarray:
    """One hot encode a numpy array.

    Args:
        labels (pd.Categorical): integer values used as indices

    Returns:
        np.ndarray: One hot encoded labels
    """
    _LOGGER.info("Found %d batches in the data set", labels.categories.size)
    one_hot = np.zeros((labels.codes.size, labels.categories.size),
                       dtype=np.float32)
    one_hot[np.arange(labels.codes.size), labels.codes] = 1.0
    return one_hot


class TFRecordsWriter(contextlib.AbstractContextManager):
    """Context manager to be used for writing tf.data.Dataset to TFRecord file.

    Args:
        out_dir (pathlib.Path): Path to the directory where to write the TFRecords.

    Attributes:
        out_dir (str): Path to the directory where to write the TFRecords.
    """

    out_dir: pathlib.Path
    _valid: tf.io.TFRecordWriter
    _train: tf.io.TFRecordWriter

    def __init__(self, out_dir: pathlib.Path):
        """Initialize the class."""
        self.out_dir = pathlib.Path(out_dir)

        self._valid: tf.io.TFRecordWriter = None
        self._train: tf.io.TFRecordWriter = None

    def write_dataset(self, dataset: tf.data.Dataset, split: str):
        """Write tf.data.Dataset to TFRecord specified by split.

        Args:
            dataset (tf.data.Dataset): Dataset to be written.
            split (str): Subfile to use: `train` or `valid`.

        Raises:
            ValueError: If split is not supported.

        """
        if split == 'train':
            self._train.write(dataset)
        elif split == 'valid':
            self._valid.write(dataset)
        else:
            raise ValueError("invalid split: %s" % split)

    def __enter__(self):
        """Initialize the TFRecordWriter objects."""
        self.out_dir.mkdir(exist_ok=True, parents=True)

        train_filename = self.out_dir.joinpath('training.tfrecords_v2')
        self._train = tf.data.experimental.TFRecordWriter(
            str(train_filename), compression_type=_COMPRESSION_TYPE)

        valid_filename = self.out_dir.joinpath('validate.tfrecords_v2')
        self._valid = tf.data.experimental.TFRecordWriter(
            str(valid_filename), compression_type=_COMPRESSION_TYPE)
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # pylint: disable=unused-argument
        "Exit the Contextmanager."
        return None
