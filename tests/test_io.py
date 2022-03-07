"""Tests discern.io."""
import pathlib
from contextlib import ExitStack as no_raise

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import tensorflow as tf
from scipy import sparse as sp_sparse

from discern import functions, io


def _get_csr_matrix_size(mat):
    return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes


def _write_tfrecords(dataset, job_path):
    worker_path = job_path.joinpath('TF_records')
    tfdatasets = io.make_dataset_from_anndata(dataset, for_tfrecord=True)

    with io.TFRecordsWriter(worker_path) as writer:
        for split, tfdataset in zip(('train', 'valid'), tfdatasets):
            writer.write_dataset(tfdataset, split)


@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.float32, np.float64))
def test_estimate_csr_nbytes(dtype):
    """Test estimate_csr_nbytes."""
    for _ in range(10):
        dim1, dim2 = np.random.randint(1, 1000, 2)
        mat = np.random.randint(0, 5, dim1 * dim2)
        mat = mat.astype(dtype)
        mat = mat.reshape((dim1, dim2))
        sparse_z = sp_sparse.csr_matrix(mat)
        true_size = _get_csr_matrix_size(sparse_z)
        estimated_size = io.estimate_csr_nbytes(mat)
        assert estimated_size == true_size


def test_tfrecords(tmp_path, anndata_file):
    """Test reading and writing tfrecords file."""
    anndata_file = anndata_file(200)
    n_genes = anndata_file.X.shape[1]
    nlabels = len(anndata_file.obs.batch.cat.categories)
    _write_tfrecords(anndata_file, tmp_path)
    train_files, valid_files = functions.prepare_train_valid(
        tmp_path.joinpath('TF_records'))
    data = io.parse_tfrecords(tfr_files=train_files,
                              genes_no=n_genes,
                              n_labels=nlabels)
    all_vals = []
    for inputs, outputs in data:
        np.testing.assert_array_equal(inputs['input_data'], outputs[0])
        np.testing.assert_array_equal(inputs['input_data'], outputs[1])
        np.testing.assert_array_equal(inputs['batch_input_enc'],
                                      inputs['batch_input_dec'])
        all_vals.append(outputs[0])
    all_vals = tf.stack(all_vals).numpy()
    all_vals = np.squeeze(all_vals)
    all_mean = all_vals.mean(axis=0)
    all_std = all_vals.std(axis=0)
    expected = anndata_file.X[anndata_file.obs.split == "train"]
    assert all_vals.shape == expected.shape
    expected_mean = expected.mean(axis=0)
    expected_std = expected.std(axis=0)
    np.testing.assert_allclose(all_mean, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(all_std, expected_std, rtol=3e-4)
    data = io.parse_tfrecords(tfr_files=valid_files,
                              genes_no=n_genes,
                              n_labels=2)
    all_vals = []
    for inputs, outputs in data:
        np.testing.assert_array_equal(inputs['input_data'], outputs[0])
        np.testing.assert_array_equal(inputs['input_data'], outputs[1])
        np.testing.assert_array_equal(inputs['batch_input_enc'],
                                      inputs['batch_input_dec'])
        all_vals.append(outputs[0])
    all_vals = tf.stack(all_vals).numpy()
    all_vals = np.squeeze(all_vals)
    all_mean = all_vals.mean(axis=0)
    all_std = all_vals.std(axis=0)
    expected = anndata_file.X[anndata_file.obs.split == "valid"]
    assert all_vals.shape == expected.shape
    expected_mean = expected.mean(axis=0)
    expected_std = expected.std(axis=0)
    np.testing.assert_allclose(all_mean, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(all_std, expected_std, rtol=3e-4)


@pytest.mark.parametrize("threshold, expect_sparse", [(0, False), (0.7, True)])
@pytest.mark.parametrize("with_rescaling", (True, False))
def test_generate_h5ad(tmp_path, anndata_file, threshold, expect_sparse,
                       with_rescaling, monkeypatch):
    """Test generate_h5ad."""
    anndata_file = anndata_file(10)
    anndata_file = anndata_file[:, 0:100]
    anndata_file.obs.split = anndata_file.obs.split.astype("category")
    outputfile = tmp_path.joinpath('tmpfile.h5ad')
    counts = np.random.rand(10, 100).astype(np.float32)

    var = anndata_file.var.copy()
    obs = anndata_file.obs.copy()
    uns = anndata_file.uns.copy()
    if with_rescaling:
        anndata_file.uns["fixed_scaling"] = True
    expected_counts = counts.copy()
    expected_counts[expected_counts < threshold] = 0

    def _check_rescale(adata, scaling):
        assert isinstance(adata, anndata.AnnData)
        np.testing.assert_equal(adata.X, counts)
        pd.testing.assert_frame_equal(adata.var, var)
        pd.testing.assert_frame_equal(adata.obs, obs)
        assert adata.uns["fixed_scaling"]
        assert scaling
        return adata

    monkeypatch.setattr(functions, "rescale_by_params", _check_rescale)

    io.generate_h5ad(counts=counts,
                     var=var,
                     obs=obs,
                     uns=uns,
                     save_path=outputfile,
                     threshold=threshold)
    assert outputfile.exists()
    got = sc.read(outputfile)
    got_counts = got.X
    if expect_sparse:
        got_counts = got_counts.toarray()
    np.testing.assert_equal(got_counts, expected_counts)
    pd.testing.assert_frame_equal(got.var, var)
    pd.testing.assert_frame_equal(got.obs, obs)
    assert got.uns == uns


@pytest.mark.parametrize("with_rescaling", (True, False))
def test_generate_h5ad_with_sampling(monkeypatch, anndata_file, tmp_path,
                                     with_rescaling):
    anndata_file = anndata_file(10)
    anndata_file = anndata_file[:, 0:100]
    anndata_file.obs.split = anndata_file.obs.split.astype("category")
    outputfile = tmp_path.joinpath('tmpfile.h5ad')
    counts = np.random.rand(10, 100).astype(np.float32)
    probas = np.random.rand(10, 100).astype(np.float32)
    outputvalue = np.random.randint(10, 100, 1)[0]
    expect_var = anndata_file.var.copy()
    expect_uns = anndata_file.uns.copy()
    if with_rescaling:
        anndata_file.uns["fixed_scaling"] = True

    monkeypatch.setattr(functions, "rescale_by_params",
                        lambda adata, *_: adata)

    def _check_sampling(sampling_counts, sampling_probas, var, uns):
        pd.testing.assert_frame_equal(var, expect_var)
        assert uns == expect_uns
        np.testing.assert_equal(sampling_counts, counts)
        np.testing.assert_equal(sampling_probas, probas)
        return np.full_like(sampling_counts, outputvalue)

    monkeypatch.setattr(functions, "sample_counts", _check_sampling)

    io.generate_h5ad(counts=(counts.copy(), probas.copy()),
                     var=anndata_file.var,
                     obs=anndata_file.obs,
                     uns=anndata_file.uns,
                     save_path=outputfile,
                     threshold=0)
    assert outputfile.exists()
    got = sc.read(outputfile)
    np.testing.assert_equal(got.X, outputvalue)
    np.testing.assert_equal(got.layers["estimated_counts"], counts)
    np.testing.assert_equal(got.layers["estimated_dropouts"], probas)


_TEST_CASES_MAKE_INPUT = [
    dict(trainfile='h5ad',
         validfile=None,
         batch_size=3,
         cachefile=None,
         exception=no_raise()),
    dict(trainfile='h5ad',
         validfile="h5ad",
         batch_size=3,
         cachefile=None,
         exception=no_raise()),
    dict(trainfile='h5ad',
         validfile=None,
         batch_size=5,
         cachefile=None,
         exception=no_raise()),
    dict(trainfile='h5ad',
         validfile=None,
         batch_size=5,
         cachefile="cachefile",
         exception=no_raise()),
    dict(trainfile='tf',
         validfile="tf",
         batch_size=3,
         cachefile=None,
         exception=no_raise()),
    dict(trainfile='tf',
         validfile="tf",
         batch_size=5,
         cachefile=None,
         exception=no_raise()),
    dict(trainfile='tf',
         validfile="tf",
         batch_size=5,
         cachefile="cachefile",
         exception=no_raise()),
    dict(trainfile='tf',
         validfile="tf",
         batch_size=5,
         cachefile="",
         exception=no_raise()),
    dict(trainfile='tf',
         validfile=None,
         batch_size=3,
         cachefile=None,
         exception=pytest.raises(
             ValueError,
             match="´validfile´ must be provided when using tfrecords")),
    dict(trainfile='unknown',
         validfile=None,
         batch_size=3,
         cachefile=None,
         exception=pytest.raises(NotImplementedError,
                                 match="Datatype of .* not understood")),
]


def test_serialize_numpy():
    """Check numpy serialization."""
    original_counts = np.random.rand(100, 10).astype(np.float32)
    original_labels = np.random.rand(100, 2).astype(np.float32)
    counts = original_counts.copy()
    labels = original_labels.copy()
    got = io._serialize_numpy(counts=counts, batch_no=labels)  # pylint: disable=protected-access
    got = np.frombuffer(got, dtype=np.float32).reshape(100, 13)
    np.testing.assert_allclose(got[:, 0:10], original_counts)
    np.testing.assert_allclose(got[:, 10:12], original_labels)
    np.testing.assert_allclose(got[:, 12], 1.0)


def check_tf_records(got: tf.data.Dataset,
                     expected_counts: np.ndarray,
                     expected_labels: np.ndarray,
                     is_serialized: bool = False):
    """Compare tf_records output with reference."""
    n_labels = expected_labels.max() - expected_labels.min() + 1
    n_genes = expected_counts.shape[1]
    dtype = tf.float32
    if is_serialized:
        total_length = (n_genes + n_labels) * dtype.size
        for i, element in enumerate(got):
            element = tf.io.decode_raw(tf.reshape(element, (-1, 1)),
                                       out_type=dtype,
                                       fixed_length=total_length)
            element = tf.reshape(element, (-1, )).numpy()
            assert element.shape == (n_labels + n_genes, )
            np.testing.assert_allclose(element[:-2], expected_counts[i])
            np.testing.assert_equal(element[-2:].argmax(), expected_labels[i])
    else:
        assert got.element_spec[0]["batch_input_dec"].shape == n_labels
        assert got.element_spec[0]["batch_input_dec"].dtype == dtype
        assert got.element_spec[0]["batch_input_enc"].shape == n_labels
        assert got.element_spec[0]["batch_input_enc"].dtype == dtype
        assert got.element_spec[0]["input_data"].shape == n_genes
        assert got.element_spec[0]["input_data"].dtype == dtype
        assert got.element_spec[1][0].shape == n_genes
        assert got.element_spec[1][0].dtype == dtype
        assert got.element_spec[1][1].shape == n_genes
        assert got.element_spec[1][1].dtype == dtype
        for i, element in enumerate(got):
            np.testing.assert_equal(
                element[0]["batch_input_enc"].numpy().argmax(),
                expected_labels[i])
            np.testing.assert_equal(
                element[0]["batch_input_dec"].numpy().argmax(),
                expected_labels[i])
            np.testing.assert_allclose(element[0]["input_data"],
                                       expected_counts[i])
            np.testing.assert_allclose(element[1][0], expected_counts[i])
            np.testing.assert_allclose(element[1][1], expected_counts[i])


@pytest.mark.parametrize('with_list', [False, True])
@pytest.mark.parametrize('n_labels', [1, 2, 5])
def test_parse_tfrecords(monkeypatch, tmp_path, with_list, n_labels):
    """Test parse_tfrecords."""
    monkeypatch.setattr(tf.data.experimental, "AUTOTUNE", 1)
    tfrecordsfile = tmp_path.joinpath('file.tfrecords_v2')
    genes_no = 10
    expected_counts = np.random.normal(scale=4,
                                       size=(100, genes_no)).astype(np.float32)
    original_labels = np.random.rand(100, n_labels).astype(np.float32)
    counts = expected_counts.copy()
    labels = original_labels.copy()
    serialized_data = io._serialize_numpy(counts=counts, batch_no=labels)  # pylint: disable=protected-access

    serialized_data = tf.data.Dataset.from_tensor_slices(serialized_data)
    tf.data.experimental.TFRecordWriter(
        str(tfrecordsfile), compression_type="GZIP").write(serialized_data)
    inputs = tfrecordsfile
    if with_list:
        inputs = [tfrecordsfile] * 2
    got = io.parse_tfrecords(tfr_files=inputs,
                             genes_no=genes_no,
                             n_labels=n_labels)
    if with_list:
        dtype = tf.float32
        assert got.element_spec[0]["batch_input_dec"].shape == n_labels
        assert got.element_spec[0]["batch_input_dec"].dtype == dtype
        assert got.element_spec[0]["batch_input_enc"].shape == n_labels
        assert got.element_spec[0]["batch_input_enc"].dtype == dtype
        assert got.element_spec[0]["input_data"].shape == genes_no
        assert got.element_spec[0]["input_data"].dtype == dtype
        assert got.element_spec[1][0].shape == genes_no
        assert got.element_spec[1][0].dtype == dtype
        assert got.element_spec[1][1].shape == genes_no
        assert got.element_spec[1][1].dtype == dtype
    else:
        check_tf_records(got, expected_counts, original_labels.argmax(axis=1),
                         False)


@pytest.mark.parametrize("min_label, max_label, size", [
    (0, 0, 100),
    (0, 1, 100),
    (0, 5, 100),
    (3, 5, 100),
    (0, 5, 200),
])
def test_np_one_hot(min_label, max_label, size):
    """Test numpy one hot encoding."""
    labels = np.arange(start=min_label, stop=max_label + 1, step=1)
    inputs = np.random.choice(labels, size=size, replace=True)
    inputs = pd.Categorical(inputs, categories=labels, ordered=True)
    got = io.np_one_hot(labels=inputs.copy())
    got = got.argmax(axis=1)
    np.testing.assert_equal(got, inputs.codes)


@pytest.mark.parametrize("for_tfrecords", [True, False])
def test_make_dataset_from_anndata(anndata_file, for_tfrecords):
    """Test creation of tf.data.Dataset from anndata object."""
    anndata_file = anndata_file(100)
    anndata_file.obs.batch = ['batch1'] * 50 + ['batch2'] * 50
    anndata_file.obs.batch = anndata_file.obs.batch.astype('category')
    anndata_file.obs["split"] = ['train', 'valid'] * 50
    got_train, got_valid = io.make_dataset_from_anndata(
        adata=anndata_file, for_tfrecord=for_tfrecords)
    expected_train = anndata_file[anndata_file.obs.split == "train"]
    expected_valid = anndata_file[anndata_file.obs.split == "valid"]
    check_tf_records(got_train, expected_train.X,
                     expected_train.obs.batch.cat.codes.values, for_tfrecords)
    check_tf_records(got_valid, expected_valid.X,
                     expected_valid.obs.batch.cat.codes.values, for_tfrecords)


class TestDISCERNData:
    """Testclass for DISCERNData."""
    # pylint: disable=no-self-use
    @pytest.mark.parametrize("batchsize", [10, 1, 100])
    @pytest.mark.parametrize("cachefile", ["", "somefile", None])
    def test_init(self, anndata_file, batchsize, cachefile):
        """Test initialization."""
        adata = anndata_file(150)
        adata.obs.split = "valid"
        got = io.DISCERNData(adata=adata,
                           batch_size=batchsize,
                           cachefile=cachefile)
        assert got.batch_size == batchsize
        assert got._tfdata == (None, None)  # pylint: disable=protected-access
        assert got.cachefile == cachefile
        tfdatas = got.tfdata
        assert isinstance(tfdatas[0], tf.data.Dataset)
        assert isinstance(tfdatas[1], tf.data.Dataset)

    @pytest.mark.parametrize(
        "expected_inputs", ["somefile", pathlib.Path("otherfile")])
    def test_read_h5ad(self, monkeypatch, expected_inputs, randomword):
        """Test h5ad reading."""
        def _check_input_read(inputs):
            assert inputs == expected_inputs
            return randomword

        def _check_init(_, X):
            # pylint: disable=invalid-name
            assert X == randomword

        monkeypatch.setattr(anndata, "read_h5ad", _check_input_read)
        monkeypatch.setattr(anndata.AnnData, "__init__", _check_init)

        got = io.DISCERNData.read_h5ad(expected_inputs, 10)
        assert got._batch_size == 10  # pylint: disable=protected-access

    @pytest.mark.parametrize("batchsize", [10, 20])
    @pytest.mark.parametrize("validcells", [5, 10, 30])
    def test_batch_size(self, batchsize, validcells, anndata_file):
        adata = anndata_file(150)
        adata.obs.split = "train"
        adata.obs.split.iloc[0:validcells] = "valid"
        assert adata.obs.split.value_counts()["valid"] == validcells
        got = io.DISCERNData(adata=adata, batch_size=batchsize)
        assert got._batch_size == batchsize  # pylint: disable=protected-access
        assert got.batch_size == min(batchsize, validcells)

    @pytest.mark.parametrize("tfrecords_exists", (True, False))
    def test_from_folder(self, monkeypatch, tmp_path, tfrecords_exists,
                         randomword):
        """Test from_folder generation."""
        batchsize = 10

        class _PatchDISCERNData:
            # pylint: disable=too-few-public-methods
            def __init__(self):
                self.tfdata = (None, None)
                self.var_names = np.zeros(10)
                self.obs = pd.DataFrame(
                    {"batch": pd.Categorical(["A", "B", "A"])})

        def _check_read(filename, batch_size, cachefile):
            assert cachefile == randomword + "cache"
            assert batch_size == batchsize
            assert filename == tmp_path.joinpath("processed_data",
                                                 "concatenated_data.h5ad")
            return _PatchDISCERNData()

        monkeypatch.setattr(io.DISCERNData, "read_h5ad", _check_read)

        outdir = None
        if tfrecords_exists:
            outdir = tmp_path.joinpath("TF_records")
            outdir.mkdir()
            outfile_train = outdir.joinpath("training.tfrecords_v2")
            outfile_valid = outdir.joinpath("validate.tfrecords_v2")
            outfile_train.touch()
            outfile_valid.touch()

        def _check_parse_tfrecords(tfr_files, genes_no, n_labels):
            assert tfr_files in (outfile_train, outfile_valid)
            assert genes_no == 10
            assert n_labels == 2
            return randomword

        monkeypatch.setattr(io, "parse_tfrecords", _check_parse_tfrecords)

        got = io.DISCERNData.from_folder(tmp_path, 10, randomword + "cache")

        if tfrecords_exists:
            assert got.tfdata == (randomword, randomword)
        else:
            assert got.tfdata == (None, None)
