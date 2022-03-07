"""Test discern.preprocessing."""

import json
import pathlib
from contextlib import ExitStack as no_raise

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy

from discern import functions, io, preprocessing


def _create_csvfile(filepath, shape, sep=','):
    data = pd.Series(['somename'] * shape[0])
    output = pd.DataFrame()
    for i in range(shape[1]):
        output[i] = data
    output.to_csv(filepath, sep=sep, index=None, header=None)


@pytest.mark.parametrize("inputtype_exception", [
    (".h5", no_raise()),
    (".h5ad", no_raise()),
    (".loom", no_raise()),
    (".txt", no_raise()),
    ("dir", no_raise()),
    ("dir+bar", no_raise()),
    (".tar", pytest.raises(ValueError,
                           match="failed, the inferred file format")),
])
@pytest.mark.parametrize("sparse", [True, False])
def test_read_raw_input(tmp_path, monkeypatch, anndata_file,
                        inputtype_exception, sparse):
    """Test read_raw_input function."""
    inputtype, exception = inputtype_exception

    anndata_file = anndata_file(10)[:, 0:10].copy()
    if sparse:
        anndata_file.X = scipy.sparse.csr_matrix(anndata_file.X)

    monkeypatch.setattr(sc, "read_10x_h5",
                        lambda *_, **unused_kwargs: anndata_file)
    monkeypatch.setattr(sc, "read", lambda *_, **unused_kwargs: anndata_file)
    monkeypatch.setattr(sc, "read_text",
                        lambda *_, **unused_kwargs: anndata_file)

    inputs = tmp_path.joinpath("inputs")
    if inputtype.startswith('dir'):
        inputs.mkdir()
        _create_csvfile(inputs.joinpath('genes.tsv'), (10, 2), sep='\t')
        if inputtype == 'dir+bar':
            _create_csvfile(inputs.joinpath('barcodes.tsv'), (10, 1))
    else:
        inputs = inputs.with_suffix(inputtype)

    with exception:
        preprocessing.read_raw_input(inputs)


@pytest.mark.parametrize("n_batches", [1, 2, 10, 100])
@pytest.mark.parametrize("batchcol", ["batch", "other_batch_col"])
def test_merge_data_sets(anndata_file, n_batches, batchcol):
    """Test merge_data_sets function."""
    anndata_file = anndata_file(10)
    anndata_file.obs.drop(columns=['batch'], inplace=True, errors='ignore')
    batches_to_check = {'batch{}'.format(i) for i in range(n_batches)}
    inputs = {}
    for key in batches_to_check:
        tmp = anndata_file.copy()
        if batchcol != "batch":
            tmp.obs[batchcol] = key + batchcol
        inputs[key] = tmp
    outputdataset, mapping = preprocessing.merge_data_sets(
        inputs, {key: batchcol
                 for key in inputs})
    outputcounts = outputdataset.obs.batch.value_counts()
    if batchcol != "batch":
        batches_to_check = {k + batchcol for k in batches_to_check}
    assert set(outputcounts.index) == batches_to_check
    assert (outputcounts == 10).all()
    for i in range(n_batches):
        subset = outputdataset[outputdataset.obs.batch.cat.codes == i]
        np.testing.assert_allclose(subset.X, anndata_file.X)
    assert len(mapping) == n_batches
    assert set(mapping.values()) == batches_to_check
    assert set(mapping.keys()) == set(range(n_batches))
    assert "dataset" in outputdataset.obs.columns
    assert set(outputdataset.obs["dataset"].unique()) == set(inputs.keys())


_TEST_CASES_KERNEL_MMD = [
    (50, 10000, {
        "all": 1293.09
    }),
    (50, 2000, {
        "all": 1293.09
    }),
]

_TEST_CASES_SPLIT = [
    (False, 0, 0.1, None, (0.1, 0.1, no_raise())),
    (False, 10, 0.1, None, (0.1, 0.1, no_raise())),
    (False, 0, 20, None, (0.02, 0.02, no_raise())),
    (False, 10, 20, None, (0.02, 0.02, no_raise())),
    (False, 0, 0.1, 0.1, (0.1, 0.01, no_raise())),
    (False, 0, 0.1, 10, (0.1, 0.01, no_raise())),
    (False, 0, 20, 0.1, (0.02, 0.002, no_raise())),
    (False, 0, 20, 10, (0.02, 0.01, no_raise())),
    (False, 0, 1.1, 0.1, (0.2, 0.02,
                          pytest.raises(ValueError,
                                        match="Ratio of validation cells"))),
    (False, 0, 20, 1.1, (0.02, 0.01,
                         pytest.raises(ValueError,
                                       match="Ratio of MMD cells"))),
    (False, 0, 20, 30, (0.02, 0.01,
                        pytest.raises(ValueError,
                                      match="Ratio of MMD cells"))),
    (True, 0, 0.1, None, (0.1, 0.1, no_raise())),
    (True, 10, 0.1, None, (0.1, 0.1, no_raise())),
    (True, 0, 20, None, (0.02, 0.02, no_raise())),
    (True, 10, 20, None, (0.02, 0.02, no_raise())),
    (True, 0, 0.1, 0.1, (0.1, 0.01, no_raise())),
    (True, 0, 0.1, 10, (0.1, 0.01, no_raise())),
    (True, 0, 20, 0.1, (0.02, 0.002, no_raise())),
    (True, 0, 20, 10, (0.02, 0.01, no_raise())),
    (True, 0, 1.1, 0.1, (0.2, 0.02,
                         pytest.raises(ValueError,
                                       match="Ratio of validation cells"))),
    (True, 0, 20, 1.1, (0.02, 0.01,
                        pytest.raises(ValueError,
                                      match="Ratio of MMD cells"))),
    (True, 0, 20, 30, (0.02, 0.01,
                       pytest.raises(ValueError, match="Ratio of MMD cells"))),
]


@pytest.fixture(autouse=True)
def no_threading(monkeypatch):
    """Disable set_gpu_and_threads for all tests."""
    monkeypatch.setattr(functions, "set_gpu_and_threads",
                        lambda *unused_args, **unused_kwargs: None)


class TestWAERecipe:
    """Test WAERecipe."""

    # pylint: disable=too-few-public-methods
    def test_init(self, tmp_path, monkeypatch, anndata_file):
        """Test __init__ of WAERecipe."""
        # pylint: disable=no-self-use
        anndata_file = anndata_file(10)

        def _merge_data_sets(*unused_args, **_):
            return anndata_file, {0: 'batch1', 1: 'batch2'}

        monkeypatch.setattr(preprocessing, 'read_raw_input', lambda *_: None)
        monkeypatch.setattr(preprocessing, 'merge_data_sets', _merge_data_sets)
        input1 = tmp_path.joinpath('input1')
        input2 = tmp_path.joinpath('input2.h5ad')
        genematrix = preprocessing.WAERecipe(input_files=[input1, input2],
                                             params={},
                                             n_jobs=3)
        assert isinstance(genematrix.sc_raw, io.DISCERNData)
        np.testing.assert_equal(genematrix.sc_raw.X, anndata_file.X)
        pd.testing.assert_frame_equal(genematrix.sc_raw.obs, anndata_file.obs)
        pd.testing.assert_frame_equal(genematrix.sc_raw.var, anndata_file.var)
        assert genematrix._n_jobs == 3  # pylint: disable=protected-access
        assert sc.settings.n_jobs == 3
        assert genematrix.config['scale'] == {}
        assert genematrix.config['mmd_kernel'] == {}
        assert genematrix.config['batch_ratios'] == {}
        assert genematrix.config['valid_cells_no'] == {}
        assert genematrix.config['train_cells_no'] == {}
        assert set(genematrix.config['batch_key'].keys()) == set((0, 1))
        assert genematrix.config['batch_key'][0] == "batch1"
        assert genematrix.config['batch_key'][1] == "batch2"

    @pytest.mark.parametrize("filter_cells", [True, False])
    @pytest.mark.parametrize("filter_genes", [True, False])
    def test_filtering(self, monkeypatch, filter_cells, filter_genes):
        """Test WAERecipe.filtering."""
        data = np.ones((20, 20))
        if filter_cells:
            data[-3:, :] = 0.
        if filter_genes:
            data[:, -3:] = 0.
        adata = anndata.AnnData(data)

        def _init_genematrix(self, *unused_args, **unused_kwargs):
            self.sc_raw = adata
            sc.settings.n_jobs = 1

        monkeypatch.setattr(preprocessing.WAERecipe, "__init__",
                            _init_genematrix)
        genematrix = preprocessing.WAERecipe(None)
        genematrix.filtering(min_genes=2, min_cells=2)
        if filter_cells:
            assert genematrix.sc_raw.shape[0] == 17
        else:
            assert genematrix.sc_raw.shape[0] == 20

        if filter_genes:
            assert genematrix.sc_raw.shape[1] == 17
        else:
            assert genematrix.sc_raw.shape[1] == 20

    @pytest.mark.parametrize("scale", [0, 1, 2, 5, 10])
    @pytest.mark.flaky(reruns=2)
    def test_scaling(self, monkeypatch, scale):
        """Test WAERecipe.scaling."""
        data = np.random.rand(20, 20)
        expected_total = data.sum(axis=1)
        adata = anndata.AnnData(data.copy())

        def _init_genematrix(self, *unused_args, **unused_kwargs):
            self.sc_raw = io.DISCERNData(adata, -1)
            sc.settings.n_jobs = 1
            self.sc_raw.config = {"scale": {}}

        monkeypatch.setattr(preprocessing.WAERecipe, "__init__",
                            _init_genematrix)
        genematrix = preprocessing.WAERecipe(None)
        genematrix.scaling(scale)
        np.testing.assert_allclose(genematrix.sc_raw.obs.n_counts.values,
                                   expected_total,
                                   rtol=2e-7)
        assert genematrix.config['scale'] == {'LSN': scale}
        expected_scale = np.repeat((scale / expected_total)[:, np.newaxis],
                                   20,
                                   axis=1)
        np.testing.assert_allclose(genematrix.sc_raw.X / data,
                                   expected_scale,
                                   rtol=3e-7)

    @pytest.mark.parametrize("colname",
                             ["cluster_names", "cluster", "celltype", None])
    def test_celltypes(self, monkeypatch, colname):
        """Test WAERecipe.celltypes."""
        shape = 20
        data = np.zeros((shape, 20))
        adata = anndata.AnnData(data.copy())
        celltypes = {
            "celltype1": 1 / 4,
            "celltype2": 1 / 4,
            "celltype3": 1 / 2
        }
        if colname is not None:
            celltypcol = []
            for cell, freq in celltypes.items():
                celltypcol += [cell] * int(shape * freq)
            adata.obs[colname] = celltypcol

        def _init_genematrix(self, *unused_args, **unused_kwargs):
            self.sc_raw = io.DISCERNData(adata, -1)
            sc.settings.n_jobs = 1
            self.sc_raw.config = {}

        monkeypatch.setattr(preprocessing.WAERecipe, "__init__",
                            _init_genematrix)
        genematrix = preprocessing.WAERecipe(None)
        genematrix.celltypes()
        if colname is not None:
            assert "celltype" in genematrix.sc_raw.obs.columns
            assert genematrix.sc_raw.obs.celltype.tolist() == celltypcol
            if colname != "celltype":
                assert colname not in genematrix.sc_raw.obs.columns
        else:
            assert "celltype" not in genematrix.sc_raw.obs.columns

    @pytest.mark.parametrize("pcs", [*list(range(10, 50, 10)), None])
    def test_projection_pca(self, monkeypatch, pcs):
        """Test WAERecipe.projection_pca."""
        np.random.seed(0)
        data = np.random.rand(100, 2000)
        adata = anndata.AnnData(data.copy())

        def _init_genematrix(self, *unused_args, **unused_kwargs):
            self.sc_raw = adata
            sc.settings.n_jobs = 1

        monkeypatch.setattr(preprocessing.WAERecipe, "__init__",
                            _init_genematrix)
        genematrix = preprocessing.WAERecipe(None)
        if pcs:
            genematrix.projection_pca(pcs)
        else:
            genematrix.projection_pca()
        assert "PCs" in genematrix.sc_raw.varm
        assert genematrix.sc_raw.varm['PCs'].shape == (2000,
                                                       pcs if pcs else 25)
        assert "pca_genes" in genematrix.sc_raw.var.columns
        n_pca_genes = genematrix.sc_raw.var.pca_genes.value_counts()[True]
        assert n_pca_genes == 999
        assert "mean_scaling" in genematrix.sc_raw.var.columns
        np.testing.assert_allclose(genematrix.sc_raw.var.mean_scaling.values,
                                   np.log1p(data).mean(axis=0))
        assert "var_scaling" in genematrix.sc_raw.var.columns
        np.testing.assert_allclose(genematrix.sc_raw.var.var_scaling.values,
                                   np.log1p(data).std(axis=0))

    @pytest.mark.parametrize("neighbors_mmd, no_cells_mmd,expected",
                             _TEST_CASES_KERNEL_MMD)
    def test_kernel_mmd(self, monkeypatch, neighbors_mmd, no_cells_mmd,
                        expected):
        """Test WAERecipe.kernel_mmd."""
        np.random.seed(0)
        data = np.random.rand(100, 2000)
        adata = anndata.AnnData(data.copy())

        def _init_genematrix(self, *unused_args, **unused_kwargs):
            self.sc_raw = io.DISCERNData(adata, -1)
            sc.settings.n_jobs = 1
            self._n_jobs = 1
            self.sc_raw.config = {"mmd_kernel": {}, "celltype_ratios": {}}

        monkeypatch.setattr(preprocessing.WAERecipe, "__init__",
                            _init_genematrix)
        genematrix = preprocessing.WAERecipe(None)
        genematrix.projection_pca()
        genematrix.kernel_mmd(neighbors_mmd=neighbors_mmd,
                              no_cells_mmd=no_cells_mmd)
        assert set(genematrix.config['mmd_kernel'].keys()) == set(
            expected.keys())
        for key, mmd_val in genematrix.config['mmd_kernel'].items():
            np.testing.assert_allclose(mmd_val, expected[key], atol=0.1)

    @pytest.mark.parametrize(
        "with_celltype,split_seed, valid_cells_ratio, mmd_cells_ratio, expected",
        _TEST_CASES_SPLIT)
    def test_split(self, monkeypatch, anndata_file, with_celltype, split_seed,
                   valid_cells_ratio, mmd_cells_ratio, expected):
        """Test WAERecipe.split."""
        #pylint: disable=too-many-locals, too-many-arguments
        expected_valid_cells, expected_mmd_cells, expected_exception = expected
        shape = 1000
        args = dict(split_seed=split_seed, valid_cells_ratio=valid_cells_ratio)
        if mmd_cells_ratio is not None:
            args["mmd_cells_ratio"] = mmd_cells_ratio

        adata = anndata_file(shape)
        batches = adata.obs['batch'].copy()
        batch_cat = set(batches.cat.categories)
        if with_celltype:
            adata.obs['celltype'] = np.random.choice(
                ["celltype1", "celltype2", "celltype3"],
                size=shape,
                p=[0.1, 0.6, 0.3],
                replace=True)

        def _init_genematrix(self, *unused_args, **unused_kwargs):
            self.sc_raw = io.DISCERNData(adata, -1)
            sc.settings.n_jobs = 1
            self.sc_raw.config = {
                "valid_cells_no": {},
                "train_cells_no": {},
                "batch_ratios": {},
                "total_train_count": None,
                "total_valid_count": None,
                "common_genes_count": None,
                "total_count": None
            }

        monkeypatch.setattr(preprocessing.WAERecipe, "__init__",
                            _init_genematrix)
        genematrix = preprocessing.WAERecipe(None)
        with expected_exception as execinfo:
            genematrix.split(**args)
        if not isinstance(execinfo, no_raise):
            return

        assert genematrix.config["total_count"] == shape
        assert genematrix.config["common_genes_count"] == adata.X.shape[1]
        np.testing.assert_allclose(genematrix.config["total_valid_count"] /
                                   shape,
                                   expected_valid_cells,
                                   atol=0.001)
        np.testing.assert_allclose(genematrix.config["total_train_count"] /
                                   shape,
                                   1 - expected_valid_cells,
                                   atol=0.001)

        expected_ratios = batches.value_counts(normalize=True,
                                               sort=False).to_dict()
        assert genematrix.config["batch_ratios"] == expected_ratios

        assert set(genematrix.config["valid_cells_no"].keys()) == batch_cat
        got_counts = genematrix.sc_raw.obs.groupby(
            'split')['batch'].value_counts()
        for key, batchratio in genematrix.config["valid_cells_no"].items():
            expected_n = round(expected_ratios[key] * shape *
                               expected_valid_cells)
            assert expected_n - 1 <= batchratio <= expected_n + 1
            assert got_counts.loc['valid'].loc[key] == batchratio

        assert set(genematrix.config["train_cells_no"].keys()) == batch_cat
        for key, batchratio in genematrix.config["train_cells_no"].items():
            expected_n = round(expected_ratios[key] * shape *
                               (1 - expected_valid_cells))
            assert expected_n - 1 <= batchratio <= expected_n + 1
            assert got_counts.loc['train'].loc[key] == batchratio

        got_for_mmd = genematrix.sc_raw.obs['for_mmd'].value_counts()[True]
        np.testing.assert_allclose(got_for_mmd / shape,
                                   expected_mmd_cells,
                                   atol=0.01)

    def test_split_celltype_stratification(self, monkeypatch, anndata_file):
        """Test WAERecipe.split with celltype stratification for 'for_mmd'."""
        adata = anndata_file(1000)
        idx_firstbatch = adata.obs['batch'] == "pbmc_8k_new"
        idx_firstbatch_size = idx_firstbatch.value_counts(sort=False)[True]
        adata.obs.loc[idx_firstbatch, 'celltype'] = np.random.choice(
            ["celltype1", "celltype2", "celltype3"],
            size=idx_firstbatch_size,
            p=[0.1, 0.6, 0.3],
            replace=True)
        adata.obs.loc[~idx_firstbatch, 'celltype'] = np.random.choice(
            ["celltype1", "celltype2", "celltype3"],
            size=1000 - idx_firstbatch_size,
            p=[0.6, 0.2, 0.2],
            replace=True)

        def _init_genematrix(self, *unused_args, **unused_kwargs):
            self.sc_raw = io.DISCERNData(adata, -1)
            sc.settings.n_jobs = 1
            self.sc_raw.config = {
                "valid_cells_no": {},
                "train_cells_no": {},
                "batch_ratios": {},
                "total_train_count": None,
                "total_valid_count": None,
                "common_genes_count": None,
                "total_count": None
            }

        monkeypatch.setattr(preprocessing.WAERecipe, "__init__",
                            _init_genematrix)
        genematrix = preprocessing.WAERecipe(None)
        genematrix.split(split_seed=0,
                         valid_cells_ratio=0.99,
                         mmd_cells_ratio=1.)
        for_mmd_cells = genematrix.sc_raw.obs[genematrix.sc_raw.obs['for_mmd']]
        mmd_freq = for_mmd_cells.groupby('batch')['celltype'].value_counts(
            normalize=True, sort=False)
        expected_freq = adata.obs.groupby('batch')['celltype'].value_counts(
            normalize=True, sort=False)
        merged = pd.merge(mmd_freq,
                          expected_freq,
                          how='outer',
                          left_index=True,
                          right_index=True)
        np.testing.assert_allclose(merged['celltype_x'].values,
                                   merged['celltype_y'].values,
                                   atol=0.01)

    def test_split_nonunique_columns(self, monkeypatch, anndata_file):

        adata = anndata_file(30)
        adata.obs['celltype'] = np.random.choice(
            ["celltype1", "celltype2", "celltype3"],
            size=30,
            p=[0.1, 0.6, 0.3],
            replace=True)
        adata.obs["celltype3"] = adata.obs.celltype.copy()
        adata.obs.rename(columns={"celltype3": "celltype"}, inplace=True)

        def _init_genematrix(self, *unused_args, **unused_kwargs):
            self.sc_raw = io.DISCERNData(adata, -1)
            sc.settings.n_jobs = 1
            self.sc_raw.config = {
                "valid_cells_no": {},
                "train_cells_no": {},
                "batch_ratios": {},
                "total_train_count": None,
                "total_valid_count": None,
                "common_genes_count": None,
                "total_count": None
            }

        monkeypatch.setattr(preprocessing.WAERecipe, "__init__",
                            _init_genematrix)
        genematrix = preprocessing.WAERecipe(None)
        with pytest.raises(
                NotImplementedError,
                match=
                r" duplicated column names, currently \{'celltype'\} are duplicated"
        ):
            genematrix.split(split_seed=0,
                             valid_cells_ratio=0.99,
                             mmd_cells_ratio=1.)

    @pytest.mark.parametrize("with_fixed_scaling", [True, False])
    @pytest.mark.parametrize("mean", [10.0])
    @pytest.mark.parametrize("var", [2.0])
    def test_mean_var_scaling(self, anndata_file, parameters,
                              with_fixed_scaling, mean, var):
        """Test mean var scaling."""
        tmp_path = parameters.parent
        anndata_file = anndata_file(1000)
        anndata_file = anndata_file[:, :1000].copy()
        anndata_file.X = np.ones_like(anndata_file.X)
        anndatafilepath = tmp_path.joinpath('tmpfile.h5ad')
        anndata_file.var["mean_scaling"] = mean
        anndata_file.var["var_scaling"] = var
        anndata_file.write(str(anndatafilepath))
        with parameters.open('r') as file:
            hparam = json.load(file)

        hparam['input_ds']['n_cpus_prepreprocessing'] = 1
        hparam['input_ds']['raw_input'] = [str(anndatafilepath)]
        hparam['input_ds']['scale'].pop('fixed_scaling', None)
        if with_fixed_scaling:
            hparam['input_ds']['scale']['fixed_scaling'] = dict(var='genes',
                                                                mean='genes')
        with parameters.open('w') as file:
            json.dump(hparam, file)

        recipe = preprocessing.WAERecipe.from_path(tmp_path)
        recipe.mean_var_scaling()
        got_anndata = recipe.sc_raw

        expected = 1.0
        if with_fixed_scaling:
            expected = (1.0 - mean) / var
            assert got_anndata.uns['fixed_scaling'] == dict(var='genes',
                                                            mean='genes')
        np.testing.assert_allclose(got_anndata.X, expected)

    @pytest.mark.parametrize("with_lsn", [True, False])
    def test_call(self, monkeypatch, anndata_file, parameters, with_lsn):
        """Test _apply_wae_recipe."""
        tmp_path = parameters.parent
        anndata_file = anndata_file(1000)
        anndata_file = anndata_file[:, :1000].copy()
        anndatafilepath = tmp_path.joinpath('tmpfile.h5ad')
        exp_anndata = anndata_file.copy()
        np.expm1(anndata_file.X, out=anndata_file.X)
        anndata_file.write(str(anndatafilepath))
        with parameters.open('r') as file:
            hparam = json.load(file)

        hparam['input_ds']['n_cpus_prepreprocessing'] = 1
        hparam['input_ds']['raw_input'] = [str(anndatafilepath)]
        hparam['input_ds']['scale'].pop('LSN', None)
        if with_lsn:
            hparam['input_ds']['scale']['LSN'] = 1
        hparam['input_ds']['scale'].pop('fixed_scaling', None)
        with parameters.open('w') as file:
            json.dump(hparam, file)

        def _check_scaling(_, scale):
            assert scale == 1

        monkeypatch.setattr(preprocessing.WAERecipe, "scaling", _check_scaling)

        recipe = preprocessing.WAERecipe.from_path(tmp_path)
        got_anndata = recipe().sc_raw
        np.testing.assert_allclose(got_anndata.X.mean(axis=0),
                                   exp_anndata.X.mean(axis=0),
                                   atol=1e-5)
        np.testing.assert_allclose(got_anndata.X.std(axis=0),
                                   exp_anndata.X.std(axis=0),
                                   atol=1e-5)
        recipe.dump(job_dir=tmp_path)
        assert tmp_path.joinpath("processed_data",
                                 "concatenated_data.h5ad").exists()
        saved = sc.read(
            str(tmp_path.joinpath("processed_data", "concatenated_data.h5ad")))
        np.testing.assert_allclose(saved.X, got_anndata.X)
        assert (saved.obs == got_anndata.obs).all().all()
        assert saved.obsm == got_anndata.obsm
        assert (saved.var == got_anndata.var).all().all()
        assert set(saved.varm.keys()) == set(got_anndata.varm.keys())
        for key in saved.varm.keys():
            assert (saved.varm[key] == got_anndata.varm[key]).all()


@pytest.mark.parametrize('with_tfrecords', [True, False])
def test_read_process_serialize(monkeypatch, parameters, with_tfrecords):
    """Test read_process_serialize."""
    tmp_path = parameters.parent

    config_dict = {
        "valid_cells_no": {},
        "train_cells_no": {},
        "batch_ratios": {},
        "total_train_count": None,
        "total_valid_count": None,
        "common_genes_count": None,
        "total_count": None
    }

    def _init_genematrix(self, *unused_args, **unused_kwargs):
        self.sc_raw = None
        sc.settings.n_jobs = 1

    def _patch_call(obj):
        obj.sc_raw = "sampledata"
        return obj

    def _check_dump(obj, job_dir):
        assert obj.sc_raw == "sampledata"
        assert isinstance(job_dir, pathlib.Path)
        assert job_dir == tmp_path

    def _check_make_dataset_from_anndata(data, for_tfrecord):
        assert with_tfrecords
        assert for_tfrecord
        assert data == "sampledata"
        return ['train', 'valid']

    def _check_write_dataset(_, dataset, split):
        assert dataset == split

    monkeypatch.setattr(preprocessing.WAERecipe, "__init__", _init_genematrix)
    monkeypatch.setattr(preprocessing.WAERecipe, "config", config_dict)
    monkeypatch.setattr(preprocessing.WAERecipe, "__call__", _patch_call)
    monkeypatch.setattr(preprocessing.WAERecipe, "dump", _check_dump)
    monkeypatch.setattr(io, "make_dataset_from_anndata",
                        _check_make_dataset_from_anndata)
    monkeypatch.setattr(io.TFRecordsWriter, "write_dataset",
                        _check_write_dataset)
    preprocessing.read_process_serialize(tmp_path,
                                         with_tfrecords=with_tfrecords)
    if with_tfrecords:
        assert tmp_path.joinpath('TF_records').exists()
