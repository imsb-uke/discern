"""Test discern.estimators.run_exp module."""

import collections
import json
import multiprocessing as mp
import pathlib

import pytest
import pandas as pd
import tensorflow as tf

from discern import functions, io
from discern.estimators import batch_integration, run_exp
from discern.estimators import callbacks as customcallbacks
from discern import online_learning as online


@pytest.fixture
def disable_gpu_settings(monkeypatch):
    """Disable CPU and GPU settings."""
    monkeypatch.setattr(functions, 'set_gpu_and_threads', lambda *_: None)


@pytest.mark.parametrize('xla', [True, False])
def test_setup_exp(parameters, xla, disable_gpu_settings):
    """Test setup_exp."""
    # pylint: disable=unused-argument, redefined-outer-name
    with parameters.open('r') as file:
        expected_params = json.load(file)
    expected_params['training']['XLA'] = xla
    with parameters.open('w') as file:
        json.dump(expected_params, file)
    expfolder = parameters.parent

    model, hparams = run_exp.setup_exp(expfolder)
    assert isinstance(model, batch_integration.DISCERN)
    assert isinstance(hparams, dict)
    assert "training" in hparams
    assert "batch_size" in hparams['training']
    assert "max_steps" in hparams['training']
    assert "early_stopping" in hparams['training']
    assert "parallel_pc" in hparams['training']
    assert "GPU" in hparams['training']
    assert "XLA" in hparams['training']
    assert tf.config.optimizer.get_jit() == xla


def test_run_exp_multiprocess(parameters):
    """Test run_exp_multiprocess with valid input."""
    expfolder = parameters.parent
    available_gpus = list(range(3))
    exit_code = run_exp.run_exp_multiprocess(
        expfolder,
        available_gpus,
        func=lambda *_, **unused_kwargs: None,
        kwargs=None)
    assert exit_code == 0
    assert sorted(available_gpus) == list(range(3))

    exit_code = run_exp.run_exp_multiprocess(
        expfolder,
        available_gpus,
        func=lambda *_, **unused_kwargs: None,
        kwargs={'some_params': 5})
    assert exit_code == 0
    assert sorted(available_gpus) == list(range(3))


def test_run_exp_multiprocess_failing(parameters):
    """Test run_exp_multiprocess with invalid input."""
    def _failing_func(*unused_args, **unused_kwargs):
        raise AssertionError

    expfolder = parameters.parent
    available_gpus = list(range(3))
    exit_code = run_exp.run_exp_multiprocess(expfolder,
                                             available_gpus,
                                             func='notrain',
                                             kwargs=None)
    assert exit_code == 1
    exit_code = run_exp.run_exp_multiprocess(expfolder,
                                             available_gpus,
                                             func=_failing_func,
                                             kwargs=None)
    assert exit_code == 1

    with pytest.raises(IndexError):
        run_exp.run_exp_multiprocess(expfolder, [],
                                     func=lambda *_, **unused_kwargs: None,
                                     kwargs=None)
    with pytest.raises(FileNotFoundError):
        run_exp.run_exp_multiprocess(pathlib.Path(expfolder, 'invalid_path'),
                                     available_gpus,
                                     func=lambda *_, **unused_kwargs: None,
                                     kwargs=None)


@pytest.mark.parametrize('continue_run', [True, False])
def test_run_train(monkeypatch, parameters, disable_gpu_settings,
                   continue_run):
    """Test run_train function."""
    # pylint: disable=unused-argument, redefined-outer-name
    expfolder = parameters.parent
    with parameters.open('r') as file:
        expected_params = json.load(file)

    input_path = expfolder

    if continue_run:
        input_path = expfolder.joinpath('second_input')
        input_path.mkdir(exist_ok=True)

    def _check_from_folder(folder, batch_size):
        assert folder == input_path
        assert batch_size == expected_params['training']["batch_size"]
        return collections.namedtuple(
            "DATA", ["config", "var_names", "obs", "zeros"])(
                config={
                    "total_train_count": 100,
                    "total_valid_count": 10
                },
                zeros=0.0,
                var_names=pd.Series(range(10)),
                obs=pd.DataFrame({"batch": pd.Categorical([1, 2])}))

    def _check_callbacks(**kwargs):
        assert kwargs["early_stopping_limits"] == expected_params['training'][
            "early_stopping"]
        assert kwargs["exp_folder"] == expfolder.resolve()
        assert kwargs["umap_cells_no"] == 10
        assert kwargs["profile_batch"] == 0
        return "Callbacks"

    monkeypatch.setattr(io.DISCERNData, 'from_folder', _check_from_folder)
    monkeypatch.setattr(
        run_exp._LOGGER,  # pylint: disable=protected-access
        "getEffectiveLevel",
        lambda *_: 100)
    monkeypatch.setattr(customcallbacks, 'create_callbacks', _check_callbacks)

    def _test_func(_, savepath, inputdata, max_steps, callbacks=None):
        # pylint: disable=too-many-arguments
        assert type(inputdata).__name__ == "DATA"
        assert savepath == expfolder.joinpath("job", "best_model.hdf5")
        assert max_steps == expected_params['training']["max_steps"]
        assert callbacks == "Callbacks"

    monkeypatch.setattr(batch_integration.DISCERN, 'training', _test_func)

    if continue_run:
        expfolder.joinpath('job').mkdir(exist_ok=True)
    run_exp.run_train(expfolder,
                      input_path=input_path if continue_run else None)


@pytest.mark.parametrize('metadatainputs', [
    (['batch:pbmc_8k_new'], True),
    (['batch:pbmc_8k_new', 'batch:pbmc_cite_new'], True),
    (['batch:', 'batch:pbmc_8k_new'], True),
    (['batch:'], True),
    ([], True),
    (["batch:batch:value"], False),
    ([""], False),
])
@pytest.mark.parametrize('infile', [True, False])
@pytest.mark.parametrize('all_batches', [True, False])
def test_run_projection(monkeypatch, parameters, disable_gpu_settings,
                        anndata_file, metadatainputs, infile, all_batches):
    """Test run_projection function."""
    # pylint: disable=unused-argument, redefined-outer-name, too-many-arguments
    expfolder = parameters.parent
    metadatainputs, is_valid = metadatainputs
    processed_data = expfolder
    expfolder.joinpath("job").mkdir(exist_ok=True)
    expfolder.joinpath("job", "best_model.hdf5").touch()
    if not infile:
        processed_data = expfolder.joinpath("processed_data")
    filename = processed_data.joinpath("concatenated_data.h5ad")
    processed_data.mkdir(exist_ok=True)
    adata = anndata_file(10)
    adata.write(str(filename))
    monkeypatch.setattr(batch_integration.DISCERN, 'restore_model',
                        lambda *_: None)
    expected_len = len(metadatainputs)
    if all_batches:
        expected_len += adata.obs.batch.nunique()

    def _test_func(_, input_data, metadata, save_path, **kwargs):
        assert isinstance(input_data, io.DISCERNData)
        assert save_path == expfolder.joinpath("projected")
        assert save_path.is_dir()
        assert len(metadata) == expected_len
        for (col, val), total in zip(metadata, metadatainputs):
            assert f"{col}:{val}" == total
        if all_batches:
            for key, val in metadata[len(metadatainputs):]:
                assert key == "batch"
                assert val in set(adata.obs.batch.unique())
        assert kwargs == {
            "store_sigmas": True,
        }

    monkeypatch.setattr(batch_integration.DISCERN, 'project_to_metadata',
                        _test_func)
    if is_valid:
        run_exp.run_projection(expfolder,
                               metadata=metadatainputs,
                               infile=filename if infile else None,
                               store_sigmas=True,
                               all_batches=all_batches)
        return

    if sum(map(len, metadatainputs)) > 0:
        with pytest.raises(
                ValueError,
                match='Could not find value `.+´ in .+, only .+ available.'):
            run_exp.run_projection(expfolder,
                                   metadata=metadatainputs,
                                   infile=filename if infile else None,
                                   store_sigmas=True,
                                   all_batches=all_batches)
        return

    with pytest.raises(
            ValueError,
            match=r"not enough values to unpack \(expected 2, got 1\)"):
        run_exp.run_projection(expfolder,
                               metadata=metadatainputs,
                               all_batches=False,
                               store_sigmas=True,
                               infile=filename if infile else None)


_TESTCASES_RUNNER = [
    ("train", dict()),
    ("project",
     dict(metadata=[], infile=None, all_batches=True, store_sigmas=True)),
]


class TestDISCERNRunner:
    """Test DISCERNRunner."""
    @pytest.mark.parametrize("debug", (True, False))
    @pytest.mark.parametrize("gpu", ([0], []))
    def test_init(self, debug, gpu):
        """Test initialization."""
        runner = run_exp.DISCERNRunner(debug, gpu)
        assert runner.available_gpus == gpu
        assert runner.debug == debug
        assert runner._run_options == {  #pylint: disable=protected-access
            "train": run_exp.run_train,
            "project": run_exp.run_projection,
        }
        assert runner.available_gpus == ([0] if gpu else [])

    @pytest.mark.parametrize("mode", _TESTCASES_RUNNER)
    @pytest.mark.parametrize("gpu", ([], [1]))
    def test_run_debug(self, mode, gpu):
        """Test debug execution."""
        runner = run_exp.DISCERNRunner(True, gpu)
        folders = [pathlib.Path('testfolder1'), pathlib.Path('testfolder2')]

        def _check_train(exp_folder):
            assert mode[0] == "train"
            assert exp_folder == folders[0]

        def _check_project(exp_folder, metadata, infile, all_batches,
                           store_sigmas):
            assert mode[0] == "project"
            assert exp_folder == folders[0]
            assert all_batches == mode[1]["all_batches"]
            assert infile == mode[1]["infile"]
            assert store_sigmas == mode[1]["store_sigmas"]
            assert metadata == mode[1]["metadata"]

        runner._run_options = {  # pylint: disable=protected-access
            "train": _check_train,
            "project": _check_project
        }

        runner(mode[0], experiments=folders, **mode[1])

    @pytest.mark.parametrize("mode", _TESTCASES_RUNNER)
    def test_run_single_threaded(self, mode):
        """Test singleprocessed execution."""
        runner = run_exp.DISCERNRunner(False, [])
        folders = [pathlib.Path('testfolder1'), pathlib.Path('testfolder2')]

        def _check_train(exp_folder):
            assert mode[0] == "train"
            assert exp_folder in folders

        def _check_project(exp_folder, metadata, infile, all_batches,
                           store_sigmas):
            assert mode[0] == "project"
            assert exp_folder in folders
            assert all_batches == mode[1]["all_batches"]
            assert infile == mode[1]["infile"]
            assert store_sigmas == mode[1]["store_sigmas"]
            assert metadata == mode[1]["metadata"]

        runner._run_options = {  # pylint: disable=protected-access
            "train": _check_train,
            "project": _check_project
        }

        runner(mode[0], experiments=folders, **mode[1])

    @pytest.mark.parametrize("expected_mode", _TESTCASES_RUNNER)
    def test_run_multiprocessed(self, expected_mode, monkey_patch_pool,
                                monkeypatch):
        """Test multiprocessed execution."""
        # pylint: disable=redefined-outer-name
        monkey_patch_pool()
        runner = run_exp.DISCERNRunner(False, [1, 2])

        _mapping = dict(train="run_train", project="run_projection")

        def _check_run_multiprocessed(exp_folder, available_gpus, func,
                                      kwargs):
            assert exp_folder in folders
            assert isinstance(available_gpus, mp.managers.ListProxy)
            assert list(available_gpus) == [1, 2]
            assert func.__name__ == _mapping[expected_mode[0]]
            assert kwargs == expected_mode[1]

        folders = [pathlib.Path('testfolder1'), pathlib.Path('testfolder2')]
        monkeypatch.setattr(run_exp, "run_exp_multiprocess",
                            _check_run_multiprocessed)
        monkey_patch_pool()
        runner(expected_mode[0], experiments=folders, **expected_mode[1])

    @pytest.mark.parametrize("gpu", ([1], [1, 2], []))
    @pytest.mark.parametrize("debug", (True, False))
    def test_call(self, gpu, debug):
        """Test __call__."""
        runner = run_exp.DISCERNRunner(debug, gpu)

        def test_func(key):
            if debug:
                assert key == "debug"
                return
            if len(gpu) > 1:
                assert key == "multi"
                return
            assert key == "single"

        runner._run_multiprocessed = lambda **_: test_func("multi")  # pylint: disable=protected-access
        runner._run_single_threaded = lambda **_: test_func("single")  # pylint: disable=protected-access
        runner._run_debug = lambda **_: test_func("debug")  # pylint: disable=protected-access
        runner("noe", ["some"])


@pytest.fixture
def monkey_patch_pool(monkeypatch):
    """Monkey patch the multiprocessing.Pool class."""
    def _func():
        class ResultsPatch:  #pylint:disable=too-few-public-methods
            """Patch multiprocessing.pool.AsyncResult."""

            #
            @staticmethod
            def get():
                """Return results."""
                return

        class PoolPatch:
            """Patch multiprocessing.Pool."""

            #
            def __init__(self, _, *unused_args, **unused_kwargs):
                return

            def __enter__(self):
                return self

            def __exit__(self, unused_param1, unused_param2, unused_param3):
                return

            @staticmethod
            def starmap_async(func, args, **unused_kwargs):
                """Simulate Pool.starmap_async."""
                for arg in args:
                    func(*arg)
                return ResultsPatch()

        monkeypatch.setattr(mp, "Pool", PoolPatch)

    return _func


class TestOnlineDISCERNRunner(TestDISCERNRunner):
    """ Test running DISCERN in an online fashion."""
    @pytest.mark.parametrize("debug", (True, False))
    @pytest.mark.parametrize("gpu", ([0], []))
    def test_init(self, debug, gpu):
        """Test initialization."""
        runner = online.OnlineDISCERNRunner(debug, gpu)
        assert runner.available_gpus == gpu
        assert runner.debug == debug
        assert runner._run_options == {  #pylint: disable=protected-access
            "train": online.online_training,
        }
        assert runner.available_gpus == ([0] if gpu else [])
