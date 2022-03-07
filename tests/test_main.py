"""Test main functions."""
import argparse
import importlib
import json
import logging
import os
import pathlib
import time

import click
import joblib
import pytest
import scanpy as sc
import tensorflow

import ray_hyperpara
import discern.estimators
import discern.online_learning
from discern import __main__ as main  # pylint: disable=no-name-in-module

# pylint: disable=no-self-use

_COMMANDLINE_TEST = [
    ("preprocess {testfile}",
     dict(command="preprocess",
          force=False,
          copy_original=False,
          without_tfrecords=False)),
    ("train {testfile}", dict(command="train")),
    ("optimize {testfile}", dict(command="optimize")),
    ("project --metadata=column:key --metadata=column: {testfile}",
     dict(metadata=["column:key", "column:"],
          command="project",
          filename=None,
          all_batches=False,
          store_sigmas=False)),
    ("preprocess {testfile} -vvvv",
     dict(verbose=4,
          command="preprocess",
          force=False,
          copy_original=False,
          without_tfrecords=False)),
    ("preprocess {testfile} --verbose --verbose",
     dict(verbose=2,
          command="preprocess",
          force=False,
          copy_original=False,
          without_tfrecords=False)),
    ("preprocess --force --copy_original --without_tfrecords {testfile}",
     dict(force=True,
          copy_original=True,
          without_tfrecords=True,
          command="preprocess")),
    ("onlinetraining --freeze --filename tmp.file {testfile}",
     dict(freeze=True, filename="tmp.file", command="onlinetraining")),
]

_DEFAULT_ARGS = dict(debug=False, verbose=0, quiet=False)


@pytest.mark.forked
class TestCommandLineParser:
    """Test command line parser."""
    def test_init(self):
        """Test init and default parsers."""
        expected = main._PARSERS.keys()  # pylint: disable=protected-access
        parser = main.CommandLineParser()
        assert parser.exp_folders is None
        assert parser.experiments_dir is None
        assert parser.parameters is None
        assert isinstance(parser.parser, argparse.ArgumentParser)
        parser = parser.parser
        positionals = parser._positionals._group_actions  # pylint: disable=protected-access
        assert positionals[0].dest == "command"
        assert positionals[0].choices.keys() == expected
        assert positionals[1].dest == "parameters"
        optionals = parser._optionals._group_actions  # pylint: disable=protected-access
        assert {c.dest
                for c in optionals} == {"help", "verbose", "debug", "quiet"}

    @pytest.mark.parametrize("commandline, expected", _COMMANDLINE_TEST)
    def test_parse_args(self, monkeypatch, commandline, tmp_path, expected):
        """test Parsing."""
        for k, val in _DEFAULT_ARGS.items():
            expected.setdefault(k, val)
        parser = main.CommandLineParser()
        testfile = tmp_path.joinpath("testfile.json")
        with testfile.open("w") as file:
            json.dump(
                {
                    'exp_param': {
                        'experiments_dir': str(tmp_path)
                    },
                    "experiments": {
                        "test_exp": None
                    }
                }, file)
        expected["parameters"] = str(testfile)
        commandline = commandline.format(testfile=testfile)  #.split()
        monkeypatch.setattr(parser, "_set_loglevels", lambda **_: None)

        def _func(args):
            assert args.__dict__ == expected

        for k in main._PARSERS.keys():  # pylint: disable=protected-access
            monkeypatch.setattr(parser, k, _func)
        parser.parse_args(commandline.split())

    @pytest.mark.parametrize("expect_debug", (True, False))
    def test_train(self, monkeypatch, expect_debug):
        """Test Training."""
        parser = main.CommandLineParser()
        parser.exp_folders = ["testfolder"]

        class _MOCKRUNNER:
            def __init__(self, debug, gpus):
                assert debug == expect_debug
                assert gpus == "NO_GPU_HERE"

            def __call__(self, mode, folders, **kwargs):
                assert mode == "train"
                assert folders == ["testfolder"]
                assert kwargs == {}

        monkeypatch.setattr(main, "get_visible_devices_as_int",
                            lambda: "NO_GPU_HERE")
        monkeypatch.setattr(discern.estimators, "DISCERNRunner", _MOCKRUNNER)
        parser.train(argparse.Namespace(debug=expect_debug))

    @pytest.mark.parametrize("expect_debug", (True, False))
    @pytest.mark.parametrize("expect_freeze", (True, False))
    def test_onlinetraining(self, monkeypatch, expect_debug, expect_freeze):
        """Test Training."""
        parser = main.CommandLineParser()
        parser.exp_folders = ["testfolder"]

        class _MOCKRUNNER:
            def __init__(self, debug, gpus):
                assert debug == expect_debug
                assert gpus == "NO_GPU_HERE"

            def __call__(self, mode, folders, **kwargs):
                assert mode == "train"
                assert folders == ["testfolder"]
                assert kwargs == {
                    "freeze": expect_freeze,
                    "filename": pathlib.Path("tmp.file")
                }

        monkeypatch.setattr(main, "get_visible_devices_as_int",
                            lambda: "NO_GPU_HERE")
        monkeypatch.setattr(discern.online_learning, "OnlineDISCERNRunner",
                            _MOCKRUNNER)
        parser.onlinetraining(
            argparse.Namespace(debug=expect_debug,
                               freeze=expect_freeze,
                               filename="tmp.file"))

    @pytest.mark.parametrize("expect_debug", (True, False))
    def test_project(self, monkeypatch, expect_debug):
        """Test projection."""
        parser = main.CommandLineParser()
        parser.exp_folders = ["testfolder"]

        class _MOCKRUNNER:
            def __init__(self, debug, gpus):
                assert debug == expect_debug
                assert gpus == "NO_GPU_HERE"

            def __call__(self, mode, folders, **kwargs):
                assert mode == "project"
                assert folders == ["testfolder"]
                assert kwargs == {
                    "infile": "somefile",
                    "metadata": ["somemeta"],
                    "all_batches": False,
                    "store_sigmas": False
                }

        monkeypatch.setattr(main, "get_visible_devices_as_int",
                            lambda: "NO_GPU_HERE")
        monkeypatch.setattr(discern.estimators, "DISCERNRunner", _MOCKRUNNER)
        parser.project(
            argparse.Namespace(debug=expect_debug,
                               filename="somefile",
                               store_sigmas=False,
                               all_batches=False,
                               metadata=["somemeta"]))

    def test_optimize(self, tmp_path, monkeypatch):
        """Test optimization."""
        parser = main.CommandLineParser()
        parser.exp_folders = [tmp_path]

        def tune_hyperparameters(filename):
            assert isinstance(filename, pathlib.Path)
            assert filename == tmp_path

        monkeypatch.setattr(ray_hyperpara, 'tune_hyperparameters',
                            tune_hyperparameters)
        parser.optimize(None)

        parser.exp_folders.append("wrong_file")
        parser.optimize(None)


@pytest.mark.parametrize('copy_original', [True, False])
def test_preprocessing(tmp_path, monkeypatch, copy_original):
    """Test main part of preprocessing pipeline."""
    monkeypatch.setattr(click, 'confirm', lambda *args, **kwargs: True)
    monkeypatch.setattr(main, 'run_processes_with_cpu_requirements',
                        lambda *args: True)

    class PreprocessingModule:  # pylint: disable=too-few-public-methods
        """Mock class for discern.preprocessing."""

        #
        @classmethod
        def read_process_serialize(cls):
            """Mock function for discern.preprocessing.read_process_serialize."""
            return True

    class _MockCommandLineParser(main.CommandLineParser):
        def __init__(self, experiments_dir, experiments):
            self.experiments_dir = experiments_dir
            self.parameters = dict(experiments=experiments)

    importlib_func = importlib.import_module

    def _import(module):
        if module == "discern.preprocessing":
            return PreprocessingModule
        return importlib_func(module)

    monkeypatch.setattr(importlib, "import_module", _import, raising=False)

    hparams = tmp_path.joinpath('hparams.json')
    with hparams.open('w') as file:
        json.dump({}, file)
    testdata1 = tmp_path.joinpath('testdata1.txt')
    with testdata1.open('w') as file:
        file.write('dummy data')
    testdata2 = tmp_path.joinpath('testdata2', "testdata2.txt")
    testdata2.parent.mkdir(parents=True, exist_ok=True)
    with testdata2.open('w') as file:
        file.write('dummy data')
    test_full_dir = tmp_path.joinpath('test3', 'processed_data')
    test_full_dir.mkdir(parents=True, exist_ok=True)
    testdata3 = tmp_path.joinpath('testdata3.txt')
    with testdata3.open('w') as file:
        file.write('dummy data')
    experiments = dict(test1={
        "hyperparameter_search": str(hparams),
        'input_ds': {
            'raw_input': [str(testdata1)],
            "n_cpus_preprocessing": 3
        }
    },
                       test2={
                           "hyperparameter_search": str(hparams),
                           'input_ds': {
                               'raw_input': [str(testdata2.parent)],
                               "n_cpus_preprocessing": 3
                           }
                       },
                       test3={
                           'input_ds': {
                               'raw_input': [str(testdata3)],
                               "n_cpus_preprocessing": 3
                           }
                       })

    parser = _MockCommandLineParser(tmp_path, experiments)
    parser.preprocess(
        argparse.Namespace(**dict(copy_original=copy_original,
                                  without_tfrecords=True,
                                  override=False,
                                  force=False)))

    monkeypatch.setattr(click, 'confirm', lambda *args, **kwargs: False)
    test_full_dir.mkdir(parents=True, exist_ok=True)
    parser = _MockCommandLineParser(tmp_path, dict(test3=experiments['test3']))
    with pytest.raises(SystemExit):
        parser.preprocess(
            argparse.Namespace(**dict(copy_original=copy_original,
                                      without_tfrecords=True,
                                      override=False,
                                      force=False)))


def test_run_processes_with_cpu_requirements(monkeypatch):
    """Test run_processes_with_cpu_requirements function."""
    def _func(value):
        time.sleep(0.1)
        return value * 2

    arguments = [[1], [2], [3], [4]]
    requirements = [1, 2, 1, 2]
    monkeypatch.setattr(joblib, 'cpu_count', lambda: 2)
    main.run_processes_with_cpu_requirements(_func,
                                             arguments=arguments,
                                             requirements=requirements)

    with pytest.raises(ValueError):
        main.run_processes_with_cpu_requirements(_func,
                                                 arguments=arguments[:1],
                                                 requirements=[4])

    def _func(_):
        raise RuntimeError

    main.run_processes_with_cpu_requirements(_func,
                                             arguments=arguments[:1],
                                             requirements=[2])


@pytest.mark.parametrize("verbosity", [0, 1, 2])
@pytest.mark.parametrize("quiet", (True, False))
def test_set_loglevels(verbosity, tmp_path, quiet):
    """Test _set_loglevels function."""
    available_levels = [40, 30, 20, 10]

    if quiet:
        available_levels = [
            x + 10 if x != 40 else 40 for x in available_levels
        ]

    class _MockCommandLineParser(main.CommandLineParser):
        def __init__(self):
            self.experiments_dir = tmp_path

    _MockCommandLineParser()._set_loglevels(verbosity, quiet=quiet)  # pylint: disable=protected-access
    assert sc.settings.verbosity == verbosity
    got = logging.getLogger("numba").getEffectiveLevel()
    assert got == available_levels[verbosity]
    got = logging.getLogger("tensorflow").getEffectiveLevel()
    assert got == available_levels[verbosity]
    tensorflow_cc_loglevel = [2, 1, 0, 0]
    got = os.environ['TF_CPP_MIN_LOG_LEVEL']
    assert int(got) == tensorflow_cc_loglevel[verbosity]


@pytest.mark.parametrize("n_gpu", [0, 1, 2])
def test_get_visible_devices_as_int(monkeypatch, n_gpu):
    """Test get_visible_devices_as_int."""
    class GpuPatch:  # pylint: disable=too-few-public-methods
        """Patch GPU results from tensorflow.config.get_visible_devices."""

        #
        def __init__(self, n):
            self.name = f"GPU:{n}"

    def get_visible_devices(device_type):  # pylint: disable=unused-argument
        return [GpuPatch(i) for i in range(n_gpu)]

    monkeypatch.setattr(tensorflow.config, "get_visible_devices",
                        get_visible_devices)
    got = main.get_visible_devices_as_int()
    assert sorted(got) == list(range(n_gpu))
