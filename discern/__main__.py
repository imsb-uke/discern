#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Main script to process the data, start the training or generate cells from an existing model."""
import argparse
import bisect
import importlib
import json
import logging
import multiprocessing
import os
import pathlib
import shutil
import sys
from typing import Any, Callable, Dict, List

import click
import joblib
import scanpy as sc


def get_visible_devices_as_int() -> List[int]:
    """Return the number of CUDA visible devices as list of ints.

    Returns:
        List[int]: Number of available GPUs

    """
    def _get_visible_devices_as_int(conn):
        tfconfig = importlib.import_module("tensorflow").config
        gpus = [
            int(gpu.name.split(":")[-1])
            for gpu in tfconfig.get_visible_devices(device_type="GPU")
        ]

        conn.send(gpus)
        conn.close()

    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(target=_get_visible_devices_as_int,
                                      args=(child_conn, ))
    process.start()
    return parent_conn.recv()


_PARSERS = {
    "train":
    dict(description="Train the model."),
    "preprocess":
    dict(description=
         "Process the raw file and generate TF records for training.",
         args=[(('--force', ),
                dict(action='store_true',
                     help="Always override folder in preprocessing.")),
               (('--copy_original', ),
                dict(action='store_true',
                     help="Copy original data to the experiment folder.")),
               (('--without_tfrecords', ),
                dict(action='store_true',
                     help="Skip TFrecord files creation."))]),
    "optimize":
    dict(description="Find the best hyperparameter for the current dataset."),
    "project":
    dict(
        description="Project/Autoencode cell.",
        args=[
            (('--filename', ),
             dict(type=str,
                  required=False,
                  help="Alternative file to process. Must be preprocessed.")),
            (('--metadata', ),
             dict(required=False,
                  action="append",
                  type=str,
                  help='Project to metadata. Can be given multiple times.'
                  'Each should be in form of `metadata column:value`.'
                  '`value` can be empty, but the `:` has to be there.'
                  'If this option is not given cells are just autoencoded.')),
            (('--all-batches', ),
             dict(
                 action="store_true",
                 help=
                 'Project every cell to all batches in `batch` column of obs.')
             ),
            (('--store-sigmas', ),
             dict(required=False,
                  action="store_true",
                  help='Store sigmas as `X_DISCERN_sigma` in obsm.')),
        ]),
    "onlinetraining":
    dict(description="Continue training with new data set.",
         args=[
             (('--filename', ),
              dict(
                  type=str,
                  required=True,
                  help="New file to preprocess and to continue training with. "
                  "Cannot be preprocessed yet.")),
             (('--freeze', ),
              dict(required=False,
                   action="store_true",
                   help='Freeze non conditional layer.')),
         ]),
}


class CommandLineParser:
    """Commandline parser."""
    def __init__(self, **kwargs):
        kwargs.setdefault("prog", "discern")
        kwargs.setdefault("formatter_class",
                          argparse.ArgumentDefaultsHelpFormatter)

        self.parser = argparse.ArgumentParser(**kwargs)

        subparsers = self.parser.add_subparsers(help='sub-command help',
                                                dest='command')
        for parsername, definition in _PARSERS.items():
            self._create_discern_subparser(name=parsername,
                                           definition=definition,
                                           parser=subparsers)
        self.parser.add_argument('--debug',
                                 required=False,
                                 default=False,
                                 action='store_true',
                                 help='Used when debugging on local machine')

        self.parser.add_argument('--verbose',
                                 '-v',
                                 action='count',
                                 default=0,
                                 help="increase verbosity")
        self.parser.add_argument('--quiet',
                                 action='store_true',
                                 help="Disable warning messages")
        self.parser.add_argument('parameters',
                                 default=False,
                                 help='Path to the parameters json file')
        self.parameters = None
        self.experiments_dir = None
        self.exp_folders = None

    def parse_args(self, *args):
        """Parse command line arguments."""
        args = self.parser.parse_args(*args)
        parameters_file = pathlib.Path(args.parameters)
        with parameters_file.open('r') as file:
            self.parameters = json.load(file)
        if "exp_param" not in self.parameters:
            self.parameters = {
                "exp_param": {
                    "experiments_dir": parameters_file.parent.parent
                },
                "experiments": {
                    parameters_file.parent.name: self.parameters
                }
            }
        self.experiments_dir = pathlib.Path(
            self.parameters['exp_param']['experiments_dir'])
        self.experiments_dir.mkdir(exist_ok=True, parents=True)
        self._set_loglevels(verbosity=args.verbose, quiet=args.quiet)
        self.exp_folders: List[pathlib.Path] = [
            self.experiments_dir.joinpath(exp)
            for exp in self.parameters['experiments']
        ]
        getattr(self, args.command)(args)

    @staticmethod
    def _create_discern_subparser(
        name: str,
        definition: Dict[str, Any],
        parser: argparse._SubParsersAction  # pylint: disable=protected-access
    ):
        arguments = definition.pop("args", [])
        sub_parser = parser.add_parser(
            name=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            **definition)
        for args in arguments:
            sub_parser.add_argument(*args[0], **args[1])

    def _set_loglevels(self, verbosity: int, quiet: bool = False):
        loglevels = _LOGLEVEL(quiet=quiet)
        logformat = logging.Formatter(
            '%(asctime)s %(levelname)s:%(name)s: (pid=%(process)d) %(message)s'
        )
        rootlogger = logging.getLogger()

        filehandler = logging.FileHandler(
            self.experiments_dir.joinpath("discern.log"))
        filehandler.setFormatter(logformat)
        rootlogger.addHandler(filehandler)

        consolehandler = logging.StreamHandler()
        consolehandler.setFormatter(logformat)
        rootlogger.addHandler(consolehandler)
        rootlogger.setLevel(loglevels(verbosity))

        sc.settings.verbosity = verbosity
        logging.getLogger("numba").setLevel(loglevels(verbosity - 1))
        tflogger = logging.getLogger("tensorflow")
        tflogger.addHandler(filehandler)
        tflogger.setLevel(loglevels(verbosity - 1))
        tensorflow_cc_loglevel = str(max(0, 2 - verbosity))
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = tensorflow_cc_loglevel
        logging.debug('Tensorflow loglevel is: %s', tensorflow_cc_loglevel)

    def onlinetraining(self, args: argparse.Namespace):
        """Run online training."""
        gpus = get_visible_devices_as_int()
        mod = importlib.import_module("discern.online_learning")
        runner = mod.OnlineDISCERNRunner(  # type: ignore
            debug=args.debug, gpus=gpus)
        runner("train",
               self.exp_folders,
               filename=pathlib.Path(args.filename),
               freeze=args.freeze)

    def train(self, args: argparse.Namespace):
        """Run Training."""
        gpus = get_visible_devices_as_int()
        mod = importlib.import_module("discern.estimators")
        runner = mod.DISCERNRunner(debug=args.debug, gpus=gpus)  # type: ignore
        runner("train", self.exp_folders)

    def optimize(self, unused_args: argparse.Namespace):
        """Run Optiization."""
        ray_hyperpara = importlib.import_module("ray_hyperpara")  # type: Any
        if len(self.exp_folders) > 1:
            logging.warning("Found more than one experiments,"
                            "which is not supported, running the first.")
        ray_hyperpara.tune_hyperparameters(self.exp_folders[0])

    def project(self, args: argparse.Namespace):
        """Run Projection."""
        gpus = get_visible_devices_as_int()
        mod = importlib.import_module("discern.estimators")
        runner = mod.DISCERNRunner(debug=args.debug, gpus=gpus)  # type: ignore
        runner("project",
               self.exp_folders,
               metadata=args.metadata or [],
               infile=args.filename,
               store_sigmas=args.store_sigmas,
               all_batches=args.all_batches)

    def preprocess(self, args: argparse.Namespace):
        """Run Preprocessing."""
        # pylint: disable=too-many-locals
        exp_dirs = list()
        cpus = list()
        for exp, params in self.parameters['experiments'].items():
            raw_input = params['input_ds']['raw_input']
            exp_dir = self.experiments_dir.joinpath(exp)
            exp_dirs.append([exp_dir, not args.without_tfrecords])
            cpus.append(params['input_ds'].get('n_cpus_preprocessing', 30))
            # 2.1) shutil.copy the expression data set file to the job directory
            if exp_dir.joinpath('processed_data').exists():
                msg = 'Do you want to override the existing folder {} with new experiment?'.format(
                    exp_dir)
                if args.force or click.confirm(msg, default=True):
                    shutil.rmtree(exp_dir)
                else:
                    logging.fatal(
                        "The experiment folder `%s` already exists,"
                        "please remove it or select new one.", exp_dir)
                    sys.exit(1)
            exp_dir.mkdir(parents=True, exist_ok=True)
            if args.copy_original:
                exp_dir.joinpath('original_data').mkdir(parents=True,
                                                        exist_ok=True)
                for filename in raw_input:
                    filename = pathlib.Path(filename)
                    new_name = exp_dir.joinpath('original_data', filename.name)
                    if filename.is_dir():
                        shutil.copytree(filename, new_name)
                    else:
                        shutil.copy(filename, new_name)
            # create param.json file in each experiments directory
            hpsearch = params.pop("hyperparameter_search", None)
            with exp_dir.joinpath('parameters.json').open('w') as file:
                json.dump(params, file, sort_keys=True, indent=4)
            if hpsearch:
                shutil.copy(hpsearch,
                            exp_dir.joinpath('hyperparameter_search.json'))

            # apply pre-processing
        preprocessing = importlib.import_module(
            "discern.preprocessing")  # type: Any
        run_processes_with_cpu_requirements(
            preprocessing.read_process_serialize, exp_dirs, cpus)


class _LOGLEVEL:  # pylint: disable=too-few-public-methods
    """Loglevel helper class."""
    def __init__(self, quiet: bool = False):
        self._levels = ['WARNING', 'INFO', 'DEBUG']
        self._quiet = quiet

    def __call__(self, numeric_level: int):
        n_levels = len(self._levels)
        level = min(n_levels - 1, numeric_level)
        if self._quiet:
            level -= 1
        if level < 0:
            return "ERROR"
        return self._levels[level]


def run_processes_with_cpu_requirements(function: Callable[..., None],
                                        arguments: List[List[Any]],
                                        requirements: List[int]):
    """Run function in seperate processes and allow for different CPU requirements per process.

    Args:
        function (Callable[..., None]): Function to be executed.
        arguments (List[List[Any]]): List of all arguments in seperate lists.
        requirements (List[int]): List of CPUs to be used by each process.

    Raises:
        AssertionError: Requested number of CPUs is higher than available CPUs.

    """
    def _func(args, value, lock, finished, threads):
        try:
            function(*args)
        except Exception as excp:  # pylint: disable=broad-except
            logging.getLogger(__name__).exception('Exception %s: %s occurred',
                                                  type(excp).__name__, excp)
        finally:
            with lock:
                value.value += threads
            finished.set()

    sorted_req, arguments = zip(*sorted(zip(requirements, arguments)))
    sorted_req = list(sorted_req)
    arguments = list(arguments)
    n_cpus = joblib.cpu_count()

    assert_msg = "Maximum number of CPUs available is {}, but requested {}".format(
        n_cpus, sorted_req[-1])
    if not n_cpus >= sorted_req[-1]:
        raise ValueError(assert_msg)

    available_cpus = multiprocessing.Value('i', n_cpus)
    lock = multiprocessing.Lock()
    finished = multiprocessing.Event()
    processes: List[multiprocessing.Process] = list()
    while sorted_req:
        with lock:
            i = bisect.bisect_right(sorted_req, available_cpus.value) - 1
        if i < 0:
            finished.wait()
            finished.clear()
            continue
        threads = sorted_req.pop(i)
        args = arguments.pop(i)
        with lock:
            available_cpus.value -= threads
        processes.append(
            multiprocessing.Process(target=_func,
                                    args=(args, available_cpus, lock, finished,
                                          threads)))
        processes[-1].start()

    for proc in processes:
        proc.join()


def main():  # pragma: no cover
    """Main Function."""
    CommandLineParser().parse_args()


if __name__ == '__main__':  # pragma: no cover
    main()
