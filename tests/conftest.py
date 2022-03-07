"""Custom fixtures."""
import json
import string

import anndata
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from discern.estimators import batch_integration


@pytest.fixture
def randomword():
    "Pytest fixture to create a random word"
    letters = list(string.ascii_lowercase)
    return ''.join(np.random.choice(letters, size=100, replace=True))


@pytest.fixture
def anndata_file():
    """Pytest fixture for creation of anndata files."""
    def _create_file(nvals):
        size = 15289 * nvals
        vals = np.zeros(size, dtype=np.float32)
        non_zero = size - int(size * 0.92)
        non_zero = int(np.random.normal(loc=non_zero, scale=10, size=1))
        rand = np.random.normal(loc=1.38, scale=0.889, size=non_zero)
        rand = np.abs(rand) + 1
        idx = np.random.choice(np.arange(0, size),
                               size=non_zero,
                               replace=False)
        vals[idx] = rand
        vals = vals.reshape(-1, 15289)
        labels = {
            "pbmc_8k_new": 0.5382136602451839,
            "pbmc_cite_new": 0.4617863397548161
        }
        batch = np.random.choice(list(labels.keys()),
                                 p=list(labels.values()),
                                 replace=True,
                                 size=nvals)
        batch = pd.Categorical(batch)
        processed = anndata.AnnData(vals)
        processed.obs["batch"] = batch
        processed.obs["split"] = np.random.choice(["train", "valid"],
                                                  p=[0.6, 0.4],
                                                  replace=True,
                                                  size=nvals)

        return processed

    return _create_file


@pytest.fixture
def parameters(tmp_path):
    """Create a dummy parameter file."""
    parameter = {
        "input_ds": {
            "clustering": {
                "neighbors": 10,
                "res": 0.15
            },
            "filtering": {
                "min_cells": 3,
                "min_genes": 10
            },
            "n_cpus_preprocessing": 6,
            "raw_input": None,
            "scale": {
                "LSN": 20000,
                "fixed_scaling": {
                    "mean": "genes",
                    "var": "genes"
                }
            },
            "split": {
                "balanced_split": True,
                "split_seed": 0,
                "test_cells": 0,
                "valid_cells": 100
            }
        },
        "model": {
            "activation_fn": "tensorflow.keras.activations.relu",
            "encoder": {
                "layers": [128, 64, 32],
                "norm_type": [
                    "conditionallayernorm", "conditionallayernorm",
                    "conditionallayernorm"
                ],
                "regularization":
                0.15
            },
            "decoder": {
                "layers": [32, 64, 128],
                "norm_type": [
                    "conditionallayernorm", "conditionallayernorm",
                    "conditionallayernorm"
                ],
                "regularization":
                0.18
            },
            "latent_dim": 16,
            "output_fn": None,
            "reconstruction_loss": {
                "name": "HuberLoss",
                "delta": 11.0,
            },
            "wae_lambda": 11.32395861201524,
            "weighting_random_encoder": 4.407237266397266e-05
        },
        "training": {
            "GPU": [0],
            "XLA": True,
            "batch_size": 128,
            "early_stopping": {
                "patience": 20,
                "min_delta": 0.01,
                'mode': 'auto',
                'monitor': 'val_loss',
                'restore_best_weights': True
            },
            "max_steps": 100,
            "optimizer": {
                "algorithm": "tensorflow_addons.optimizers.RectifiedAdam",
                "amsgrad": False,
                "beta_1": 0.85,
                "beta_2": 0.95,
                "learning_rate": 0.0009136083007127199
            },
            "parallel_pc": 5
        }
    }
    parameters_file = tmp_path.joinpath('parameters.json')
    with parameters_file.open('w') as file:
        json.dump(parameter, file)
    return parameters_file


@pytest.fixture
def default_model():
    """Fixture for creating default model."""
    tf.keras.backend.clear_session()
    config = {
        "latent_dim": 32,
        "decoder": {
            "layers": [64, 128, 256],
            "norm_type": [
                "conditionallayernorm", "conditionallayernorm",
                "conditionallayernorm"
            ],
            'regularization':
            0.1
        },
        "encoder": {
            "layers": [256, 128, 64],
            "norm_type": [
                "conditionallayernorm", "conditionallayernorm",
                "conditionallayernorm"
            ],
            'regularization':
            0.1
        },
        "output_lsn": 20000,
        "activation_fn": "tensorflow.keras.activations.relu",
        "output_fn": "tensorflow.keras.activations.softplus",
        "recon_loss_type": dict(name='Lnorm', p=2, use_root=False, axis=-1),
        "wae_lambda": 11.32395861201524,
        "weighting_random_encoder": 4.407237266397266e-05,
        "optimizer": {
            "algorithm": "tensorflow.keras.optimizers.Adam",
            "amsgrad": True,
            "beta_1": 0.85,
            "beta_2": 0.95,
            "learning_rate": 0.0009136083007127199
        },
    }
    return batch_integration.DISCERN(**config)


def pytest_addoption(parser):
    parser.addoption("--runslow",
                     action="store_true",
                     default=False,
                     help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
