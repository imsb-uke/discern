

.. image:: https://github.com/imsb-uke/discern/actions/workflows/test.yml/badge.svg
   :target: https://github.com/imsb-uke/discern/actions/workflows/test.yml
   :alt: pipeline status

.. image:: https://readthedocs.org/projects/discern/badge/?version=latest
   :target: https://discern.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

DISCERN
=======

DISCERN is a deep learning approach to reconstruction expression information
of single-cell RNAseq data sets using a high quality reference.
Please also have a look into our preprint `DiSCERN - Deep Single Cell Expression ReconstructioN for improved cell clustering and cell subtype and state detection <https://doi.org/10.1101/2022.03.09.483600>`_.

Getting Started
---------------

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
The the full `documentation <https://discern.readthedocs.io/en/latest/>`_ contains further information and `tutorials <https://discern.readthedocs.io/en/latest/Tutorials.html>`_ to get started.

Prerequisites
^^^^^^^^^^^^^

We use `poetry <https://python-poetry.org/>`_ for dependency management. You can get poetry by

.. code-block:: sh

   pip install poetry

or (the officially recommended way)

.. code-block:: sh

   curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

Installing
^^^^^^^^^^

To get discern you can clone the repository by

.. code-block:: sh

   git clone https://github.com/imsb-uke/discern.git

poetry can be used to install all further dependencies in an virtual environment.

.. code-block:: sh

   cd discern
   poetry install --no-dev

To finally run discern you can also directly use poetry with

.. code-block:: sh

   poetry run commands

or spawn a new shell in the virtual environment

.. code-block:: sh

   poetry shell

For further examples the first approach is presented.

Using discern
^^^^^^^^^^^^^

You can use the main function of discern for most use cases. Usually you have to preprocess your data by:

.. code-block:: sh

   poetry run discern process <parameters.json>

An example parameters.json is provided together with an hyperparameter_search.json for hyperparameter optimization using ray[tune].
The training can be done with

.. code-block:: sh

   poetry run discern train <parameters.json>

Hyperparameter optimization needs a ray server with can be started with

.. code-block:: sh

   poetry run ray start --head --port 57780 --redis-password='password'

and can started with

.. code-block:: sh

   poetry run discern optimize <parameters.json>

For projection 2 different modes are available:
Eval mode, which is a more general approach and can save a lot of files:

.. code-block:: sh

   poetry run discern project --all_batches <parameters.json>

Or projection mode which offers a more fine grained controll to which is projected.

.. code-block:: sh

   poetry run discern project --metadata="metadatacolumn:value" --metadata="metadatacolumn:" <parameters.json>

which creates to files, one is projected to the average batch calculated by a
``metadatacolumn`` and a contained ``value``.
The second file is projected to the the average for each value in "metadatacolumn"; individually.

DISCERN also supports online training. You can add new batches to your dataset after the usual ``train`` with:

.. code-block:: sh

   poetry run discern onlinetraining --freeze --filename=<new_not_preprocessed_batch[es].h5ad> <parameters.json>

The data gets automatically preprocessed and added to the dataset. You can run ``project`` afterwards as usual (without the ``--filename`` flag).
``--freeze`` is important to freeze non-conditional layers in training.

Authors
-------

* Can Ergen
* Pierre Machart
* Fabian Hausmann
