.. image:: https://img.shields.io/pypi/l/discern-reconstruction
   :target: LICENSE
   :alt: License
.. image:: https://readthedocs.org/projects/discern/badge/?version=latest
   :target: https://discern.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/pypi/v/discern-reconstruction
   :target: https://pypi.org/project/discern-reconstruction/
   :alt: PyPI - Version
.. image:: https://img.shields.io/docker/v/fhausmann/discern/latest?label=Docker
   :target: https://hub.docker.com/repository/docker/fhausmann/discern
   :alt: Docker Image Version (tag latest semver)


DISCERN
=======

DISCERN is a deep learning approach to reconstruction expression of single-cell RNAseq data sets using a high-quality reference.
Please also look at our manuscript `DISCERN: deep single-cell expression reconstruction for improved cell clustering and cell subtype and state detection <https://doi.org/10.1186/s13059-023-03049-x>`_.

Getting Started
---------------

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
The full `documentation <https://discern.readthedocs.io/en/latest/>`_ contains further information and `tutorials <https://discern.readthedocs.io/en/latest/Tutorials.html>`_ to get started.

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

poetry can be used to install all further dependencies in a virtual environment.

.. code-block:: sh

   cd discern
   poetry install --no-dev

To finally run discern you can also directly use poetry with

.. code-block:: sh

   poetry run commands

or spawn a new shell in the virtual environment

.. code-block:: sh

   poetry shell

For all further examples, the first approach is used.

Using discern
^^^^^^^^^^^^^

You can use the main function of discern for most use cases. Usually, you have to preprocess your data by:

.. code-block:: sh

   poetry run discern process <parameters.json>

An example parameters.json is provided together with an hyperparameter_search.json for hyperparameter optimization using ray[tune].
The training can be done with

.. code-block:: sh

   poetry run discern train <parameters.json>

Hyperparameter optimization needs a ray server with can be started with

.. code-block:: sh

   poetry run ray start --head --port 57780 --redis-password='password'

and can be started with

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

Citation
--------


If you use `discern` in your work, please cite the publication as follows:

   **DISCERN: deep single-cell expression reconstruction for improved cell clustering and cell subtype and state detection**

   Fabian Hausmann, Can Ergen, Robin Khatri, Mohamed Marouf, Sonja Hänzelmann, Nicola Gagliani, Samuel Huber, Pierre Machart & Stefan Bonn

   *Genome Biology* 2023  doi: `10.1186/s13059-023-03049-x <https://doi.org/10.1186/s13059-023-03049-x>`_.
