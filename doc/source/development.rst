Contributing
============

Further contributions to DISCERN are very welcome.
To provide a consistent coding style and ensure a running program we implemented tests and code style guidelines.

Testing
-------

For critical parts of the model several tests has been implemented. They can be run with:

.. code-block:: sh

   poetry run pytest --cov=discern --cov-report=term

(Requires the development version of discern).

Some tests are slow and don't run by default, but you can run them using:

.. code-block:: sh

   poetry run pytest --runslow --cov=discern --cov-report=term

Coding style
------------

To enforce code style guidelines `pylint <https://www.pylint.org/>`_ and `mypy <http://mypy-lang.org/>`_ are use. Example commands are shown below:

.. code-block:: sh

   poetry run pylint discern ray_hyperpara.py
   poetry run mypy discern ray_hyperpara.py

For automatic code formatting `yapf <https://github.com/google/yapf>`_ was used:

.. code-block:: sh

   yapf -i <filename.py>

These tools are included in the dev-dependencies.
