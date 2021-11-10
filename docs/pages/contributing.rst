Contributing to EnvPool
=======================

Build From Source
-----------------

See :doc:`/pages/build`.

Adding A New Environment
------------------------

To add a new environment in C++ that will be parallelly run by EnvPool,
we provide a developer interface in `envpool/core/env.h
<https://github.com/sail-sg/envpool/blob/master/envpool/core/env.h>`_.

- For a quick and annotated example, please refer to
  `envpool/dummy/ <https://github.com/sail-sg/envpool/tree/master/envpool/dummy>`_.
- `envpool/atari
  <https://github.com/sail-sg/envpool/tree/master/envpool/atari>`_ serves as
  a more complex, real example.


Lint Check
----------

We use several tools to secure code quality, including

- PEP8 code style: flake8, yapf, isort;
- Type check: mypy;
- C++ Google-style: cpplint, clang-format;
- Bazel build file: buildifier;
- License: addlicense;
- Documentation: pydocstyle, doc8.

To make things easier, we create several shortcuts as follows.

To automatically format the code, run:

.. code-block:: bash

    make format

To check if everything conforms to the specification, run:

.. code-block:: bash

    make lint


Test Locally
------------

This command will run automatic tests in the main directory

.. code-block:: bash

    $ make bazel-test


Documentation
-------------

Documentations are written under the ``docs/`` directory as ReStructuredText
(``.rst``) files. ``index.rst`` is the main page. A Tutorial on
ReStructuredText can be found `here
<https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_.

To compile documentation into web page, run

.. code-block:: bash

    $ make doc

And the website is in `http://localhost:8000 <http://localhost:8000>`_
