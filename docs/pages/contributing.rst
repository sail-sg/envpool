Contributing to EnvPool
=======================


Build From Source
-----------------

See :doc:`/pages/build`.


Adding A New Environment
------------------------

See :doc:`/pages/env`.


Lint Check
----------

We use several tools to secure code quality, including

- PEP8 code style: flake8, yapf, isort;
- Type check: mypy;
- C++ Google-style: cpplint, clang-format, clang-tidy;
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

This command will run automatic tests in the main directory:

.. code-block:: bash

    make bazel-test

If you only want to debug for Bazel build:

.. code-block:: bash

    # this is for general use case
    make bazel-debug
    # this is for a special folder "envpool/classic_control"
    bazel build //envpool/classic_control --config=debug


If you'd like to run only a single test, for example, testing Mujoco
integration; however, you don't want to build other stuff such as OpenCV:

.. code-block:: bash

    bazel test --test_output=all //envpool/mujoco:mujoco_test --config=release

Feel free to customize the command in ``Makefile``!


Documentation
-------------

Documentations are written under the ``docs/`` directory as ReStructuredText
(``.rst``) files. ``index.rst`` is the main page. A Tutorial on
ReStructuredText can be found `here
<https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_.

To compile documentation into the web page, run:

.. code-block:: bash

    make doc

And the website is in `http://localhost:8000 <http://localhost:8000>`_
