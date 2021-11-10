Build From Source
=================

EnvPool is managed with bazel, install bazel via bazelisk

.. code-block:: bash

    sudo apt install npm python3-dev
    npm install -g @bazel/bazelisk

To build a release version, run the following

.. code-block:: bash

    bazel build --config=release //:setup -- bdist_wheel

This creates a wheel under ``bazel-bin/setup.runfiles/envpool/dist``.


Use Shortcut
------------

We provide several shortcuts to make things easier.

.. code-block:: bash

    # This will install bazelisk via golang, need sudo
    make bazel-install

    # This will build python wheel (.whl) file under `dist/` folder
    make bazel-build

    # This will automatically run the tests
    make bazel-test
