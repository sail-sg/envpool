Build From Source
=================

We recommend to build EnvPool on Ubuntu 20.04 environment.

We use `bazel <https://bazel.build/>`_ to build EnvPool. Comparing with
`pip <https://pip.pypa.io/>`_, using bazel to build python package with
C++ .so files has some advantages:

- bazel allows us to build from source code but no need to directly includes
  the code in our repo;
- no need to write complex cmake files, especially with multiple third-party
  dependencies;
- using bazel for CI test pipeline can only run the test with modified part,
  so that it can save a lot of time.


Install Bazelisk
----------------

Bazelisk is a version controller for bazel. You can install it via
`npm <https://nodejs.org/en/download/package-manager/#debian-and-ubuntu-based-linux-distributions>`_

.. code-block:: bash

    sudo apt install -y npm
    npm install -g @bazel/bazelisk

or `golang <https://golang.org/doc/install>`_ with version >= 1.16:

.. code-block:: bash

    sudo apt install -y golang
    export PATH=$HOME/go/bin:$PATH
    go install github.com/bazelbuild/bazelisk@latest
    ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel


Install Other Dependencies
--------------------------

EnvPool requires gcc/g++ version >= 9.0 to build the source code. To install:

.. code-block:: bash

    sudo apt install -y gcc-9 g++-9
    # to change the default cc to gcc-9:
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

It also requires python version >= 3.7:

.. code-block:: bash

    sudo apt install -y python3-dev python3-pip
    sudo ln -sf /usr/bin/python3 /usr/bin/python


Build Wheel
-----------

To build a release version, type

.. code-block:: bash

    bazel run --config=release //:setup -- bdist_wheel

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


Use Docker to Create Develop Environment
----------------------------------------

We also provide dockerfile for building such a container. To create a docker
develop environment, run

.. code-block:: bash

    make docker-dev

The code is under ``/app`` and you can communicate with host machine file
system via ``/host``.
