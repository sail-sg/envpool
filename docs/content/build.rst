Build From Source
=================

We recommend building EnvPool on Ubuntu 20.04 environment.

We use `bazel <https://bazel.build/>`_ to build EnvPool. Comparing with
`pip <https://pip.pypa.io/>`_, using Bazel to build python package with C++ .so
files has some advantages:

- Bazel allows us to build from source code, but no need to directly includes
  the code in our repo;
- no need to write complex CMake files, especially with multiple third-party
  dependencies;
- using Bazel for CI test pipeline can only run the test with modified part,
  so that it can save a lot of time.


Install Bazelisk
----------------

Bazelisk is a version controller for Bazel. You can install it via
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


.. note ::

    For users in mainland China, please do the following step to install go and bazel:

    1. Install golang >= 1.16 from other sites, e.g., https://studygolang.com/dl
    2. Change go proxy: ``go env -w GOPROXY=https://goproxy.cn``
    3. Install bazel from https://mirrors.huaweicloud.com/bazel/

    .. code-block:: bash

        wget https://studygolang.com/dl/golang/go1.18.1.linux-amd64.tar.gz
        # then follow the instructions on golang official website
        go env -w GOPROXY=https://goproxy.cn

        wget https://mirrors.huaweicloud.com/bazel/5.1.1/bazel-5.1.1-linux-x86_64
        chmod +x bazel-5.1.1-linux-x86_64
        mkdir -p $HOME/go/bin
        mv bazel-5.1.1-linux-x86_64 $HOME/go/bin/bazel

        export PATH=$PATH:$HOME/go/bin  # or write to .bashrc / .zshrc

        # check if successfully installed
        bazel

    See `Issue #87 <https://github.com/sail-sg/envpool/issues/87>`_.


Install Other Dependencies
--------------------------

EnvPool requires **GCC/G++ version >= 9.0** to build the source code. To install:

.. code-block:: bash

    # optional
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test

    # install
    sudo apt install -y gcc-9 g++-9 build-essential

    # to change the default cc to gcc-9:
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

It also requires **Python version >= 3.7**:

.. code-block:: bash

    sudo apt install -y python3-dev python3-pip
    sudo ln -sf /usr/bin/python3 /usr/bin/python

Install CUDA to enable XLA: see https://developer.nvidia.com/cuda-downloads

Install other dependencies: see
`Dockerfile <https://github.com/sail-sg/envpool/tree/main/docker>`_.


Build Wheel
-----------

To build a release version, type:

.. code-block:: bash

    bazel run --config=release //:setup -- bdist_wheel

This creates a wheel under ``bazel-bin/setup.runfiles/envpool/dist``.


.. note ::

    For users in mainland China:

    - If you find ``pip install`` is quite slow to fetch 3rd-party libraries,
      the solution is to uncomment ``extra_args`` in ``envpool/pip.bzl`` to
      switch the pip source.
    - If you find ``bazel build`` is quite slow to fetch 3rd-party libraries,
      please refer https://docs.bazel.build/versions/main/external.html#using-proxies

      .. code-block:: bash

        export HTTP_PROXY=http://...
        export HTTPS_PROXY=http://...
        # then run the command to build

    See `Issue #87 <https://github.com/sail-sg/envpool/issues/87>`_.


Use Shortcut
------------

We provide several shortcuts to make things easier.

.. code-block:: bash

    # This will install bazelisk via golang, need sudo
    make bazel-install

    # This will verbose all compile commands to help debug
    make bazel-debug

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

The code is under ``/app``, and you can communicate with the host machine file
system via ``/host``.

.. note ::

    For users in mainland China:

    .. code-block:: bash

        make docker-dev-cn

    See `Issue #87 <https://github.com/sail-sg/envpool/issues/87>`_.
