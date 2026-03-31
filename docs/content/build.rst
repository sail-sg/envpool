Build From Source
=================

We recommend developing EnvPool on Ubuntu 24.04. Release wheels are built for
Python 3.11-3.14 in ``manylinux_2_28_x86_64`` and
``manylinux_2_28_aarch64`` environments for Linux, on ``macos-14`` for
macOS, and on ``windows-2022`` for Windows.

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

    sudo apt install -y golang-go
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

        wget https://mirrors.huaweicloud.com/bazel/8.6.0/bazel-8.6.0-linux-x86_64
        chmod +x bazel-8.6.0-linux-x86_64
        mkdir -p $HOME/go/bin
        mv bazel-8.6.0-linux-x86_64 $HOME/go/bin/bazel

        export PATH=$PATH:$HOME/go/bin  # or write to .bashrc / .zshrc

        # check if successfully installed
        bazel

    See `Issue #87 <https://github.com/sail-sg/envpool/issues/87>`_.


Install Other Dependencies
--------------------------

EnvPool source builds share a few common requirements across platforms:

- **Python >= 3.11**
- **Java 17**
- **Go >= 1.22** plus ``bazelisk`` / ``bazel``
- **SWIG**
- **Qt 5**

The default build and test shortcuts in this repo use **Bazel 8.6.0** via
``bazelisk``.

Linux (Ubuntu 24.04)
^^^^^^^^^^^^^^^^^^^^

EnvPool currently builds with the system GCC/G++ toolchain on Ubuntu 24.04. To
install the required development packages:

.. code-block:: bash

    sudo apt install -y build-essential openjdk-17-jdk python3-dev \
      python3-pip python-is-python3 golang-go cmake ninja-build swig qtbase5-dev \
      qtdeclarative5-dev

    # Some Bazel Qt rules still look for this legacy include path.
    sudo ln -sf "$(qmake -query QT_INSTALL_HEADERS)" /usr/include/qt

macOS
^^^^^

Install the Xcode Command Line Tools first, then install the user-space
dependencies with Homebrew:

.. code-block:: bash

    xcode-select --install
    brew install go openjdk@17 cmake ninja swig qt@5
    sudo ln -sf "$(brew --prefix cmake)/bin/cmake" /usr/local/bin/cmake
    sudo ln -sf "$(brew --prefix ninja)/bin/ninja" /usr/local/bin/ninja

    export PATH="$(brew --prefix openjdk@17)/bin:$PATH"
    export BAZEL_RULES_QT_DIR="$(brew --prefix qt@5)"

Windows
^^^^^^^

Install the following before building from source:

- Visual Studio 2022 (or Build Tools 2022) with the **Desktop development with
  C++** workload
- Git for Windows
- Java 17
- Go 1.22+
- Qt 5.15.2 with the ``msvc2019_64`` toolchain

Then install the remaining command-line dependencies and export the Bazel / Qt
environment variables:

.. code-block:: powershell

    choco install -y cmake make ninja strawberryperl swig
    $env:BAZEL_SH = "C:/Program Files/Git/usr/bin/bash.exe"
    $env:QT_ROOT_DIR = "C:/Qt/5.15.2/msvc2019_64"
    $env:BAZEL_RULES_QT_DIR = $env:QT_ROOT_DIR
    $env:PATH = "$env:QT_ROOT_DIR\\bin;$env:PATH"

Qt runtime in wheels
^^^^^^^^^^^^^^^^^^^^

Windows release wheels currently bundle the Qt runtime DLLs required by
Procgen (``Qt5Core.dll`` and ``Qt5Gui.dll``) directly next to
``procgen_envpool.pyd``. End users installing the wheel do **not** need a
separate Qt installation at runtime.

Source builds still require a local Qt 5 installation so Bazel can compile and
link against Qt. Linux and macOS continue to rely on system Qt packages at
build time.

Install CUDA to enable XLA: see https://developer.nvidia.com/cuda-downloads

Install other dependencies: see
`Dockerfile <https://github.com/sail-sg/envpool/tree/main/docker>`_.


Build Wheel
-----------

To build a release version, type:

.. code-block:: bash

    cp third_party/pip_requirements/requirements-release.txt third_party/pip_requirements/requirements.txt
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

    # This will build a wheel for release
    make bazel-release


Use Docker to Create Develop Environment
----------------------------------------

We also provide dockerfile for building such a container. To create a docker
develop environment, run

.. code-block:: bash

    make docker-dev

The code is under ``/app``, and you can communicate with the host machine file
system via ``/host``.
