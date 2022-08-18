FROM ubuntu:16.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/go/bin:$PATH

WORKDIR $HOME

# install base dependencies

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update \
    && apt-get install -y git curl wget gcc-9 g++-9 build-essential patchelf make libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev swig \
    python3.7 python3.8 python3.9 python3.10 \
    python3.7-dev python3.8-dev python3.9-dev python3.10-dev \
    python3.8-distutils python3.9-distutils python3.10-distutils
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# install pip

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN for i in 7 8 9 10; do ln -sf /usr/bin/python3.$i /usr/bin/python3; python3 get-pip.py; done

# install go from source

RUN wget https://golang.org/dl/go1.17.3.linux-amd64.tar.gz
RUN rm -rf /usr/local/go && tar -C /usr/local -xzf go1.17.3.linux-amd64.tar.gz
RUN ln -sf /usr/local/go/bin/go /usr/bin/go

# install bazel

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel

# install big wheels

RUN for i in 7 8 9; do ln -sf /usr/bin/python3.$i /usr/bin/python3; pip3 install torch opencv-python-headless; done

RUN bazel version

WORKDIR /app
COPY . .

# compile and test release wheels

RUN for i in 7 8 9; do ln -sf /usr/bin/python3.$i /usr/bin/python3; make pypi-wheel BAZELOPT="--remote_cache=http://bazel-cache.sail:8080"; pip3 install wheelhouse/*cp3$i*.whl; rm dist/*.whl; make release-test; done
