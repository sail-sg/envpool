FROM ubuntu:16.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/go/bin:$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH

WORKDIR $HOME

# install base dependencies

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update \
    && apt-get install -y git curl wget gcc-9 g++-9 build-essential patchelf make libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev 
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# install go from source

RUN wget https://golang.org/dl/go1.17.3.linux-amd64.tar.gz
RUN rm -rf /usr/local/go && tar -C /usr/local -xzf go1.17.3.linux-amd64.tar.gz
RUN ln -sf /usr/local/go/bin/go /usr/bin/go

# install bazel

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel

# download pyenv

RUN curl https://pyenv.run | bash

# install python3

WORKDIR /app/scripts
COPY scripts/pyenv_setup.sh .
RUN bash ./pyenv_setup.sh

WORKDIR /app
COPY . .

# compile and test release wheels

RUN bash ./scripts/pyenv_build.sh
RUN bash ./scripts/pyenv_test.sh
