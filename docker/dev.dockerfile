FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ARG PATH=$PATH:$HOME/go/bin

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang-1.16 clang-format-11 git wget swig \
    && rm -rf /var/lib/apt/lists/* \
RUN apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update && apt-get install -y cuda
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@latest
RUN pip3 install --upgrade pip isort yapf cpplint flake8 flake8_bugbear mypy && rm -rf ~/.pip/cache

WORKDIR /app
COPY . .
