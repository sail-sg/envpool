# Need docker >= 20.10.9, see https://stackoverflow.com/questions/71941032/why-i-cannot-run-apt-update-inside-a-fresh-ubuntu22-04

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ARG PATH=$PATH:$HOME/go/bin

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang-1.18 clang-format-11 git wget swig \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/lib/go-1.18/bin/go /usr/bin/go
RUN go env -w GOPROXY=https://goproxy.cn

RUN wget https://mirrors.huaweicloud.com/bazel/6.0.0/bazel-6.0.0-linux-x86_64
RUN chmod +x bazel-6.0.0-linux-x86_64
RUN mkdir -p $HOME/go/bin
RUN mv bazel-6.0.0-linux-x86_64 $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@latest
RUN $HOME/go/bin/bazel version
RUN pip3 install --upgrade pip isort yapf cpplint flake8 flake8_bugbear mypy

WORKDIR /app
COPY . .
