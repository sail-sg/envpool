FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ARG PATH=$PATH:$HOME/go/bin

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang-1.16 clang-format-11 git wget \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go
RUN go env -w GOPROXY=https://goproxy.cn

RUN wget https://mirrors.huaweicloud.com/bazel/5.1.1/bazel-5.1.1-linux-x86_64
RUN chmod +x bazel-5.1.1-linux-x86_64
RUN mkdir -p $HOME/go/bin
RUN mv bazel-5.1.1-linux-x86_64 $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@latest
RUN pip3 install --upgrade pip isort yapf cpplint flake8 flake8_bugbear mypy && rm -rf ~/.pip/cache

WORKDIR /app
COPY . .
