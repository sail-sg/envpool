FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang clang-format-11 git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN go get github.com/bazelbuild/bazelisk && ln -s $HOME/go/bin/bazelisk /usr/bin/bazel
RUN go get github.com/bazelbuild/buildtools/buildifier && ln -s $HOME/go/bin/buildifier /usr/bin/buildifier
RUN pip3 install --upgrade pip isort yapf cpplint flake8 flake8_bugbear mypy && rm -rf ~/.pip/cache

WORKDIR /app
COPY . .
