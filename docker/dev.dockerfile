# Need docker >= 20.10.9, see https://stackoverflow.com/questions/71941032/why-i-cannot-run-apt-update-inside-a-fresh-ubuntu22-04

FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/go/bin:$PATH
ENV USE_BAZEL_VERSION=8.6.0

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev python-is-python3 golang-go git wget curl zsh tmux vim \
    && rm -rf /var/lib/apt/lists/*
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
WORKDIR $HOME
RUN git clone https://github.com/gpakosz/.tmux.git
RUN ln -s -f .tmux/.tmux.conf
RUN cp .tmux/.tmux.conf.local .
RUN echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local
RUN echo "set-option -g history-limit 10000" >> .tmux.conf.local
RUN echo "export PATH=$PATH:$HOME/go/bin" >> .zshrc

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@latest
RUN $HOME/go/bin/bazel version

RUN useradd -ms /bin/zsh github-action

RUN apt-get update \
    && apt-get install -y clang-format clang-tidy swig qtbase5-dev qtdeclarative5-dev \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/include/x86_64-linux-gnu/qt5 /usr/include/qt

WORKDIR /app
