FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$HOME/go/bin:$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
ENV USE_BAZEL_VERSION=8.6.0

WORKDIR $HOME

RUN apt-get update \
    && apt-get install -y git curl wget zsh gcc g++ build-essential make tmux \
    python3-pip python3-dev python-is-python3 golang-go \
    swig zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncurses-dev libffi-dev liblzma-dev \
    llvm xz-utils tk-dev libxml2-dev libxmlsec1-dev libssl-dev qtbase5-dev qtdeclarative5-dev \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/include/x86_64-linux-gnu/qt5 /usr/include/qt
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
WORKDIR $HOME
RUN git clone https://github.com/gpakosz/.tmux.git
RUN ln -s -f .tmux/.tmux.conf
RUN cp .tmux/.tmux.conf.local .
RUN echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local
RUN echo "set-option -g history-limit 10000" >> .tmux.conf.local

RUN curl https://pyenv.run | sh

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel

RUN bazel version

# install python

RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /etc/profile
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /etc/profile
RUN echo 'eval "$(pyenv init -)"' >> /etc/profile

RUN pyenv install -v 3.11-dev
RUN pyenv install -v 3.12-dev
RUN pyenv install -v 3.13-dev

WORKDIR /__w/envpool/envpool
COPY . .

# cache bazel build (cpp only)

RUN bazel build //envpool/utils:image_process_test --config=release
RUN bazel build //envpool/vizdoom/bin:vizdoom_bin --config=release

WORKDIR /app
