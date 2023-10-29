# Need docker >= 20.10.9, see https://stackoverflow.com/questions/71941032/why-i-cannot-run-apt-update-inside-a-fresh-ubuntu22-04

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ARG PATH=$PATH:$HOME/go/bin

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang-1.18 git wget curl zsh tmux vim \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/lib/go-1.18/bin/go /usr/bin/go
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
WORKDIR $HOME
RUN git clone https://github.com/gpakosz/.tmux.git
RUN ln -s -f .tmux/.tmux.conf
RUN cp .tmux/.tmux.conf.local .
RUN echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local
RUN echo "set-option -g history-limit 10000" >> .tmux.conf.local
RUN go env -w GOPROXY=https://goproxy.cn

RUN wget https://mirrors.huaweicloud.com/bazel/6.0.0/bazel-6.0.0-linux-x86_64
RUN chmod +x bazel-6.0.0-linux-x86_64
RUN mkdir -p $HOME/go/bin
RUN mv bazel-6.0.0-linux-x86_64 $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@latest
RUN $HOME/go/bin/bazel version

RUN useradd -ms /bin/zsh github-action

RUN apt-get update \
    && apt-get install -y clang-format clang-tidy swig qtdeclarative5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

WORKDIR /app
