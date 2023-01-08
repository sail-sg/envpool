FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/go/bin:$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR $HOME

# setup env

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update \
    && apt-get install -y git curl wget zsh gcc-9 g++-9 build-essential make tmux \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
WORKDIR $HOME
RUN git clone https://github.com/gpakosz/.tmux.git
RUN ln -s -f .tmux/.tmux.conf
RUN cp .tmux/.tmux.conf.local .
RUN echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local
RUN echo "set-option -g history-limit 10000" >> .tmux.conf.local

RUN curl https://pyenv.run | sh

# install go from source

RUN wget https://golang.org/dl/go1.19.4.linux-amd64.tar.gz
RUN rm -rf /usr/local/go && tar -C /usr/local -xzf go1.19.4.linux-amd64.tar.gz
RUN ln -sf /usr/local/go/bin/go /usr/bin/go

# install bazel

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel

RUN bazel version

# install base dependencies

RUN apt-get update \
    && apt-get install -y swig zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev libffi-dev liblzma-dev \
    llvm xz-utils tk-dev libxml2-dev libxmlsec1-dev qtdeclarative5-dev \
    && rm -rf /var/lib/apt/lists/*
# use self-compiled openssl instead of system provided (1.0.2)
RUN apt-get remove -y libssl-dev

# install newest openssl (for py3.10 and py3.11)

RUN wget https://www.openssl.org/source/openssl-1.1.1s.tar.gz
RUN tar xf openssl-1.1.1s.tar.gz
WORKDIR $HOME/openssl-1.1.1s
RUN ./config
RUN make -j
RUN make install

# install python

RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /etc/profile
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /etc/profile
RUN echo 'eval "$(pyenv init -)"' >> /etc/profile

RUN LDFLAGS="-Wl,-rpath,/root/openssl-1.1.1s/lib" CONFIGURE_OPTS="-with-openssl=/root/openssl-1.1.1s" pyenv install -v 3.11-dev
RUN LDFLAGS="-Wl,-rpath,/root/openssl-1.1.1s/lib" CONFIGURE_OPTS="-with-openssl=/root/openssl-1.1.1s" pyenv install -v 3.10-dev
RUN LDFLAGS="-Wl,-rpath,/root/openssl-1.1.1s/lib" CONFIGURE_OPTS="-with-openssl=/root/openssl-1.1.1s" pyenv install -v 3.9-dev
RUN LDFLAGS="-Wl,-rpath,/root/openssl-1.1.1s/lib" CONFIGURE_OPTS="-with-openssl=/root/openssl-1.1.1s" pyenv install -v 3.8-dev
RUN LDFLAGS="-Wl,-rpath,/root/openssl-1.1.1s/lib" CONFIGURE_OPTS="-with-openssl=/root/openssl-1.1.1s" pyenv install -v 3.7-dev

WORKDIR /__w/envpool/envpool
COPY . .

# cache bazel build (cpp only)

RUN bazel build //envpool/utils:image_process_test --config=release
RUN bazel build //envpool/vizdoom/bin:vizdoom_bin --config=release

WORKDIR /app
