FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/go/bin:$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH

WORKDIR $HOME

# install base dependencies

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update \
    && apt-get install -y git curl wget gcc-9 g++-9 build-essential swig make libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN curl https://pyenv.run | sh

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# install go from source

RUN wget https://golang.org/dl/go1.19.4.linux-amd64.tar.gz
RUN rm -rf /usr/local/go && tar -C /usr/local -xzf go1.19.4.linux-amd64.tar.gz
RUN ln -sf /usr/local/go/bin/go /usr/bin/go

# install bazel

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel

RUN bazel version

# install newest openssl (for py3.10 and py3.11)

RUN wget https://www.openssl.org/source/openssl-1.1.1s.tar.gz
RUN tar xf openssl-1.1.1s.tar.gz
WORKDIR $HOME/openssl-1.1.1s
RUN ./config no-shared
RUN make -j
RUN make install

# install python

RUN CPPFLAGS=-I$(pwd)/include LDFLAGS=-L$(pwd)/lib pyenv install 3.10-dev
RUN CPPFLAGS=-I$(pwd)/include LDFLAGS=-L$(pwd)/lib pyenv install 3.9-dev
RUN CPPFLAGS=-I$(pwd)/include LDFLAGS=-L$(pwd)/lib pyenv install 3.8-dev
RUN CPPFLAGS=-I$(pwd)/include LDFLAGS=-L$(pwd)/lib pyenv install 3.7-dev

RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /etc/profile
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /etc/profile
RUN echo 'eval "$(pyenv init -)"' >> /etc/profile
