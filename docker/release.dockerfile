FROM quay.io/pypa/manylinux_2_28_x86_64

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/go/bin:$PATH
ENV USE_BAZEL_VERSION=8.6.0

WORKDIR $HOME

RUN dnf install -y \
    git curl wget zsh gcc gcc-c++ make tmux golang java-17-openjdk-devel \
    qt5-qtbase-devel qt5-qtdeclarative-devel perl-IO-Compress \
    && dnf clean all
RUN ln -sf "$(qmake-qt5 -query QT_INSTALL_HEADERS)" /usr/include/qt

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel

RUN bazel version

WORKDIR /__w/envpool/envpool
COPY . .

# cache bazel build (cpp only)

RUN PATH=/opt/python/cp311-cp311/bin:$PATH bazel build //envpool/utils:image_process_test --config=release
RUN PATH=/opt/python/cp311-cp311/bin:$PATH bazel build //envpool/vizdoom/bin:vizdoom_bin --config=release

WORKDIR /app
