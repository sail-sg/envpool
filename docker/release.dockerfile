ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64
FROM ${MANYLINUX_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ENV PATH=$HOME/go/bin:$PATH
ENV USE_BAZEL_VERSION=9.2.0

WORKDIR $HOME

RUN dnf install -y \
    git curl wget zsh gcc gcc-c++ make tmux golang java-17-openjdk-devel \
    perl-IO-Compress mesa-libEGL-devel mesa-libGL-devel libglvnd-devel mesa-dri-drivers \
    && dnf clean all

ENV PATH=/opt/python/cp312-cp312/bin:$PATH
RUN python3 -m pip install --upgrade cmake ninja
COPY third_party/qt/build_release.sh /tmp/build_qt_release.sh
RUN bash /tmp/build_qt_release.sh /opt/envpool-qt6
ENV BAZEL_RULES_QT_DIR=/opt/envpool-qt6
ENV WHEEL_LICENSE_DIR=/opt/envpool-qt6/licenses/qt6

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel

RUN bazel version

WORKDIR /__w/envpool/envpool
COPY . .

# cache bazel build (cpp only)

RUN PATH=/opt/python/cp312-cp312/bin:$PATH bazel build //envpool/utils:image_process_test --config=release
RUN PATH=/opt/python/cp312-cp312/bin:$PATH bazel build //envpool/vizdoom/bin:vizdoom_bin --config=release

WORKDIR /app
