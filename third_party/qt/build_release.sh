#!/usr/bin/env bash
# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the shared Qt raster runtime in the target manylinux image, not on the
# host: upstream Qt binaries have a newer glibc baseline, especially on ARM64.
set -Eeuo pipefail

qt_version=6.11.1
qt_sha256=d9594a31228aa23ad6b531719a29b45f0f3989fe6c136d45767ea179f233c1ac
qt_url="https://download.qt.io/official_releases/qt/6.11/$qt_version/submodules/qtbase-everywhere-src-$qt_version.tar.xz"
prefix=${1:-/opt/envpool-qt6}
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
archive=${QT_SOURCE_ARCHIVE:-$work_dir/qtbase.tar.xz}
if [[ -z ${QT_SOURCE_ARCHIVE:-} ]]; then
  curl --fail --location --retry 3 --output "$archive" "$qt_url"
fi
printf '%s  %s\n' "$qt_sha256" "$archive" | sha256sum --check
tar --no-same-owner -xf "$archive" -C "$work_dir"
source_dir="$work_dir/qtbase-everywhere-src-$qt_version"
mkdir "$work_dir/build"
cd "$work_dir/build"

# Procgen uses QImage/QPainter and PNG assets, without windows or text. Keep
# PNG in QtGui and build its small dependencies from Qt's pinned sources;
# disabling ICU and display/font backends avoids a large system runtime stack.
"$source_dir/configure" \
  -prefix "$prefix" -libdir lib -release -shared \
  -opensource -confirm-license -nomake examples -nomake tests \
  -no-feature-network -no-feature-dbus -no-feature-sql \
  -no-feature-widgets -no-feature-printsupport -no-feature-testlib \
  -no-feature-xml -no-feature-concurrent \
  -no-icu -no-glib -no-zstd -no-opengl -no-feature-vulkan \
  -no-xcb -no-feature-wayland -no-feature-eglfs -no-feature-linuxfb \
  -no-fontconfig -no-freetype -no-harfbuzz -no-libjpeg \
  -qt-libpng -qt-zlib -qt-pcre \
  -- -DQT_INSTALL_CONFIG_INFO_FILES=ON
cmake --build . --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
cmake --install . --strip

# Ship upstream license texts and attributions alongside the dynamically
# linked libraries. The SBOM describes the Qt build before auditwheel repair.
license_dir="$prefix/licenses/qt6"
mkdir -p "$license_dir"
cp -R "$source_dir/LICENSES" "$prefix/sbom" "$license_dir/"
cp "$prefix/config_qtbase.opt" "$prefix/config_qtbase.summary" "$license_dir/"
cat > "$license_dir/README.txt" <<EOF
This wheel includes Qt Core and Qt Gui $qt_version, dynamically linked under
LGPL-3.0-only, and the third-party code identified in the Qt build SBOM.
Copyright (C) The Qt Company Ltd. and other contributors.
License texts are in LICENSES/; component copyright notices are in sbom/.
The SBOM describes the upstream build before auditwheel renaming and stripping.

Unmodified corresponding Qt sources: $qt_url
SHA256: $qt_sha256
Build instructions and configuration: third_party/qt/build_release.sh in
https://github.com/sail-sg/envpool
The libraries in envpool.libs can be replaced with compatible modified Qt
libraries; preserve their auditwheel-assigned filenames and dependency names.
EOF
