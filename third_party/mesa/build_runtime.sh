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

# Build only the headless software renderer used to test installed wheels.
# AlmaLinux 8 ships Mesa 23.1.4, whose llvmpipe resource lifetime race crashes
# concurrent rendering. The upstream fix first shipped in 23.1.6:
# https://docs.mesa3d.org/relnotes/23.1.6.html
# Keep this prefix out of the wheel build and auditwheel search paths.
set -Eeuo pipefail

mesa_version=24.3.4
mesa_sha256=e641ae27191d387599219694560d221b7feaa91c900bcec46bf444218ed66025
prefix=${1:-/opt/envpool-mesa}
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
archive=${MESA_SOURCE_ARCHIVE:-$work_dir/mesa.tar.xz}
if [[ -z ${MESA_SOURCE_ARCHIVE:-} ]]; then
  curl --fail --location --retry 3 --output "$archive" \
    "https://archive.mesa3d.org/mesa-$mesa_version.tar.xz"
fi
printf '%s  %s\n' "$mesa_sha256" "$archive" | sha256sum --check
tar --no-same-owner -xf "$archive" -C "$work_dir"

meson setup "$work_dir/build" "$work_dir/mesa-$mesa_version" \
  --prefix "$prefix" --libdir lib --buildtype release --wrap-mode nofallback \
  -Dplatforms= -Dgallium-drivers=llvmpipe -Dvulkan-drivers= \
  -Dglx=disabled -Degl=enabled -Dglvnd=enabled -Dgbm=disabled \
  -Dgles1=disabled -Dgles2=disabled -Dllvm=enabled -Dshared-llvm=enabled \
  -Dgallium-vdpau=disabled -Dgallium-va=disabled -Dgallium-xa=disabled \
  -Dvalgrind=disabled -Dbuild-tests=false
meson compile -C "$work_dir/build" -j "${MESA_BUILD_JOBS:-2}"
meson install -C "$work_dir/build" --no-rebuild --strip
cp "$work_dir/build/meson-logs/meson-log.txt" "$prefix/build.log"
