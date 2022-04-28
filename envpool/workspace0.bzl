# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EnvPool workspace initialization, this is loaded in WORKSPACE."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def workspace():
    """Load requested packages."""
    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "9fcf91dbcc31fde6d1edb15f117246d912c33c36f44cf681976bd886538deba6",
        strip_prefix = "rules_python-0.8.0",
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.8.0.tar.gz",
    )

    maybe(
        http_archive,
        name = "rules_foreign_cc",
        sha256 = "6041f1374ff32ba711564374ad8e007aef77f71561a7ce784123b9b4b88614fc",
        strip_prefix = "rules_foreign_cc-0.8.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.8.0.tar.gz",
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        sha256 = "fec6281e4109115c5157ca720b8fe20c8f655f773172290b03f57353c11869c2",
        strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
        url = "https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.zip",
    )

    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1",
        strip_prefix = "pybind11-2.9.2",
        url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz",
    )

    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "dcf71b9cba8dc0ca9940c4b316a0c796be8fab42b070bb6b7cab62b48f0e66c4",
        strip_prefix = "abseil-cpp-20211102.0",
        url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
    )

    maybe(
        http_archive,
        name = "com_github_gflags_gflags",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        url = "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
    )

    maybe(
        http_archive,
        name = "com_github_google_glog",
        sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
        strip_prefix = "glog-0.6.0",
        url = "https://github.com/google/glog/archive/v0.6.0.zip",
    )

    maybe(
        http_archive,
        name = "com_google_googletest",
        sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
        strip_prefix = "googletest-release-1.11.0",
        url = "https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz",
    )

    maybe(
        http_archive,
        name = "glibc_version_header",
        sha256 = "57db74f933b7a9ea5c653498640431ce0e52aaef190d6bb586711ec4f8aa2b9e",
        strip_prefix = "glibc_version_header-0.1/version_headers/",
        url = "https://github.com/wheybags/glibc_version_header/archive/refs/tags/0.1.tar.gz",
        build_file = "//third_party/glibc_version_header:glibc_version_header.BUILD",
    )

    maybe(
        http_archive,
        name = "concurrentqueue",
        sha256 = "eb37336bf9ae59aca7b954db3350d9b30d1cab24b96c7676f36040aa76e915e8",
        strip_prefix = "concurrentqueue-1.0.3",
        url = "https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.3.tar.gz",
        build_file = "//third_party/concurrentqueue:concurrentqueue.BUILD",
    )

    maybe(
        http_archive,
        name = "threadpool",
        sha256 = "18854bb7ecc1fc9d7dda9c798a1ef0c81c2dd331d730c76c75f648189fa0c20f",
        strip_prefix = "ThreadPool-9a42ec1329f259a5f4881a291db1dcb8f2ad9040",
        url = "https://github.com/progschj/ThreadPool/archive/9a42ec1329f259a5f4881a291db1dcb8f2ad9040.zip",
        build_file = "//third_party/threadpool:threadpool.BUILD",
    )

    maybe(
        http_archive,
        name = "zlib",
        sha256 = "d8688496ea40fb61787500e863cc63c9afcbc524468cedeb478068924eb54932",
        strip_prefix = "zlib-1.2.12",
        url = "https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz",
        build_file = "//third_party/zlib:zlib.BUILD",
    )

    maybe(
        http_archive,
        name = "opencv",
        url = "https://github.com/opencv/opencv/archive/refs/tags/4.5.5.tar.gz",
        sha256 = "a1cfdcf6619387ca9e232687504da996aaa9f7b5689986b8331ec02cb61d28ad",
        strip_prefix = "opencv-4.5.5",
        build_file = "//third_party/opencv:opencv.BUILD",
    )

    maybe(
        http_archive,
        name = "ale",
        sha256 = "31ff9a51187a1237b683c7a5eff232b294f0bd48ca078ab57eabc9e3564ff1c1",
        strip_prefix = "Arcade-Learning-Environment-0.7.5",
        url = "https://github.com/mgbellemare/Arcade-Learning-Environment/archive/refs/tags/v0.7.5.tar.gz",
        build_file = "//third_party/ale:ale.BUILD",
    )

    maybe(
        http_archive,
        name = "atari_roms",
        sha256 = "e39e9fc379fe3f336911d928ce0a52e6ff6861258906efc5e849390867ff35f5",
        urls = [
            "https://roms8.s3.us-east-2.amazonaws.com/Roms.tar.gz",
            "https://cdn.sail.sea.com/sail/Roms.tar.gz",
        ],
        build_file = "//third_party/atari_roms:atari_roms.BUILD",
    )

    maybe(
        http_archive,
        name = "libjpeg_turbo",
        sha256 = "b3090cd37b5a8b3e4dbd30a1311b3989a894e5d3c668f14cbc6739d77c9402b7",
        strip_prefix = "libjpeg-turbo-2.0.5",
        url = "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.5.tar.gz",
        build_file = "//third_party/jpeg:jpeg.BUILD",
    )

    maybe(
        http_archive,
        name = "nasm",
        sha256 = "63ec86477ad3f0f6292325fd89e1d93aea2e2fd490070863f17d48f7cd387011",
        strip_prefix = "nasm-2.13.03",
        url = "https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.13.03.tar.bz2",
        build_file = "//third_party/nasm:nasm.BUILD",
    )

    maybe(
        http_archive,
        name = "sdl2",
        sha256 = "c56aba1d7b5b0e7e999e4a7698c70b63a3394ff9704b5f6e1c57e0c16f04dd06",
        strip_prefix = "SDL2-2.0.20",
        url = "https://www.libsdl.org/release/SDL2-2.0.20.tar.gz",
        build_file = "//third_party/sdl2:sdl2.BUILD",
    )

    maybe(
        http_archive,
        name = "com_github_nelhage_rules_boost",
        sha256 = "ce95d0705592e51eda91b38870e847c303f65219871683f7c34233caad150b0b",
        strip_prefix = "rules_boost-32164a62e2472077320f48f52b8077207cd0c9c8",
        url = "https://github.com/nelhage/rules_boost/archive/32164a62e2472077320f48f52b8077207cd0c9c8.tar.gz",
    )

    maybe(
        http_archive,
        name = "boost",
        build_file = "@com_github_nelhage_rules_boost//:BUILD.boost",
        patch_cmds = ["rm -f doc/pdf/BUILD"],
        sha256 = "475d589d51a7f8b3ba2ba4eda022b170e562ca3b760ee922c146b6c65856ef39",
        strip_prefix = "boost_1_79_0",
        url = "https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.bz2",
    )

    maybe(
        http_archive,
        name = "freedoom",
        sha256 = "f42c6810fc89b0282de1466c2c9c7c9818031a8d556256a6db1b69f6a77b5806",
        strip_prefix = "freedoom-0.12.1/",
        url = "https://github.com/freedoom/freedoom/releases/download/v0.12.1/freedoom-0.12.1.zip",
        build_file = "//third_party/freedoom:freedoom.BUILD",
    )

    maybe(
        http_archive,
        name = "vizdoom",
        sha256 = "e379a242ada7e1028b7a635da672b0936d99da3702781b76a4400b83602d78c4",
        strip_prefix = "ViZDoom-1.1.13/src/vizdoom/",
        url = "https://github.com/mwydmuch/ViZDoom/archive/refs/tags/1.1.13.tar.gz",
        build_file = "//third_party/vizdoom:vizdoom.BUILD",
        patches = [
            "//third_party/vizdoom:sdl_thread.patch",
        ],
    )

    maybe(
        http_archive,
        name = "vizdoom_lib",
        sha256 = "e379a242ada7e1028b7a635da672b0936d99da3702781b76a4400b83602d78c4",
        strip_prefix = "ViZDoom-1.1.13/",
        url = "https://github.com/mwydmuch/ViZDoom/archive/refs/tags/1.1.13.tar.gz",
        build_file = "//third_party/vizdoom_lib:vizdoom_lib.BUILD",
    )

    maybe(
        http_archive,
        name = "vizdoom_extra_maps",
        sha256 = "325440fe566ff478f35947c824ea5562e2735366845d36c5a0e40867b59f7d69",
        strip_prefix = "DirectFuturePrediction-b4757769f167f1bd7fb1ece5fdc6d874409c68a9/",
        url = "https://github.com/isl-org/DirectFuturePrediction/archive/b4757769f167f1bd7fb1ece5fdc6d874409c68a9.zip",
        build_file = "//third_party/vizdoom_extra_maps:vizdoom_extra_maps.BUILD",
    )

    maybe(
        http_archive,
        name = "procgen",
        sha256 = "8d443b7b8fba44ef051b182e9a87abfa4e05292568e476ca1e5f08f9666a1b72",
        strip_prefix = "procgen-0.10.7/procgen/src/",
        url = "https://github.com/openai/procgen/archive/refs/tags/0.10.7.zip",
        patches = [
            "//third_party/procgen:assetgen.patch",
            "//third_party/procgen:qt-utils.patch",
            "//third_party/procgen:libenv.patch",
            "//third_party/procgen:procgen_games.patch",
        ],
        build_file = "//third_party/procgen:procgen.BUILD",
    )

    # Atari/VizDoom pretrained weight for testing pipeline

    maybe(
        http_archive,
        name = "pretrain_weight",
        sha256 = "b1b64e0db84cf7317c2a96b27f549147dfcb4074ed2d799334c23a067075ac1c",
        url = "https://cdn.sail.sea.com/sail/pretrain.tar.gz",
        build_file = "//third_party/pretrain_weight:pretrain_weight.BUILD",
    )

workspace0 = workspace
