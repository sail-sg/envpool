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
        sha256 = "954aa89b491be4a083304a2cb838019c8b8c3720a7abb9c4cb81ac7a24230cea",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz",
            "https://github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_foreign_cc",
        sha256 = "69023642d5781c68911beda769f91fcbc8ca48711db935a75da7f6536b65047f",
        strip_prefix = "rules_foreign_cc-0.6.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.6.0.tar.gz",
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        sha256 = "a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d",
        strip_prefix = "pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
        urls = [
            "https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "8ff2fff22df038f5cd02cea8af56622bc67f5b64534f1b83b9f133b8366acff2",
        strip_prefix = "pybind11-2.6.2",
        urls = [
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.6.2.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "59b862f50e710277f8ede96f083a5bb8d7c9595376146838b9580be90374ee1f",
        strip_prefix = "abseil-cpp-20210324.2",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20210324.2.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_github_gflags_gflags",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
    )

    maybe(
        http_archive,
        name = "com_github_google_glog",
        sha256 = "21bc744fb7f2fa701ee8db339ded7dce4f975d0d55837a97be7d46e8382dea5a",
        strip_prefix = "glog-0.5.0",
        urls = ["https://github.com/google/glog/archive/v0.5.0.zip"],
    )

    maybe(
        http_archive,
        name = "com_google_googletest",
        sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
        strip_prefix = "googletest-release-1.11.0",
        urls = [
            "https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "glibc_version_header",
        sha256 = "57db74f933b7a9ea5c653498640431ce0e52aaef190d6bb586711ec4f8aa2b9e",
        strip_prefix = "glibc_version_header-0.1/version_headers/",
        urls = [
            "https://github.com/wheybags/glibc_version_header/archive/refs/tags/0.1.tar.gz",
        ],
        build_file = "//third_party/glibc_version_header:glibc_version_header.BUILD",
    )

    maybe(
        http_archive,
        name = "concurrentqueue",
        sha256 = "eb37336bf9ae59aca7b954db3350d9b30d1cab24b96c7676f36040aa76e915e8",
        strip_prefix = "concurrentqueue-1.0.3",
        urls = [
            "https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.3.tar.gz",
        ],
        build_file = "//third_party/concurrentqueue:concurrentqueue.BUILD",
    )

    maybe(
        http_archive,
        name = "threadpool",
        sha256 = "18854bb7ecc1fc9d7dda9c798a1ef0c81c2dd331d730c76c75f648189fa0c20f",
        strip_prefix = "ThreadPool-9a42ec1329f259a5f4881a291db1dcb8f2ad9040",
        urls = [
            "https://github.com/progschj/ThreadPool/archive/9a42ec1329f259a5f4881a291db1dcb8f2ad9040.zip",
        ],
        build_file = "//third_party/threadpool:threadpool.BUILD",
    )

    maybe(
        http_archive,
        name = "zlib",
        sha256 = "629380c90a77b964d896ed37163f5c3a34f6e6d897311f1df2a7016355c45eff",
        strip_prefix = "zlib-1.2.11",
        urls = [
            "https://github.com/madler/zlib/archive/refs/tags/v1.2.11.tar.gz",
        ],
        build_file = "//third_party/zlib:zlib.BUILD",
    )

    maybe(
        http_archive,
        name = "opencv",
        urls = [
            "https://github.com/opencv/opencv/archive/refs/tags/4.5.4.tar.gz",
        ],
        sha256 = "c20bb83dd790fc69df9f105477e24267706715a9d3c705ca1e7f613c7b3bad3d",
        strip_prefix = "opencv-4.5.4",
        build_file = "//third_party/opencv:opencv.BUILD",
    )

    maybe(
        http_archive,
        name = "ale",
        sha256 = "e3bada34cc6c116377c4a807c24d9890ce33afa854ffc45e32dc90ba0dcc9140",
        strip_prefix = "Arcade-Learning-Environment-0.7.3",
        urls = [
            "https://github.com/mgbellemare/Arcade-Learning-Environment/archive/refs/tags/v0.7.3.tar.gz",
        ],
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
        urls = [
            "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.5.tar.gz",
        ],
        build_file = "//third_party/jpeg:jpeg.BUILD",
    )

    maybe(
        http_archive,
        name = "nasm",
        sha256 = "63ec86477ad3f0f6292325fd89e1d93aea2e2fd490070863f17d48f7cd387011",
        strip_prefix = "nasm-2.13.03",
        urls = [
            "https://cdn.sail.sea.com/sail/bazel/nasm-2.13.03.tar.bz2",
            "https://storage.googleapis.com/mirror.tensorflow.org/www.nasm.us/pub/nasm/releasebuilds/2.13.03/nasm-2.13.03.tar.bz2",
        ],
        build_file = "//third_party/nasm:nasm.BUILD",
    )

    maybe(
        http_archive,
        name = "sdl2",
        sha256 = "65be9ff6004034b5b2ce9927b5a4db1814930f169c4b2dae0a1e4697075f287b",
        strip_prefix = "SDL2-2.0.16",
        urls = [
            "https://cdn.sail.sea.com/sail/bazel/SDL2-2.0.16.tar.gz",
            "https://www.libsdl.org/release/SDL2-2.0.16.tar.gz",
        ],
        build_file = "//third_party/sdl2:sdl2.BUILD",
    )

    maybe(
        http_archive,
        name = "com_github_nelhage_rules_boost",
        sha256 = "b3cbdceaa95b8cfe3a69ff37f8ad0e53a77937433234f6b9a6add2eff5bde333",
        strip_prefix = "rules_boost-1e3a69bf2d5cd10c34b74f066054cd335d033d71",
        urls = [
            "https://github.com/nelhage/rules_boost/archive/1e3a69bf2d5cd10c34b74f066054cd335d033d71.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "boost",
        build_file = "@com_github_nelhage_rules_boost//:BUILD.boost",
        patch_cmds = ["rm -f doc/pdf/BUILD"],
        sha256 = "d73a8da01e8bf8c7eda40b4c84915071a8c8a0df4a6734537ddde4a8580524ee",
        strip_prefix = "boost_1_71_0",
        urls = [
            "https://cdn.sail.sea.com/sail/bazel/boost_1_71_0.tar.bz2",
            "https://mirror.bazel.build/dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.bz2",
            "https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.bz2",
        ],
    )

    maybe(
        http_archive,
        name = "freedoom",
        sha256 = "f42c6810fc89b0282de1466c2c9c7c9818031a8d556256a6db1b69f6a77b5806",
        strip_prefix = "freedoom-0.12.1/",
        urls = [
            "https://github.com/freedoom/freedoom/releases/download/v0.12.1/freedoom-0.12.1.zip",
        ],
        build_file = "//third_party/freedoom:freedoom.BUILD",
    )

    maybe(
        http_archive,
        name = "vizdoom",
        sha256 = "1ec04a66da0f77616d31b8f052d0233070ae5f8e5da1cacfb0c45c04b18c99d6",
        strip_prefix = "ViZDoom-1.1.11/src/vizdoom/",
        urls = [
            "https://github.com/mwydmuch/ViZDoom/archive/refs/tags/1.1.11.tar.gz",
        ],
        build_file = "//third_party/vizdoom:vizdoom.BUILD",
        patches = [
            "//third_party/vizdoom:sdl_thread.patch",
        ],
    )

    maybe(
        http_archive,
        name = "vizdoom_lib",
        sha256 = "1ec04a66da0f77616d31b8f052d0233070ae5f8e5da1cacfb0c45c04b18c99d6",
        strip_prefix = "ViZDoom-1.1.11/",
        urls = [
            "https://github.com/mwydmuch/ViZDoom/archive/refs/tags/1.1.11.tar.gz",
        ],
        build_file = "//third_party/vizdoom_lib:vizdoom_lib.BUILD",
    )

    maybe(
        http_archive,
        name = "vizdoom_extra_maps",
        sha256 = "325440fe566ff478f35947c824ea5562e2735366845d36c5a0e40867b59f7d69",
        strip_prefix = "DirectFuturePrediction-b4757769f167f1bd7fb1ece5fdc6d874409c68a9/",
        urls = [
            "https://github.com/isl-org/DirectFuturePrediction/archive/b4757769f167f1bd7fb1ece5fdc6d874409c68a9.zip",
        ],
        build_file = "//third_party/vizdoom_extra_maps:vizdoom_extra_maps.BUILD",
    )

    # Atari/VizDoom pretrained weight for testing pipeline

    maybe(
        http_archive,
        name = "pretrain_weight",
        sha256 = "b1b64e0db84cf7317c2a96b27f549147dfcb4074ed2d799334c23a067075ac1c",
        urls = ["https://cdn.sail.sea.com/sail/pretrain.tar.gz"],
        build_file = "//third_party/pretrain_weight:pretrain_weight.BUILD",
    )

    mypy_integration_version = "0.2.0"  # Latest @ 26th June 2021

    maybe(
        http_archive,
        name = "mypy_integration",
        sha256 = "621df076709dc72809add1f5fe187b213fee5f9b92e39eb33851ab13487bd67d",
        strip_prefix = "bazel-mypy-integration-{version}".format(version = mypy_integration_version),
        urls = [
            "https://github.com/thundergolfer/bazel-mypy-integration/archive/refs/tags/{version}.tar.gz".format(version = mypy_integration_version),
        ],
    )

    maybe(
        http_archive,
        name = "procgen",
        sha256 = "8d443b7b8fba44ef051b182e9a87abfa4e05292568e476ca1e5f08f9666a1b72",
        strip_prefix = "procgen-0.10.7/procgen/src/",
        urls = [
            "https://github.com/openai/procgen/archive/refs/tags/0.10.7.zip",
        ],
        patches = [
            "//third_party/procgen:assetgen.patch",
            "//third_party/procgen:qt-utils.patch",
            "//third_party/procgen:libenv.patch",
        ],
        build_file = "//third_party/procgen:procgen.BUILD",
    )

workspace0 = workspace
