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
load("//third_party/cuda:cuda.bzl", "cuda_configure")

def workspace():
    """Load requested packages."""
    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "b593d13bb43c94ce94b483c2858e53a9b811f6f10e1e0eedc61073bd90e58d9c",
        strip_prefix = "rules_python-0.12.0",
        urls = [
            "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.12.0.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/bazelbuild/rules_python/0.12.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_foreign_cc",
        sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
        strip_prefix = "rules_foreign_cc-0.9.0",
        urls = [
            "https://github.com/bazelbuild/rules_foreign_cc/archive/0.9.0.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/bazelbuild/rules_foreign_cc/0.9.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        sha256 = "a185aa68c93b9f62c80fcb3aadc3c83c763854750dc3f38be1dadcb7be223837",
        strip_prefix = "pybind11_bazel-faf56fb3df11287f26dbc66fdedf60a2fc2c6631",
        urls = [
            "https://github.com/pybind/pybind11_bazel/archive/faf56fb3df11287f26dbc66fdedf60a2fc2c6631.zip",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/pybind/pybind11_bazel/faf56fb3df11287f26dbc66fdedf60a2fc2c6631.zip",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "93bd1e625e43e03028a3ea7389bba5d3f9f2596abc074b068e70f4ef9b1314ae",
        strip_prefix = "pybind11-2.10.2",
        urls = [
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.2.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/pybind/pybind11/v2.10.2.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "91ac87d30cc6d79f9ab974c51874a704de9c2647c40f6932597329a282217ba8",
        strip_prefix = "abseil-cpp-20220623.1",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/abseil/abseil-cpp/20220623.1.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_github_gflags_gflags",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        urls = [
            "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/gflags/gflags/v2.2.2.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_github_google_glog",
        sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
        strip_prefix = "glog-0.6.0",
        urls = [
            "https://github.com/google/glog/archive/v0.6.0.zip",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/google/glog/v0.6.0.zip",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_googletest",
        sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
        strip_prefix = "googletest-release-1.12.1",
        urls = [
            "https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/google/googletest/release-1.12.1.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_justbuchanan_rules_qt",
        sha256 = "6b42a58f062b3eea10ada5340cd8f63b47feb986d16794b0f8e0fde750838348",
        strip_prefix = "bazel_rules_qt-3196fcf2e6ee81cf3a2e2b272af3d4259b84fcf9",
        urls = [
            "https://github.com/justbuchanan/bazel_rules_qt/archive/3196fcf2e6ee81cf3a2e2b272af3d4259b84fcf9.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/justbuchanan/bazel_rules_qt/3196fcf2e6ee81cf3a2e2b272af3d4259b84fcf9.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "glibc_version_header",
        sha256 = "57db74f933b7a9ea5c653498640431ce0e52aaef190d6bb586711ec4f8aa2b9e",
        strip_prefix = "glibc_version_header-0.1/version_headers/",
        urls = [
            "https://github.com/wheybags/glibc_version_header/archive/refs/tags/0.1.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/wheybags/glibc_version_header/0.1.tar.gz",
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
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/cameron314/concurrentqueue/v1.0.3.tar.gz",
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
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/progschj/ThreadPool/9a42ec1329f259a5f4881a291db1dcb8f2ad9040.zip",
        ],
        build_file = "//third_party/threadpool:threadpool.BUILD",
    )

    maybe(
        http_archive,
        name = "zlib",
        sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
        strip_prefix = "zlib-1.2.13",
        urls = [
            "https://github.com/madler/zlib/releases/download/v1.2.13/zlib-1.2.13.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/madler/zlib/zlib-1.2.13.tar.gz",
        ],
        build_file = "//third_party/zlib:zlib.BUILD",
    )

    maybe(
        http_archive,
        name = "opencv",
        sha256 = "8df0079cdbe179748a18d44731af62a245a45ebf5085223dc03133954c662973",
        strip_prefix = "opencv-4.7.0",
        urls = [
            "https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/opencv/opencv/4.7.0.tar.gz",
        ],
        build_file = "//third_party/opencv:opencv.BUILD",
    )

    maybe(
        http_archive,
        name = "pugixml",
        sha256 = "40c0b3914ec131485640fa57e55bf1136446026b41db91c1bef678186a12abbe",
        strip_prefix = "pugixml-1.13/src",
        urls = [
            "https://github.com/zeux/pugixml/releases/download/v1.13/pugixml-1.13.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/zeux/pugixml/pugixml-1.13.tar.gz",
        ],
        build_file = "//third_party/pugixml:pugixml.BUILD",
    )

    maybe(
        http_archive,
        name = "ale",
        sha256 = "9a9f1ad6cd61dfb26895314d409ba69da038b7def295b509964e579027fefd99",
        strip_prefix = "Arcade-Learning-Environment-0.8.0",
        urls = [
            "https://github.com/mgbellemare/Arcade-Learning-Environment/archive/refs/tags/v0.8.0.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/mgbellemare/Arcade-Learning-Environment/v0.8.0.tar.gz",
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
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/atari/Roms.tar.gz",
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
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/libjpeg-turbo/libjpeg-turbo/2.0.5.tar.gz",
        ],
        build_file = "//third_party/jpeg:jpeg.BUILD",
    )

    maybe(
        http_archive,
        name = "nasm",
        sha256 = "63ec86477ad3f0f6292325fd89e1d93aea2e2fd490070863f17d48f7cd387011",
        strip_prefix = "nasm-2.13.03",
        urls = [
            "https://www.nasm.us/pub/nasm/releasebuilds/2.13.03/nasm-2.13.03.tar.bz2",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/nasm/nasm-2.13.03.tar.bz2",
        ],
        build_file = "//third_party/nasm:nasm.BUILD",
    )

    maybe(
        http_archive,
        name = "sdl2",
        sha256 = "02537cc7ebd74071631038b237ec4bfbb3f4830ba019e569434da33f42373e04",
        strip_prefix = "SDL2-2.26.1",
        urls = [
            "https://www.libsdl.org/release/SDL2-2.26.1.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/libsdl/SDL2-2.26.1.tar.gz",
        ],
        build_file = "//third_party/sdl2:sdl2.BUILD",
    )

    maybe(
        http_archive,
        name = "com_github_nelhage_rules_boost",
        sha256 = "6ded3e8c064054c92b79aeadde2d78821c889598e634c595133da0ea8f0f0b85",
        strip_prefix = "rules_boost-f1065639e6f33741abe2a6a78fa79dd1a07bbf5d",
        urls = [
            "https://github.com/nelhage/rules_boost/archive/f1065639e6f33741abe2a6a78fa79dd1a07bbf5d.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/nelhage/rules_boost/f1065639e6f33741abe2a6a78fa79dd1a07bbf5d.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "boost",
        build_file = "@com_github_nelhage_rules_boost//:BUILD.boost",
        patch_cmds = ["rm -f doc/pdf/BUILD"],
        sha256 = "71feeed900fbccca04a3b4f2f84a7c217186f28a940ed8b7ed4725986baf99fa",
        strip_prefix = "boost_1_81_0",
        urls = [
            "https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.bz2",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/boost/boost_1_81_0.tar.bz2",
        ],
    )

    maybe(
        http_archive,
        name = "freedoom",
        sha256 = "f42c6810fc89b0282de1466c2c9c7c9818031a8d556256a6db1b69f6a77b5806",
        strip_prefix = "freedoom-0.12.1/",
        urls = [
            "https://github.com/freedoom/freedoom/releases/download/v0.12.1/freedoom-0.12.1.zip",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/freedoom/freedoom/freedoom-0.12.1.zip",
        ],
        build_file = "//third_party/freedoom:freedoom.BUILD",
    )

    maybe(
        http_archive,
        name = "vizdoom",
        sha256 = "e379a242ada7e1028b7a635da672b0936d99da3702781b76a4400b83602d78c4",
        strip_prefix = "ViZDoom-1.1.13/src/vizdoom/",
        urls = [
            "https://github.com/mwydmuch/ViZDoom/archive/refs/tags/1.1.13.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/mwydmuch/ViZDoom/1.1.13.tar.gz",
        ],
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
        urls = [
            "https://github.com/mwydmuch/ViZDoom/archive/refs/tags/1.1.13.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/mwydmuch/ViZDoom/1.1.13.tar.gz",
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
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/isl-org/DirectFuturePrediction/b4757769f167f1bd7fb1ece5fdc6d874409c68a9.zip",
        ],
        build_file = "//third_party/vizdoom_extra_maps:vizdoom_extra_maps.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco",
        sha256 = "d1cb3a720546240d894cd315b7fd358a2b96013a1f59b6d718036eca6f6edac2",
        strip_prefix = "mujoco-2.2.1",
        urls = [
            "https://github.com/deepmind/mujoco/releases/download/2.2.1/mujoco-2.2.1-linux-x86_64.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/deepmind/mujoco/mujoco-2.2.1-linux-x86_64.tar.gz",
        ],
        build_file = "//third_party/mujoco:mujoco.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco_gym_xml",
        sha256 = "96a5fc8345bd92b73a15fc25112d53a294f86fcace1c5e4ef7f0e052b5e1bdf4",
        strip_prefix = "gym-0.26.2/gym/envs/mujoco",
        urls = [
            "https://github.com/openai/gym/archive/refs/tags/0.26.2.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/openai/gym/0.26.2.tar.gz",
        ],
        build_file = "//third_party/mujoco_gym_xml:mujoco_gym_xml.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco_dmc_xml",
        sha256 = "fb8d57cbeb92bebe56a992dab8401bc00b3bff61b62526eb563854adf3dfb595",
        strip_prefix = "dm_control-1.0.9/dm_control",
        urls = [
            "https://github.com/deepmind/dm_control/archive/refs/tags/1.0.9.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/deepmind/dm_control/1.0.9.tar.gz",
        ],
        build_file = "//third_party/mujoco_dmc_xml:mujoco_dmc_xml.BUILD",
    )

    maybe(
        http_archive,
        name = "box2d",
        sha256 = "d6b4650ff897ee1ead27cf77a5933ea197cbeef6705638dd181adc2e816b23c2",
        strip_prefix = "box2d-2.4.1",
        urls = [
            "https://github.com/erincatto/box2d/archive/refs/tags/v2.4.1.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/erincatto/box2d/v2.4.1.tar.gz",
        ],
        build_file = "//third_party/box2d:box2d.BUILD",
    )

    # Atari/VizDoom pretrained weight for testing pipeline

    maybe(
        http_archive,
        name = "pretrain_weight",
        sha256 = "b1b64e0db84cf7317c2a96b27f549147dfcb4074ed2d799334c23a067075ac1c",
        urls = [
            "https://cdn.sail.sea.com/sail/pretrain.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/pretrain.tar.gz",
        ],
        build_file = "//third_party/pretrain_weight:pretrain_weight.BUILD",
    )

    maybe(
        http_archive,
        name = "procgen",
        sha256 = "22940ad0f1fdb4ad1eab3303ce23d3a0ea536700bb1d7c299bee64dbc7c57e9b",
        strip_prefix = "procgen-0.10.7/procgen/src",
        urls = [
            "https://github.com/openai/procgen/archive/refs/tags/0.10.7.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/openai/procgen/0.10.7.tar.gz",
        ],
        build_file = "//third_party/procgen:procgen.BUILD",
        patches = [
            "//third_party/procgen:assetgen.patch",
            "//third_party/procgen:qt-utils.patch",
        ],
    )

    maybe(
        http_archive,
        name = "gym3_libenv",
        sha256 = "9a764d79d4215609c2612b2c84fec8bcea6609941bdcb7051f3335ed4576b8ef",
        strip_prefix = "gym3-4c3824680eaf9dd04dce224ee3d4856429878226/gym3",
        urls = [
            "https://github.com/openai/gym3/archive/4c3824680eaf9dd04dce224ee3d4856429878226.zip",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/openai/gym3/4c3824680eaf9dd04dce224ee3d4856429878226.zip",
        ],
        build_file = "//third_party/gym3_libenv:gym3_libenv.BUILD",
    )

    maybe(
        http_archive,
        name = "bazel_clang_tidy",
        sha256 = "ec8c5bf0c02503b928c2e42edbd15f75e306a05b2cae1f34a7bc84724070b98b",
        strip_prefix = "bazel_clang_tidy-783aa523aafb4a6798a538c61e700b6ed27975a7",
        urls = [
            "https://github.com/erenon/bazel_clang_tidy/archive/783aa523aafb4a6798a538c61e700b6ed27975a7.zip",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/erenon/bazel_clang_tidy/783aa523aafb4a6798a538c61e700b6ed27975a7.zip",
        ],
    )

    maybe(
        cuda_configure,
        name = "cuda",
    )

workspace0 = workspace
