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
        sha256 = "9fcf91dbcc31fde6d1edb15f117246d912c33c36f44cf681976bd886538deba6",
        strip_prefix = "rules_python-0.8.0",
        urls = [
            "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.8.0.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/bazelbuild/rules_python/0.8.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_foreign_cc",
        sha256 = "6041f1374ff32ba711564374ad8e007aef77f71561a7ce784123b9b4b88614fc",
        strip_prefix = "rules_foreign_cc-0.8.0",
        urls = [
            "https://github.com/bazelbuild/rules_foreign_cc/archive/0.8.0.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/bazelbuild/rules_foreign_cc/0.8.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        sha256 = "fec6281e4109115c5157ca720b8fe20c8f655f773172290b03f57353c11869c2",
        strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
        urls = [
            "https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.zip",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/pybind/pybind11_bazel/72cbbf1fbc830e487e3012862b7b720001b70672.zip",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1",
        strip_prefix = "pybind11-2.9.2",
        urls = [
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/pybind/pybind11/v2.9.2.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "dcf71b9cba8dc0ca9940c4b316a0c796be8fab42b070bb6b7cab62b48f0e66c4",
        strip_prefix = "abseil-cpp-20211102.0",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/abseil/abseil-cpp/20211102.0.tar.gz",
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
        sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
        strip_prefix = "googletest-release-1.11.0",
        urls = [
            "https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/google/googletest/release-1.11.0.tar.gz",
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
        sha256 = "d8688496ea40fb61787500e863cc63c9afcbc524468cedeb478068924eb54932",
        strip_prefix = "zlib-1.2.12",
        urls = [
            "https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/madler/zlib/v1.2.12.tar.gz",
        ],
        build_file = "//third_party/zlib:zlib.BUILD",
    )

    maybe(
        http_archive,
        name = "opencv",
        urls = [
            "https://github.com/opencv/opencv/archive/refs/tags/4.5.5.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/opencv/opencv/4.5.5.tar.gz",
        ],
        sha256 = "a1cfdcf6619387ca9e232687504da996aaa9f7b5689986b8331ec02cb61d28ad",
        strip_prefix = "opencv-4.5.5",
        build_file = "//third_party/opencv:opencv.BUILD",
    )

    maybe(
        http_archive,
        name = "pugixml",
        urls = [
            "https://github.com/zeux/pugixml/releases/download/v1.12.1/pugixml-1.12.1.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/zeux/pugixml/pugixml-1.12.1.tar.gz",
        ],
        sha256 = "dcf671a919cc4051210f08ffd3edf9e4247f79ad583c61577a13ee93af33afc7",
        strip_prefix = "pugixml-1.12/src",
        build_file = "//third_party/pugixml:pugixml.BUILD",
    )

    maybe(
        http_archive,
        name = "ale",
        sha256 = "31ff9a51187a1237b683c7a5eff232b294f0bd48ca078ab57eabc9e3564ff1c1",
        strip_prefix = "Arcade-Learning-Environment-0.7.5",
        urls = [
            "https://github.com/mgbellemare/Arcade-Learning-Environment/archive/refs/tags/v0.7.5.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/mgbellemare/Arcade-Learning-Environment/v0.7.5.tar.gz",
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
        sha256 = "c56aba1d7b5b0e7e999e4a7698c70b63a3394ff9704b5f6e1c57e0c16f04dd06",
        strip_prefix = "SDL2-2.0.20",
        urls = [
            "https://www.libsdl.org/release/SDL2-2.0.20.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/libsdl/SDL2-2.0.20.tar.gz",
        ],
        build_file = "//third_party/sdl2:sdl2.BUILD",
    )

    maybe(
        http_archive,
        name = "com_github_nelhage_rules_boost",
        sha256 = "ce95d0705592e51eda91b38870e847c303f65219871683f7c34233caad150b0b",
        strip_prefix = "rules_boost-32164a62e2472077320f48f52b8077207cd0c9c8",
        urls = [
            "https://github.com/nelhage/rules_boost/archive/32164a62e2472077320f48f52b8077207cd0c9c8.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/nelhage/rules_boost/32164a62e2472077320f48f52b8077207cd0c9c8.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "boost",
        build_file = "@com_github_nelhage_rules_boost//:BUILD.boost",
        patch_cmds = ["rm -f doc/pdf/BUILD"],
        sha256 = "475d589d51a7f8b3ba2ba4eda022b170e562ca3b760ee922c146b6c65856ef39",
        strip_prefix = "boost_1_79_0",
        urls = [
            "https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.bz2",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/boost/boost_1_79_0.tar.bz2",
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
        sha256 = "7feff9b58b96c0d763429c0670c720d64d7799414cd9a8b70a9eac5b5509a57a",
        strip_prefix = "gym-0.25.1/gym/envs/mujoco",
        urls = [
            "https://github.com/openai/gym/archive/refs/tags/0.25.1.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/openai/gym/0.25.1.tar.gz",
        ],
        build_file = "//third_party/mujoco_gym_xml:mujoco_gym_xml.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco_dmc_xml",
        sha256 = "0ede3050a5deec4b81ed8f42805469e291e622b7b3d4bc6721deed899623dcf9",
        strip_prefix = "dm_control-1.0.5/dm_control",
        urls = [
            "https://github.com/deepmind/dm_control/archive/refs/tags/1.0.5.tar.gz",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/deepmind/dm_control/1.0.5.tar.gz",
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
        name = "bazel_clang_tidy",
        sha256 = "ec8c5bf0c02503b928c2e42edbd15f75e306a05b2cae1f34a7bc84724070b98b",
        strip_prefix = "bazel_clang_tidy-783aa523aafb4a6798a538c61e700b6ed27975a7",
        urls = [
            "https://github.com/erenon/bazel_clang_tidy/archive/783aa523aafb4a6798a538c61e700b6ed27975a7.zip",
            "https://ml.cs.tsinghua.edu.cn/~jiayi/envpool/erenon/bazel_clang_tidy/783aa523aafb4a6798a538c61e700b6ed27975a7.zip",
        ],
    )
    
    maybe(
        http_archive,
        name = "gfootball_engine",
        sha256 = "1b0fdcfa78b7fadc3730585ee7f0f412ba825c27e422b2b85ea0cf7ba57800b6",
        urls = [
            "https://files.pythonhosted.org/packages/98/63/b111538b5db47b8081d8ca82280fadaa145fbd31aa249f49675a01abb8eb/gfootball-2.10.2.tar.gz"
        ],
        build_file = "//third_party/football:football.BUILD",
    )

    maybe(
        cuda_configure,
        name = "cuda",
    )

workspace0 = workspace
