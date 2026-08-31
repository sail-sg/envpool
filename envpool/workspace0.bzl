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

"""Pinned native repositories for EnvPool's Bzlmod extension."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party/craftax:repo.bzl", "craftax_archive")
load("//third_party/cuda:cuda.bzl", "cuda_configure")
load("//third_party/dmc_locomotion:repositories.bzl", "dmc_locomotion_repositories")
load("//third_party/freedoom:defs.bzl", "freedoom_archive")
load("//third_party/gfootball:repo.bzl", "gfootball_archive")
load("//third_party/mjlab:repositories.bzl", "mjlab_repositories")
load("//third_party/qt:qt_configure.bzl", "qt_configure")
load("//third_party/vizdoom:repo.bzl", "vizdoom_archive")

def workspace():
    """Load requested packages."""

    craftax_archive()
    dmc_locomotion_repositories()
    mjlab_repositories()

    maybe(
        http_file,
        name = "jumanji_pacman_constants",
        downloaded_file_path = "constants.py",
        sha256 = "04430c2a20edaa573639fe58ffe9c515f9b930dfc9bf7ca6f06315ecf09f0ec4",
        urls = [
            "https://raw.githubusercontent.com/instadeepai/jumanji/0584fdc4ddb3f616e28546f7aaf65f1dd59aeb48/jumanji/environments/routing/pac_man/constants.py",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        patches = [
            "//third_party/pybind11_bazel:build_defs_rules_cc_defs.patch",
            "//third_party/pybind11_bazel:pybind11_build_rules_cc_defs.patch",
        ],
        sha256 = "f11ba4e4b409e60088493b5429686c7ab8f67e936a06eb0e53dc5c121ebf8613",
        strip_prefix = "pybind11_bazel-3.0.3",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/pybind/pybind11_bazel/tar.gz/refs/tags/v3.0.3",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
        sha256 = "ef712655692a2e9bf7bb7874c022564a45f91d847ddee987e720cd9e28849665",
        strip_prefix = "pybind11-3.1.0",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/pybind/pybind11/tar.gz/refs/tags/v3.1.0",
        ],
    )

    maybe(
        http_archive,
        name = "openxla_ffi_headers",
        build_file = "//third_party/openxla_ffi:ffi_api.BUILD",
        sha256 = "4c89ecfff5a662a6edfb4e2d403fedf55be40a3a0079e3c2f8ba47b37c16eaab",
        strip_prefix = "xla-dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/ffi/api",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/openxla/xla/tar.gz/dcf304bc5dca1932b99f740b911dbd73631a1a69",
            "https://github.com/openxla/xla/archive/dcf304bc5dca1932b99f740b911dbd73631a1a69.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "f7e05179df39c45434cad433f5783840bb3788ef322976f9138bc6b72b3a107d",
        strip_prefix = "abseil-cpp-20260817.0",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20260817.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_googletest",
        sha256 = "6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5",
        strip_prefix = "googletest-1.18.0",
        urls = [
            "https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "concurrentqueue",
        sha256 = "4d6368a27492d86011fde5ca0cf386dce7c49cd425aa3d9b063ca6ec373a6ef3",
        strip_prefix = "concurrentqueue-1.0.5",
        urls = [
            "https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.5.tar.gz",
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
            "https://codeload.github.com/progschj/ThreadPool/zip/9a42ec1329f259a5f4881a291db1dcb8f2ad9040",
        ],
        build_file = "//third_party/threadpool:threadpool.BUILD",
        patches = [
            "//third_party/threadpool:invoke_result.patch",
        ],
    )

    maybe(
        http_archive,
        name = "opencv",
        patch_args = ["-p1"],
        patches = [
            "//third_party/opencv:windows_msvc_flag_check.patch",
            "//third_party/opencv:windows_cpu_baseline_flags.patch",
        ],
        sha256 = "b0528f5a1d379d59d4701cb28c36e22214cc51cf64594e5b56f2d3e6c0233095",
        strip_prefix = "opencv-5.0.0",
        urls = [
            "https://github.com/opencv/opencv/archive/refs/tags/5.0.0.tar.gz",
        ],
        build_file = "//third_party/opencv:opencv.BUILD",
    )

    maybe(
        http_archive,
        name = "pugixml",
        sha256 = "357bcab8877dc9943f355d3a72daba1b053238ba955f50fa81586afb65090219",
        strip_prefix = "pugixml-1.16/src",
        urls = [
            "https://github.com/zeux/pugixml/archive/refs/tags/v1.16.tar.gz",
        ],
        build_file = "//third_party/pugixml:pugixml.BUILD",
    )

    maybe(
        http_archive,
        name = "ale",
        sha256 = "ade05f76416b4a49e8d6e5cc9bebb0745ae69f813aaeabe5813043f288db8ab3",
        strip_prefix = "Arcade-Learning-Environment-0.12.1",
        urls = [
            "https://github.com/Farama-Foundation/Arcade-Learning-Environment/archive/refs/tags/v0.12.1.tar.gz",
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
        sha256 = "6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e",
        strip_prefix = "libjpeg-turbo-3.2.0",
        urls = [
            "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.0/libjpeg-turbo-3.2.0.tar.gz",
        ],
        build_file = "//third_party/jpeg:jpeg.BUILD",
    )

    maybe(
        http_archive,
        name = "nasm",
        sha256 = "39e251d3048c9f68678903c6b05b83942c66c71e467e8c5c3c1b26cff2ef1586",
        strip_prefix = "nasm-nasm-3.02",
        urls = [
            "https://github.com/netwide-assembler/nasm/archive/refs/tags/nasm-3.02.tar.gz",
        ],
        patches = ["//third_party/nasm:windows_sdk_headers.patch"],
        patch_cmds = ["""
set -eux
perl -Iperllib -I. x86/preinsns.pl x86/insns.dat x86/insns.xda
perl -Iperllib -I. x86/insns.pl -fc x86/insns.xda x86/iflag.c
perl -Iperllib -I. x86/insns.pl -fh x86/insns.xda x86/iflaggen.h
perl -Iperllib -I. x86/insns.pl -b x86/insns.xda x86/insnsb.c
perl -Iperllib -I. x86/insns.pl -a x86/insns.xda x86/insnsa.c
perl -Iperllib -I. x86/insns.pl -d x86/insns.xda x86/insnsd.c
perl -Iperllib -I. x86/insns.pl -i x86/insns.xda x86/insnsi.h
perl -Iperllib -I. x86/insns.pl -n x86/insns.xda x86/insnsn.c
perl -Iperllib -I. version.pl h < version > version.h
perl -Iperllib -I. version.pl mac < version > version.mac
perl -Iperllib -I. x86/regs.pl c x86/regs.dat > x86/regs.c
perl -Iperllib -I. x86/regs.pl fc x86/regs.dat > x86/regflags.c
perl -Iperllib -I. x86/regs.pl dc x86/regs.dat > x86/regdis.c
perl -Iperllib -I. x86/regs.pl dh x86/regs.dat > x86/regdis.h
perl -Iperllib -I. x86/regs.pl vc x86/regs.dat > x86/regvals.c
perl -Iperllib -I. x86/regs.pl h x86/regs.dat > x86/regs.h
perl -Iperllib -I. asm/tokhash.pl c x86/insnsn.c x86/regs.dat asm/tokens.dat > asm/tokhash.c
perl -Iperllib -I. asm/tokhash.pl h x86/insnsn.c x86/regs.dat asm/tokens.dat > asm/tokens.h
perl -Iperllib -I. asm/pptok.pl h asm/pptok.dat asm/pptok.h
perl -Iperllib -I. asm/pptok.pl c asm/pptok.dat asm/pptok.c
perl -Iperllib -I. asm/pptok.pl ph asm/pptok.dat asm/pptok.ph
perl -Iperllib -I. nasmlib/perfhash.pl h asm/directiv.dat asm/directiv.h
perl -Iperllib -I. nasmlib/perfhash.pl c asm/directiv.dat asm/directbl.c
perl -Iperllib -I. asm/warnings.pl c asm/warnings_c.h asm/warnings.dat
perl -Iperllib -I. asm/warnings.pl h include/warnings.h asm/warnings.dat
perl -Iperllib -I. macros/macros.pl version.mac 'macros/*.mac' 'output/*.mac'
"""],
        build_file = "//third_party/nasm:nasm.BUILD",
    )

    maybe(
        http_archive,
        name = "sdl2",
        patch_args = ["-p1"],
        patches = ["//third_party/sdl2:static_sdl3_compat.patch"],
        sha256 = "998fa62557eb46ffe7e5c3e2c123bc332f7df9d9f593b3ceed88ed1158428a44",
        strip_prefix = "sdl2-compat-2.32.70",
        urls = [
            "https://github.com/libsdl-org/sdl2-compat/releases/download/release-2.32.70/sdl2-compat-2.32.70.tar.gz",
        ],
        build_file = "//third_party/sdl2:sdl2.BUILD",
    )

    maybe(
        http_archive,
        name = "sdl3",
        patch_args = ["-p1"],
        patches = ["//third_party/sdl2:static_sdl3_namespace.patch"],
        sha256 = "30d4aa2b3037718142b32dffd4e72f917ebb6cc5227150e7bb9c45efb2153aeb",
        strip_prefix = "SDL3-3.4.14",
        urls = [
            "https://github.com/libsdl-org/SDL/releases/download/release-3.4.14/SDL3-3.4.14.tar.gz",
        ],
        build_file = "//third_party/sdl2:sdl3.BUILD",
    )

    maybe(
        http_archive,
        name = "freetype",
        sha256 = "dc49de6b01a266eef4876a4dd34d9842c475d3e28ff2eff63bd2fb760ab56261",
        strip_prefix = "freetype-VER-2-14-3",
        type = "tar.gz",
        urls = [
            "https://github.com/freetype/freetype/archive/refs/tags/VER-2-14-3.tar.gz",
            "https://codeload.github.com/freetype/freetype/tar.gz/refs/tags/VER-2-14-3",
        ],
        build_file = "//third_party/freetype:freetype.BUILD",
    )

    maybe(
        http_archive,
        name = "sdl2_ttf",
        sha256 = "2c45241a56203a59d66ec6b4eae9457e5675fc609376566a257391fd29d341a2",
        strip_prefix = "SDL_ttf-release-2.24.0",
        type = "tar.gz",
        urls = [
            "https://github.com/libsdl-org/SDL_ttf/archive/refs/tags/release-2.24.0.tar.gz",
        ],
        build_file = "//third_party/sdl2_ttf:sdl2_ttf.BUILD",
    )

    maybe(
        http_archive,
        name = "sdl2_gfx",
        sha256 = "358042f1e63ba1ef4d6484047998ab08c1ee72ab589826805bfbba8fd50abe8b",
        strip_prefix = "SDL2_gfx-c4aca6b9700ec0db0abd316809e7e6038c511ce2",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/ferzkopp/SDL2_gfx/tar.gz/c4aca6b9700ec0db0abd316809e7e6038c511ce2",
        ],
        build_file = "//third_party/sdl2_gfx:sdl2_gfx.BUILD",
    )

    maybe(
        http_archive,
        name = "boost",
        build_file = "@com_github_nelhage_rules_boost//:boost.BUILD",
        patch_cmds = ["rm -f doc/pdf/BUILD"],
        sha256 = "f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984",
        strip_prefix = "boost-1.92.0",
        urls = [
            "https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-cmake.tar.gz",
        ],
    )

    maybe(
        freedoom_archive,
        name = "freedoom",
        attempts = 8,
        build_file = "//third_party/freedoom:freedoom.BUILD",
        sha256 = "3f9b264f3e3ce503b4fb7f6bdcb1f419d93c7b546f4df3e874dd878db9688f59",
        strip_prefix = "freedoom-0.13.0/",
        type = "zip",
        urls = [
            "https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip",
        ],
    )

    maybe(
        http_archive,
        name = "re2c_4_5_1",
        build_file = "//third_party/re2c:re2c.BUILD",
        sha256 = "ffea067c11aa668bcb42885be6e6cd000302000b7747d2bb213299ec66b7864e",
        strip_prefix = "re2c-4.5.1",
        urls = [
            "https://github.com/skvadrik/re2c/releases/download/4.5.1/re2c-4.5.1.tar.xz",
        ],
    )

    maybe(
        vizdoom_archive,
        name = "vizdoom",
        patch_args = [
            "-p0",
            "-l",
        ],
        patch_tool = "patch",
        sha256 = "76ddf186d7f093ef85cbcb0e7e387757d60e45190eb5da6d075aab31ffc316ed",
        strip_prefix = "ViZDoom-1.3.0/src/vizdoom/",
        urls = [
            "https://github.com/Farama-Foundation/ViZDoom/archive/refs/tags/1.3.0.tar.gz",
        ],
        build_file = "//third_party/vizdoom:vizdoom.BUILD",
        patches = [
            "//third_party/vizdoom:sdl_thread.patch",
            "//third_party/vizdoom:windows_msvc_compat.patch",
            "//third_party/vizdoom:concurrent_runtime_directory.patch",
            "//third_party/vizdoom:macos_registration_sections.patch",
        ],
    )

    maybe(
        http_archive,
        name = "vizdoom_lib",
        patch_args = [
            "-p0",
            "-l",
        ],
        patches = [
            "//third_party/vizdoom_lib:windows_create_process.patch",
            "//third_party/vizdoom_lib:unique_instance_ids.patch",
            "//third_party/vizdoom_lib:failed_init_cleanup.patch",
        ],
        sha256 = "76ddf186d7f093ef85cbcb0e7e387757d60e45190eb5da6d075aab31ffc316ed",
        strip_prefix = "ViZDoom-1.3.0/",
        urls = [
            "https://github.com/Farama-Foundation/ViZDoom/archive/refs/tags/1.3.0.tar.gz",
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

    maybe(
        http_archive,
        name = "tinyxml2",
        sha256 = "ab1a6700074ab4d468e46535545bb33aa4a74d794ab514fac64cc297fc7a2545",
        strip_prefix = "tinyxml2-e6caeae85799003f4ca74ff26ee16a789bc2af48",
        urls = [
            "https://github.com/leethomason/tinyxml2/archive/e6caeae85799003f4ca74ff26ee16a789bc2af48.tar.gz",
        ],
        build_file = "//third_party/tinyxml2:tinyxml2.BUILD",
    )

    maybe(
        http_archive,
        name = "lodepng",
        sha256 = "83d828c5478ffe7bad0e8ed80678ef826206becbd8cf70097b6cc4d29549389b",
        strip_prefix = "lodepng-17d08dd26cac4d63f43af217ebd70318bfb8189c",
        urls = [
            "https://github.com/lvandeve/lodepng/archive/17d08dd26cac4d63f43af217ebd70318bfb8189c.tar.gz",
        ],
        build_file = "//third_party/lodepng:lodepng.BUILD",
    )

    maybe(
        http_archive,
        name = "tinyobjloader",
        sha256 = "e334b2900380efdc19a0ea42e5e966a6a6a04831dd830dd42a80e28ce6d1e9be",
        strip_prefix = "tinyobjloader-1421a10d6ed9742f5b2c1766d22faa6cfbc56248",
        urls = [
            "https://github.com/tinyobjloader/tinyobjloader/archive/1421a10d6ed9742f5b2c1766d22faa6cfbc56248.tar.gz",
        ],
        build_file = "//third_party/tinyobjloader:tinyobjloader.BUILD",
    )

    maybe(
        http_archive,
        name = "marchingcubecpp",
        sha256 = "227c10b2cffe886454b92a0e9ef9f0c9e8e001d00ea156cc37c8fc43055c9ca6",
        strip_prefix = "MarchingCubeCpp-f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d",
        urls = [
            "https://github.com/aparis69/MarchingCubeCpp/archive/f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d.tar.gz",
        ],
        build_file = "//third_party/marchingcubecpp:marchingcubecpp.BUILD",
    )

    maybe(
        http_archive,
        name = "ccd",
        sha256 = "479994a86d32e2effcaad64204142000ee6b6b291fd1859ac6710aee8d00a482",
        strip_prefix = "libccd-7931e764a19ef6b21b443376c699bbc9c6d4fba8",
        urls = [
            "https://github.com/danfis/libccd/archive/7931e764a19ef6b21b443376c699bbc9c6d4fba8.tar.gz",
        ],
        build_file = "//third_party/ccd:ccd.BUILD",
    )

    maybe(
        http_archive,
        name = "qhull",
        sha256 = "421177cc21a7dcb4c1bbc51f65bc16f21d1f157814116bb5c341d694e23d154d",
        strip_prefix = "qhull-d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6",
        urls = [
            "https://github.com/qhull/qhull/archive/d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6.tar.gz",
        ],
        build_file = "//third_party/qhull:qhull.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco",
        patch_args = ["-p1"],
        patches = [
            "//third_party/mujoco:idempotent_obj_decoder.patch",
            "//third_party/mujoco:idempotent_stl_decoder.patch",
            "//third_party/mujoco:windows_msvc_compat.patch",
            "//third_party/mujoco:windows_msvc_c11_compat.patch",
        ],
        sha256 = "f6346a0bab22bc0db5cabfe299fd3819b8b9bab67907c1ec0d243c675635ea3d",
        strip_prefix = "mujoco-3.11.0",
        urls = [
            "https://github.com/google-deepmind/mujoco/archive/refs/tags/3.11.0.tar.gz",
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
        ],
        build_file = "//third_party/mujoco_gym_xml:mujoco_gym_xml.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco_dmc_xml",
        sha256 = "2989aedd24a28966f472b8882376f82b08290350527724b5591ae38d5937aee7",
        strip_prefix = "dm_control-1.0.44/dm_control",
        urls = [
            "https://github.com/google-deepmind/dm_control/archive/refs/tags/1.0.44.tar.gz",
            "https://codeload.github.com/google-deepmind/dm_control/tar.gz/refs/tags/1.0.44",
        ],
        build_file = "//third_party/mujoco_dmc_xml:mujoco_dmc_xml.BUILD",
    )

    maybe(
        http_archive,
        name = "gymnasium_robotics_assets",
        sha256 = "e5f49da77b31c3f7be02eea7f9788d7fd817d4eccdcef625b1a85a77e221532f",
        strip_prefix = "Gymnasium-Robotics-1.4.2/gymnasium_robotics/envs",
        urls = [
            "https://github.com/Farama-Foundation/Gymnasium-Robotics/archive/refs/tags/v1.4.2.tar.gz",
            "https://codeload.github.com/Farama-Foundation/Gymnasium-Robotics/tar.gz/refs/tags/v1.4.2",
        ],
        build_file = "//third_party/gymnasium_robotics_assets:gymnasium_robotics_assets.BUILD",
    )

    maybe(
        http_archive,
        name = "metaworld_assets",
        sha256 = "fbcfbb07eacec784f32c0efa5dcaa0ee361a39ada8780298aedde3ad7ef40417",
        strip_prefix = "Metaworld-3.1.1",
        urls = [
            "https://github.com/Farama-Foundation/Metaworld/archive/refs/tags/v3.1.1.tar.gz",
            "https://codeload.github.com/Farama-Foundation/Metaworld/tar.gz/refs/tags/v3.1.1",
        ],
        build_file = "//third_party/metaworld_assets:metaworld_assets.BUILD",
    )

    maybe(
        http_archive,
        name = "myosuite_source",
        patch_args = ["-p1"],
        patches = ["//third_party/myosuite:numpy2_scalars.patch"],
        sha256 = "9fc2c610c5d71d2640cc75a0ea989c4f864622b49f67d38bfe7f1f6623396257",
        strip_prefix = "myosuite-2.12.2",
        urls = [
            "https://github.com/MyoHub/myosuite/archive/refs/tags/v2.12.2.tar.gz",
            "https://codeload.github.com/MyoHub/myosuite/tar.gz/refs/tags/v2.12.2",
        ],
        build_file = "//third_party/myosuite:myosuite_source.BUILD",
    )

    maybe(
        http_archive,
        name = "myosuite_myo_sim",
        # MyoSuite 2.12.2 still includes the legacy hand/elbow XML fragments.
        # myo_sim 0.2 replaces them with MjSpec composition, so keep its gitlink.
        sha256 = "bd8fdf313b46dbefcd25bf42cf8ddcc45066798164bb3551a990690cad514ebd",
        strip_prefix = "myo_sim-33f3ded946f55adbdcf963c99999587aadaf975f",
        urls = [
            "https://github.com/MyoHub/myo_sim/archive/33f3ded946f55adbdcf963c99999587aadaf975f.tar.gz",
            "https://codeload.github.com/MyoHub/myo_sim/tar.gz/33f3ded946f55adbdcf963c99999587aadaf975f",
        ],
        build_file = "//third_party/myosuite:simhive_source.BUILD",
    )

    maybe(
        http_archive,
        name = "myosuite_object_sim",
        sha256 = "70ec63c83dc11d7c9f597b91daec5b40f94d6c09ccd2127b61b7efec99d2ca5b",
        strip_prefix = "object_sim-0.1.1",
        urls = [
            # MyoSuite v2.12.2 gitlinks vikashplus/object_sim@87cd8dd, but
            # that commit is no longer fetchable from GitHub archives.
            "https://github.com/MyoHub/object_sim/archive/refs/tags/v0.1.1.tar.gz",
            "https://codeload.github.com/MyoHub/object_sim/tar.gz/refs/tags/v0.1.1",
        ],
        build_file = "//third_party/myosuite:simhive_source.BUILD",
    )

    maybe(
        http_archive,
        name = "myosuite_mpl_sim",
        sha256 = "591fce117832c789e227499ea45c601a9ca142c7dd636492f8bbcd825d54ea0a",
        strip_prefix = "MPL_sim-58dd1abc6058e0dc06e62f13a61c36adb4916815",
        urls = [
            "https://github.com/vikashplus/MPL_sim/archive/58dd1abc6058e0dc06e62f13a61c36adb4916815.tar.gz",
            "https://codeload.github.com/vikashplus/MPL_sim/tar.gz/58dd1abc6058e0dc06e62f13a61c36adb4916815",
        ],
        build_file = "//third_party/myosuite:simhive_source.BUILD",
    )

    maybe(
        http_archive,
        name = "myosuite_ycb_sim",
        sha256 = "200ea58c4d4add1eabf68ee735c88a6cb8503d518b9c3bc6b0ce3ad7ee845ccf",
        strip_prefix = "YCB_sim-57546b87f4724c947eadd4241a7892473febb88d",
        urls = [
            "https://github.com/vikashplus/YCB_sim/archive/57546b87f4724c947eadd4241a7892473febb88d.tar.gz",
            "https://codeload.github.com/vikashplus/YCB_sim/tar.gz/57546b87f4724c947eadd4241a7892473febb88d",
        ],
        build_file = "//third_party/myosuite:simhive_source.BUILD",
    )

    maybe(
        http_archive,
        name = "myosuite_furniture_sim",
        sha256 = "5fb42ed8c932f7c820a72fbb86ea736957476020bdf008e17277380c3693ce9e",
        strip_prefix = "furniture_sim-c97995afb81c9e2d7325b0069f9abc9a2c74a2f0",
        urls = [
            "https://github.com/vikashplus/furniture_sim/archive/c97995afb81c9e2d7325b0069f9abc9a2c74a2f0.tar.gz",
            "https://codeload.github.com/vikashplus/furniture_sim/tar.gz/c97995afb81c9e2d7325b0069f9abc9a2c74a2f0",
        ],
        build_file = "//third_party/myosuite:simhive_source.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco_playground_source",
        patch_args = ["-p1"],
        patches = [
            "//third_party/mujoco_playground:aero_hand_asset_paths.patch",
            "//third_party/mujoco_playground:aloha_asset_paths.patch",
            "//third_party/mujoco_playground:apollo_mesh_paths.patch",
            "//third_party/mujoco_playground:go1_mesh_paths.patch",
            "//third_party/mujoco_playground:h1_mesh_paths.patch",
            "//third_party/mujoco_playground:jax_clip.patch",
            "//third_party/mujoco_playground:op3_mesh_paths.patch",
            "//third_party/mujoco_playground:oracle_imports.patch",
            "//third_party/mujoco_playground:panda_include_paths.patch",
            "//third_party/mujoco_playground:panda_robotiq_asset_paths.patch",
            "//third_party/mujoco_playground:t1_mesh_paths.patch",
        ],
        sha256 = "6348dee222e52318098dc425ff9708af46e0b6d4d0a17c44336a1e4f53c90f04",
        strip_prefix = "mujoco_playground-0.2.0",
        urls = [
            "https://github.com/google-deepmind/mujoco_playground/archive/refs/tags/v0.2.0.tar.gz",
            "https://codeload.github.com/google-deepmind/mujoco_playground/tar.gz/refs/tags/v0.2.0",
        ],
        build_file = "//third_party/mujoco_playground:mujoco_playground_source.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco_menagerie_playground",
        patch_args = ["-p1"],
        patches = [
            "//third_party/mujoco_playground:panda_menagerie_mesh_paths.patch",
        ],
        sha256 = "b03591082fc46b4334d79785b6cc24864c6d6cda8be6947b87a0f4f5e731d7f5",
        strip_prefix = "mujoco_menagerie-da76818e269b82289eba39808e2fb91d679d6994",
        urls = [
            "https://github.com/google-deepmind/mujoco_menagerie/archive/da76818e269b82289eba39808e2fb91d679d6994.tar.gz",
            "https://codeload.github.com/google-deepmind/mujoco_menagerie/tar.gz/da76818e269b82289eba39808e2fb91d679d6994",
        ],
        build_file = "//third_party/mujoco_playground:mujoco_menagerie.BUILD",
    )

    maybe(
        http_archive,
        name = "box2d",
        sha256 = "5471722f290b7285dcbdee9bef61d1cb424e5a610fa6e19e9ddeb854c7e3b937",
        strip_prefix = "pybox2d-2.3.10",
        urls = [
            "https://github.com/pybox2d/pybox2d/archive/refs/tags/2.3.10.tar.gz",
        ],
        build_file = "//third_party/box2d:box2d.BUILD",
        patch_cmds = [
            "sed -i.bak 's/^#define USE_EXCEPTIONS$/\\/\\/ #define USE_EXCEPTIONS/' Box2D/Common/b2Settings.h",
        ],
    )

    # Atari/VizDoom pretrained weight for testing pipeline

    maybe(
        http_archive,
        name = "pretrain_weight",
        sha256 = "b1b64e0db84cf7317c2a96b27f549147dfcb4074ed2d799334c23a067075ac1c",
        urls = [
            "https://cdn.sail.sea.com/sail/pretrain.tar.gz",
        ],
        build_file = "//third_party/pretrain_weight:pretrain_weight.BUILD",
    )

    maybe(
        http_archive,
        name = "procgen",
        sha256 = "22940ad0f1fdb4ad1eab3303ce23d3a0ea536700bb1d7c299bee64dbc7c57e9b",
        strip_prefix = "procgen-0.10.7/procgen",
        urls = [
            "https://github.com/openai/procgen/archive/refs/tags/0.10.7.tar.gz",
        ],
        build_file = "//third_party/procgen:procgen.BUILD",
        patches = [
            "//third_party/procgen:envpool.patch",
        ],
    )

    maybe(
        http_archive,
        name = "gym3_libenv",
        sha256 = "9a764d79d4215609c2612b2c84fec8bcea6609941bdcb7051f3335ed4576b8ef",
        strip_prefix = "gym3-4c3824680eaf9dd04dce224ee3d4856429878226/gym3",
        urls = [
            "https://github.com/openai/gym3/archive/4c3824680eaf9dd04dce224ee3d4856429878226.zip",
        ],
        build_file = "//third_party/gym3_libenv:gym3_libenv.BUILD",
    )

    maybe(
        gfootball_archive,
        name = "google_research_football",
        sha256 = "458100b893aaa530fa269a8ac17484e6f05812e266c81712834dfefc7ecd196b",
        strip_prefix = "football-3d9e754720a95621bba6475c4d3b0d56fe919014",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/google-research/football/tar.gz/3d9e754720a95621bba6475c4d3b0d56fe919014",
        ],
        build_file = "//third_party/gfootball:gfootball.BUILD",
        patch_args = ["-p1"],
        patches = ["//third_party/gfootball:envpool.patch"],
    )

    maybe(
        http_archive,
        name = "bazel_clang_tidy",
        sha256 = "fdc45b36544abca36c2fcb85c951915c4e6cb986b2be2977582f18f5b70b99ec",
        strip_prefix = "bazel_clang_tidy-c4d35e0d0b838309358e57a2efed831780f85cd0",
        urls = [
            "https://github.com/erenon/bazel_clang_tidy/archive/c4d35e0d0b838309358e57a2efed831780f85cd0.zip",
        ],
    )

    maybe(
        http_archive,
        name = "marlgrid",
        sha256 = "8871232e3abf0946dd7181dd4c332a521fb38a6b33bfb730d575c879990ff8cc",
        strip_prefix = "marlgrid-e88c40bad07653575ac11fe2f3a115e4de3d13e9",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/kandouss/marlgrid/tar.gz/e88c40bad07653575ac11fe2f3a115e4de3d13e9",
        ],
        build_file = "//third_party/marlgrid:marlgrid.BUILD",
    )

    maybe(
        cuda_configure,
        name = "cuda",
    )

    qt_configure(name = "qt")

def _repositories_impl(_module_ctx):
    workspace()

repositories = module_extension(implementation = _repositories_impl)
