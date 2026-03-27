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
load("//third_party/vizdoom:repo.bzl", "vizdoom_archive")

def workspace():
    """Load requested packages."""

    maybe(
        http_archive,
        name = "rules_shell",
        patches = [
            "//third_party/rules_shell:bazel_sh_environ.patch",
        ],
        sha256 = "410e8ff32e018b9efd2743507e7595c26e2628567c42224411ff533b57d27c28",
        strip_prefix = "rules_shell-0.2.0",
        urls = [
            "https://github.com/bazelbuild/rules_shell/releases/download/v0.2.0/rules_shell-v0.2.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_cc",
        patches = [
            "//third_party/rules_cc:rules_shell_bazel_sh.patch",
        ],
        sha256 = "283fa1cdaaf172337898749cf4b9b1ef5ea269da59540954e51fba0e7b8f277a",
        strip_prefix = "rules_cc-0.2.17",
        urls = [
            "https://github.com/bazelbuild/rules_cc/releases/download/0.2.17/rules_cc-0.2.17.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_java",
        sha256 = "9de4e178c2c4f98d32aafe5194c3f2b717ae10405caa11bdcb460ac2a6f61516",
        urls = [
            "https://github.com/bazelbuild/rules_java/releases/download/9.6.1/rules_java-9.6.1.tar.gz",
        ],
    )

    # Keep both repo names while this WORKSPACE still mixes legacy rules_python
    # and newer rules_cc/rules_java consumers.
    maybe(
        http_archive,
        name = "protobuf",
        sha256 = "9f0933ad12abb0a8957b3b64aefc34774ccbe57f77f890c66cb4067519bb83c1",
        strip_prefix = "protobuf-34.1",
        urls = [
            "https://github.com/protocolbuffers/protobuf/releases/download/v34.1/protobuf-34.1.bazel.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_protobuf",
        sha256 = "9f0933ad12abb0a8957b3b64aefc34774ccbe57f77f890c66cb4067519bb83c1",
        strip_prefix = "protobuf-34.1",
        urls = [
            "https://github.com/protocolbuffers/protobuf/releases/download/v34.1/protobuf-34.1.bazel.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_python",
        patches = [
            "//third_party/rules_python:zipapp_explicit_init.patch",
        ],
        sha256 = "2f5c284fbb4e86045c2632d3573fc006facbca5d1fa02976e89dc0cd5488b590",
        strip_prefix = "rules_python-1.6.3",
        urls = [
            "https://github.com/bazelbuild/rules_python/releases/download/1.6.3/rules_python-1.6.3.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_foreign_cc",
        sha256 = "32759728913c376ba45b0116869b71b68b1c2ebf8f2bcf7b41222bc07b773d73",
        strip_prefix = "rules_foreign_cc-0.15.1",
        urls = [
            "https://github.com/bazel-contrib/rules_foreign_cc/releases/download/0.15.1/rules_foreign_cc-0.15.1.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "bazel_features",
        sha256 = "c26b4e69cf02fea24511a108d158188b9d8174426311aac59ce803a78d107648",
        strip_prefix = "bazel_features-1.43.0",
        urls = [
            "https://github.com/bazel-contrib/bazel_features/releases/download/v1.43.0/bazel_features-v1.43.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        patches = [
            "//third_party/pybind11_bazel:build_defs_rules_cc_defs.patch",
            "//third_party/pybind11_bazel:pybind11_build_rules_cc_defs.patch",
        ],
        sha256 = "4a8983811b562a02ef9ccc545c93dc275e23054ac8996170b8975fa387afa890",
        strip_prefix = "pybind11_bazel-3.0.1",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/pybind/pybind11_bazel/tar.gz/refs/tags/v3.0.1",
        ],
    )

    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
        sha256 = "2f20a0af0b921815e0e169ea7fec63909869323581b89d7de1553468553f6a2d",
        strip_prefix = "pybind11-3.0.2",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/pybind/pybind11/tar.gz/refs/tags/v3.0.2",
        ],
    )

    maybe(
        http_archive,
        name = "openxla_ffi_headers",
        build_file = "//third_party/openxla_ffi:ffi_api.BUILD",
        sha256 = "753df38eab0d430da20e614316401663bcfca433905b976745a6e59998635ce8",
        strip_prefix = "xla-187a5eb58277a85847d1516bd1e20b7faf03d5ef/xla/ffi/api",
        urls = [
            "https://github.com/openxla/xla/archive/187a5eb58277a85847d1516bd1e20b7faf03d5ef.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "05597c3c532197690a31ebad50a7c9c3fb682d3c5a681b20eb03655ffb4e9483",
        strip_prefix = "abseil-cpp-20260107.1",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20260107.1.zip",
        ],
    )

    maybe(
        http_archive,
        name = "com_github_gflags_gflags",
        sha256 = "f619a51371f41c0ad6837b2a98af9d4643b3371015d873887f7e8d3237320b2f",
        strip_prefix = "gflags-2.3.0",
        urls = [
            "https://github.com/gflags/gflags/archive/v2.3.0.tar.gz",
        ],
        patches = [
            "//third_party/gflags:rules_cc_defs.patch",
        ],
    )

    maybe(
        http_archive,
        name = "com_github_google_glog",
        sha256 = "c17d85c03ad9630006ef32c7be7c65656aba2e7e2fbfc82226b7e680c771fc88",
        strip_prefix = "glog-0.7.1",
        urls = [
            "https://github.com/google/glog/archive/v0.7.1.zip",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_googletest",
        sha256 = "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c",
        strip_prefix = "googletest-1.17.0",
        urls = [
            "https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz",
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
        sha256 = "87fbc9884d60d0d4bf3462c18f4c0ee0a9311d0519341cac7cbd361c885e5281",
        strip_prefix = "concurrentqueue-1.0.4",
        urls = [
            "https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.4.tar.gz",
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
        patches = [
            "//third_party/threadpool:invoke_result.patch",
        ],
    )

    maybe(
        http_archive,
        name = "zlib",
        sha256 = "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16",
        strip_prefix = "zlib-1.3.2",
        urls = [
            "https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz",
        ],
        build_file = "//third_party/zlib:zlib.BUILD",
    )

    maybe(
        http_archive,
        name = "opencv",
        patch_args = ["-p1"],
        patches = [
            "//third_party/opencv:windows_msvc_flag_check.patch",
        ],
        sha256 = "1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d",
        strip_prefix = "opencv-4.13.0",
        urls = [
            "https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz",
        ],
        build_file = "//third_party/opencv:opencv.BUILD",
    )

    maybe(
        http_archive,
        name = "pugixml",
        sha256 = "b39647064d9e28297a34278bfb897092bf33b7c487906ddfc094c9e8868bddcb",
        strip_prefix = "pugixml-1.15/src",
        urls = [
            "https://github.com/zeux/pugixml/archive/refs/tags/v1.15.tar.gz",
        ],
        build_file = "//third_party/pugixml:pugixml.BUILD",
    )

    maybe(
        http_archive,
        name = "ale",
        sha256 = "d6ac9406690bb3533b37a99253bdfc59bc27779c5e1b6855c763d0b367bcbf96",
        strip_prefix = "Arcade-Learning-Environment-0.11.2",
        urls = [
            "https://github.com/Farama-Foundation/Arcade-Learning-Environment/archive/refs/tags/v0.11.2.tar.gz",
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
        sha256 = "075920b826834ac4ddf97661cc73491047855859affd671d52079c6867c1c6c0",
        strip_prefix = "libjpeg-turbo-3.1.3",
        urls = [
            "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.3/libjpeg-turbo-3.1.3.tar.gz",
        ],
        build_file = "//third_party/jpeg:jpeg.BUILD",
    )

    maybe(
        http_archive,
        name = "nasm",
        sha256 = "af2f241ecc061205d73ba4f781f075d025dabaeab020b676b7db144bf7015d6d",
        strip_prefix = "nasm-nasm-3.01",
        urls = [
            "https://github.com/netwide-assembler/nasm/archive/refs/tags/nasm-3.01.tar.gz",
        ],
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
        sha256 = "5f5993c530f084535c65a6879e9b26ad441169b3e25d789d83287040a9ca5165",
        strip_prefix = "SDL2-2.32.10",
        urls = [
            "https://www.libsdl.org/release/SDL2-2.32.10.tar.gz",
            "https://github.com/libsdl-org/SDL/releases/download/release-2.32.10/SDL2-2.32.10.tar.gz",
        ],
        build_file = "//third_party/sdl2:sdl2.BUILD",
    )

    maybe(
        http_archive,
        name = "com_github_nelhage_rules_boost",
        # sha256 = "2215e6910eb763a971b1f63f53c45c0f2b7607df38c96287666d94d954da8cdc",
        strip_prefix = "rules_boost-e60cf50996da9fe769b6e7a31b88c54966ecb191",
        urls = [
            "https://github.com/nelhage/rules_boost/archive/e60cf50996da9fe769b6e7a31b88c54966ecb191.tar.gz",
        ],
        patches = [
            "//third_party/boost:filesystem_scope_headers.patch",
            "//third_party/boost:atomic_windows_build.patch",
            "//third_party/boost:lzma_windows_copts.patch",
            "//third_party/boost:windows_common_defines.patch",
            "//third_party/boost:bzip2_system_headers.patch",
            "//third_party/boost:zlib_system_headers.patch",
        ],
    )

    maybe(
        http_archive,
        name = "boost",
        build_file = "@com_github_nelhage_rules_boost//:boost.BUILD",
        patch_cmds = ["rm -f doc/pdf/BUILD"],
        sha256 = "e848446c6fec62d8a96b44ed7352238b3de040b8b9facd4d6963b32f541e00f5",
        strip_prefix = "boost-1.90.0",
        urls = [
            "https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-b2-nodocs.tar.gz",
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
        sha256 = "78029ba5d4276fab14df5ef54dd8b72a3c835ef400db022373b3488999bb6c38",
        strip_prefix = "qhull-62ccc56af071eaa478bef6ed41fd7a55d3bb2d80",
        urls = [
            "https://github.com/qhull/qhull/archive/62ccc56af071eaa478bef6ed41fd7a55d3bb2d80.tar.gz",
        ],
        build_file = "//third_party/qhull:qhull.BUILD",
    )

    maybe(
        http_archive,
        name = "mujoco",
        patch_args = ["-p1"],
        patches = [
            "//third_party/mujoco:windows_msvc_compat.patch",
            "//third_party/mujoco:windows_msvc_c11_compat.patch",
        ],
        sha256 = "0d6bb25612da5d1c398d21cf5b039367d4deef15c064b049c3127043d613b539",
        strip_prefix = "mujoco-3.6.0",
        urls = [
            "https://github.com/google-deepmind/mujoco/archive/refs/tags/3.6.0.tar.gz",
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
        sha256 = "23e86e28ef6ba9d2fec95103d45bd2061cfed35c8b0012b1ac5ee41b080d56c6",
        strip_prefix = "dm_control-1.0.38/dm_control",
        urls = [
            "https://github.com/deepmind/dm_control/archive/refs/tags/1.0.38.tar.gz",
        ],
        build_file = "//third_party/mujoco_dmc_xml:mujoco_dmc_xml.BUILD",
    )

    maybe(
        http_archive,
        name = "box2d",
        sha256 = "85b9b104d256c985e6e244b4227d447897fac429071cc114e5cc819dae848852",
        strip_prefix = "box2d-2.4.2",
        urls = [
            "https://github.com/erincatto/box2d/archive/refs/tags/v2.4.2.tar.gz",
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
        http_archive,
        name = "bazel_clang_tidy",
        sha256 = "fdc45b36544abca36c2fcb85c951915c4e6cb986b2be2977582f18f5b70b99ec",
        strip_prefix = "bazel_clang_tidy-c4d35e0d0b838309358e57a2efed831780f85cd0",
        urls = [
            "https://github.com/erenon/bazel_clang_tidy/archive/c4d35e0d0b838309358e57a2efed831780f85cd0.zip",
        ],
    )

    maybe(
        cuda_configure,
        name = "cuda",
    )

workspace0 = workspace
