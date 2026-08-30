# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned MJLab oracle, native Warp core, and PyTorch's matching math library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(":mkl.bzl", "mjlab_mkl_repository")

def mjlab_repositories():
    maybe(mjlab_mkl_repository, name = "mjlab_mkl")
    for name, repo, commit, checksum, build, patches in [
        (
            "mjlab_source",
            "mujocolab/mjlab",
            "b517e0c489139e7fcee95702cfb2b01931264985",
            "fc0c7e9d31877b3d921b1f97a5ff0c9a55b3e1d8621e80043f4749048d138c42",
            "mjlab.BUILD",
            [],
        ),
        (
            "mjlab_warp",
            "NVIDIA/warp",
            "b943176fbe3ab70e90708e74d2c48e6f50557145",
            "2ae42982c08f32ea3a2797bd13fb192b87b3de4396754caafe3c28073f7f0e08",
            "warp.BUILD",
            ["//third_party/mjlab:warp_native.patch"],
        ),
        (
            "mjlab_sleef",
            "shibatch/sleef",
            "5a1d179df9cf652951b59010a2d2075372d67f68",
            "afd1b92010ae7918e20eec3e5bb270b7ca4828b5cb19e4d820a0a4558a04ce63",
            "sleef.BUILD",
            [],
        ),
    ]:
        maybe(
            http_archive,
            name = name,
            urls = ["https://codeload.github.com/" + repo + "/tar.gz/" + commit],
            sha256 = checksum,
            type = "tar.gz",
            strip_prefix = repo.split("/")[-1] + "-" + commit,
            build_file = "//third_party/mjlab:" + build,
            patches = patches,
            patch_args = ["-p1"],
        )
