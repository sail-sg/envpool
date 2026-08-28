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

"""Official native maze generator and pinned CMU locomotion trajectories."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def dmc_locomotion_repositories():
    maybe(
        http_archive,
        name = "labmaze_source",
        urls = ["https://files.pythonhosted.org/packages/93/0a/139c4ae896b9413bd4ca69c62b08ee98dcfc78a9cbfdb7cadd0dce2ad31d/labmaze-1.0.6.tar.gz"],
        sha256 = "2e8de7094042a77d6972f1965cf5c9e8f971f1b34d225752f343190a825ebe73",
        strip_prefix = "labmaze-1.0.6",
        build_file = "//third_party/dmc_locomotion:labmaze.BUILD",
        patches = ["//third_party/dmc_locomotion:labmaze.patch"],
        patch_args = ["-p1"],
    )
    for version, filename, checksum in [
        ("2019", "cmu_2019_08756c01.h5", "08756c01cb4ac20da9918e70e85c32d4880c6c8c16189b02a18b79a5e79afa2b"),
        ("2020", "cmu_2020_dfe3e9e0.h5", "dfe3e9e0b08d32960bdafbf89e541339ca8908a9a5e7f4a2c986362890d72863"),
    ]:
        maybe(
            http_file,
            name = "dmc_cmu_mocap_" + version,
            urls = ["https://storage.googleapis.com/dm_control/" + filename],
            sha256 = checksum,
            downloaded_file_path = filename,
        )
