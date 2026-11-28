# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gymnax and the original IDX files used by MNISTBandit."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# Gymnax 1.1.0
_COMMIT = "b88dd87ca3917f1723d14016f0b1c9719e893d77"

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}

def _mnist_impl(ctx):
    for name, sha in MNIST_FILES.items():
        ctx.download(
            url = [
                "https://storage.googleapis.com/cvdf-datasets/mnist/" + name,
                "https://ossci-datasets.s3.amazonaws.com/mnist/" + name,
            ],
            output = name,
            sha256 = sha,
        )
    ctx.file("BUILD.bazel", 'exports_files(glob(["*.gz"]))')

_mnist = repository_rule(implementation = _mnist_impl)

def gymnax_repositories():
    maybe(
        http_archive,
        name = "gymnax_upstream",
        type = "tar.gz",
        urls = ["https://codeload.github.com/RobertTLange/gymnax/tar.gz/" + _COMMIT],
        sha256 = "f9642fa8cf707eda2e7183bde02c21931407fc0e2b3ed32eefaba56698d0b9dc",
        strip_prefix = "gymnax-" + _COMMIT,
        build_file = "//third_party/gymnax:gymnax.BUILD",
    )
    maybe(_mnist, name = "mnist")
