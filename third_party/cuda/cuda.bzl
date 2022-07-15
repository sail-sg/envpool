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

"""This is loaded in workspace0.bzl to provide cuda library."""

_CUDA_DIR = "CUDA_DIR"

def _impl(rctx):
    cuda_dir = rctx.os.environ.get(_CUDA_DIR, default = "/usr/local/cuda")
    rctx.symlink("{}/include".format(cuda_dir), "include")
    rctx.symlink("{}/lib64".format(cuda_dir), "lib64")
    rctx.file("WORKSPACE")
    rctx.file("BUILD", content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudart_static",
    srcs = ["lib64/libcudart_static.a"],
    hdrs = glob([
        "include/*.h",
        "include/**/*.h",
    ]),
    strip_include_prefix = "include",
)
""")

cuda_configure = repository_rule(
    implementation = _impl,
    environ = [
        _CUDA_DIR,
    ],
)
