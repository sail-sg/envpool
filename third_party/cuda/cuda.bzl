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

_STUB_CUDA_RUNTIME_API_H = """
#ifndef CUDA_RUNTIME_API_H_
#define CUDA_RUNTIME_API_H_

#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cudaStream_t;
typedef int cudaError_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
};

static inline cudaError_t cudaMemcpyAsync(
    void* dst, const void* src, size_t count, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    (void)dst;
    (void)src;
    (void)count;
    (void)kind;
    (void)stream;
    abort();
    return 1;
}

static inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    abort();
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif  /* CUDA_RUNTIME_API_H_ */
"""

def _impl(rctx):
    cuda_dir = rctx.os.environ.get(_CUDA_DIR, default = "/usr/local/cuda")
    cuda_include = rctx.path("{}/include/cuda_runtime_api.h".format(cuda_dir))
    cudart_static = rctx.path("{}/lib64/libcudart_static.a".format(cuda_dir))
    rctx.file("WORKSPACE")
    if cuda_include.exists and cudart_static.exists:
        rctx.symlink("{}/include".format(cuda_dir), "include")
        rctx.symlink("{}/lib64".format(cuda_dir), "lib64")
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
    else:
        rctx.file("include/cuda_runtime_api.h", _STUB_CUDA_RUNTIME_API_H)
        rctx.file("BUILD", content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudart_static",
    hdrs = ["include/cuda_runtime_api.h"],
    strip_include_prefix = "include",
)
""")

cuda_configure = repository_rule(
    implementation = _impl,
    environ = [
        _CUDA_DIR,
    ],
)
