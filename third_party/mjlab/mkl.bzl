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

"""The official Torch 2.9 CPU wheels' x64 VML libraries, linked statically."""

def _mkl_impl(ctx):
    windows = "windows" in ctx.os.name.lower()
    if ctx.os.arch not in ["amd64", "x86_64"] or not (windows or "linux" in ctx.os.name.lower()):
        # The ARM oracles use SLEEF. Do not fetch an unused x64 SDK there.
        ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_library(name = "math")
filegroup(name = "licenses")
""")
        return
    if windows:
        version = "2025.2.0"
        platform = "win_amd64"
        archives = [
            ("include", "06/87/3eee37bf95c6b820b6394ad98e50132798514ecda1b2584c71c2c96b973c", "d20305b4adfa36407a808ec6a16dc5d6da6f8b9cb4a96bdcc0e0ab3239c43816"),
            ("static", "16/94/9f2519aa7dc0678ac70fa6044808c0eb411ca22a0ee5716956ea764ae26a", "8e619ce1b77ee9e25e1f700f73813018eb1fd45041bd5f3287b5253afc57c555"),
        ]
    else:
        version = "2024.2.0"
        platform = "manylinux1_x86_64"
        archives = [
            ("include", "80/e4/93ddfd475420f1c24d96f3bba1f87ec31a1eea847884c4ccb243cb336a61", "63ed16ece64d9420e9fe1d5e1b55e0680632b61ad1c0e5f207b17f85233fcc09"),
            ("static", "c1/44/42ea3ad7bbaa65acb54c977961118d7b24ea687e7c3d64aba0a019cbfa19", "8c2a6c6a144c5619f1df75fd550b32730f3e0632b55a15a42a95516e142ccf47"),
        ]
    for kind, path, checksum in archives:
        filename = "mkl_{}-{}-py2.py3-none-{}.whl".format(kind, version, platform)
        ctx.download_and_extract(
            url = "https://files.pythonhosted.org/packages/" + path + "/" + filename,
            sha256 = checksum,
            type = "zip",
        )
    prefix = "Library/" if windows else ""
    library_root = "mkl_static-" + version + ".data/data/" + prefix + "lib/"
    libraries = [library_root + ("mkl_" + name + ".lib" if windows else "libmkl_" + name + ".a") for name in ["intel_lp64", "sequential", "core"]]
    ctx.template("BUILD.bazel", ctx.attr.build_file, {
        "%HEADER_ROOT%": "mkl_include-" + version + ".data/data/" + prefix + "include",
        "%LIBRARIES%": repr(libraries),
        "%LINK_PREFIX%": "[]" if windows else repr(["-Wl,--start-group"]),
        # Hide the static SDK's symbols so importing Torch cannot interpose a
        # different MKL copy. No OpenMP or MKL shared library is needed.
        "%LINK_SUFFIX%": "[]" if windows else repr(["-Wl,--end-group", "-Wl,--exclude-libs,ALL", "-ldl", "-lpthread", "-lm"]),
    })

mjlab_mkl_repository = repository_rule(
    implementation = _mkl_impl,
    attrs = {"build_file": attr.label(default = "//third_party/mjlab:mkl.BUILD.tpl", allow_single_file = True)},
    configure = True,
)
