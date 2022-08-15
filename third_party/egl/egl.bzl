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

"""This is loaded in workspace0.bzl to provide EGL library."""

_EGL_DIR = "/usr/include"

def _impl(rctx):
    egl_dir = rctx.os.environ.get(_EGL_DIR, default = "/usr/include")
    rctx.symlink("{}/include".format(egl_dir), "include")
    rctx.file("WORKSPACE")
    rctx.file("BUILD", content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "egl_headers",
    hdrs = [
        "include/EGL/egl.h",
        "include/EGL/eglext.h",
        "include/EGL/eglplatform.h",
        "include/KHR/khrplatform.h",
    ],
    defines = ["USE_OZONE"],
)
""")

egl_headers = repository_rule(
    implementation = _impl,
    environ = [
        _EGL_DIR,
    ],
)
