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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

_LIBRARIES = %LIBRARIES%

cc_library(
    name = "math",
    hdrs = glob(["%HEADER_ROOT%/**/*.h"]),
    includes = ["%HEADER_ROOT%"],
    defines = ["MJLAB_USE_MKL"],
    additional_linker_inputs = _LIBRARIES,
    linkopts = %LINK_PREFIX% + ["$(location " + library + ")" for library in _LIBRARIES] + %LINK_SUFFIX%,
)

filegroup(
    name = "licenses",
    srcs = glob(["**/LICENSE.txt", "**/licensing/*.txt"]),
)
