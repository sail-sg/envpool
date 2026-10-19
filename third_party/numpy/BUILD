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

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

[genrule(
    name = name + "_header",
    srcs = ["@numpy_" + name + "//file"],
    outs = [name + ".h"],
    cmd = "cp $< $@",
    cmd_bat = 'copy /Y "$<" "$@" >NUL',
) for name in [
    "pcg64",
    "ziggurat_constants",
    "pocketfft_hdronly",
]]

cc_library(
    name = "random_headers",
    hdrs = [
        ":pcg64_header",
        ":ziggurat_constants_header",
    ],
)

cc_library(
    name = "fft",
    hdrs = [":pocketfft_hdronly_header"],
)
