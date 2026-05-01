# Copyright 2026 Garena Online Private Limited
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

load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "re2c",
    srcs = glob(
        ["src/**/*.cc"],
        exclude = ["src/test/**"],
    ) + [
        "bootstrap/src/msg/help_re2c.cc",
        "bootstrap/src/options/parse_opts.cc",
        "bootstrap/src/parse/conf_lexer.cc",
        "bootstrap/src/parse/conf_parser.cc",
        "bootstrap/src/parse/lexer.cc",
        "bootstrap/src/parse/parser.cc",
    ] + glob([
        "bootstrap/src/**/*.h",
        "src/**/*.h",
    ]),
    copts = [
        "-DRE2C_STDLIB_DIR=\\\"\\\"",
    ] + select({
        "@envpool//:windows": [
            "/D_CRT_SECURE_NO_WARNINGS",
            "/DNOMINMAX",
        ],
        "//conditions:default": [],
    }),
    includes = [
        ".",
        "bootstrap",
    ],
    deps = ["@envpool//third_party/re2c:config"],
)
