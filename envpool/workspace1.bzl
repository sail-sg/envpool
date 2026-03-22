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

"""EnvPool workspace initialization, load after workspace0."""

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
load("@com_justbuchanan_rules_qt//:qt_configure.bzl", "qt_configure")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_python//python:repositories.bzl", "python_register_toolchains")
load("//envpool:python_tool_versions.bzl", "PYTHON_TOOL_VERSIONS")

def workspace():
    """Configure pip requirements."""
    python_register_toolchains(
        name = "python3_12",
        python_version = "3.12.7",
        ignore_root_user_error = True,
        tool_versions = PYTHON_TOOL_VERSIONS,
    )

    rules_foreign_cc_dependencies(
        register_built_pkgconfig_toolchain = False,
    )

    boost_deps()

    qt_configure()

workspace1 = workspace
