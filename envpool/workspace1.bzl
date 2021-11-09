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

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@mypy_integration//repositories:repositories.bzl", mypy_integration_repositories = "repositories")
load("@mypy_integration//repositories:deps.bzl", mypy_integration_deps = "deps")
load("@mypy_integration//:config.bzl", "mypy_configuration")

def workspace():
    """Configure pip requirements and mypy integration."""
    python_configure(
        name = "local_config_python",
        python_version = "3",
    )

    rules_foreign_cc_dependencies()

    mypy_integration_repositories()

    mypy_integration_deps(mypy_requirements_file = "@envpool//third_party/mypy:mypy_version.txt")

    if "mypy_integration_config" not in native.existing_rules().keys():
        mypy_configuration("@envpool//third_party/mypy:mypy.ini")

workspace1 = workspace
