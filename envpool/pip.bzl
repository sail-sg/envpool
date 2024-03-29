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

"""EnvPool pip requirements initialization, this is loaded in WORKSPACE."""

load("@rules_python//python:pip.bzl", "pip_install")

def workspace():
    """Configure pip requirements."""

    if "pip_requirements" not in native.existing_rules().keys():
        pip_install(
            name = "pip_requirements",
            python_interpreter = "python3",
            # default timeout value is 600, change it if you failed.
            # timeout = 3600,
            quiet = False,
            requirements = "@envpool//third_party/pip_requirements:requirements.txt",
            # extra_pip_args = ["--extra-index-url", "https://mirrors.aliyun.com/pypi/simple"],
        )
