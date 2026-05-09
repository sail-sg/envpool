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

"""Workspace setup for the pinned Jumanji oracle dependencies."""

load("@python_versions//:pip.bzl", "multi_pip_parse")

def jumanji_oracle_pip_workspace():
    """Configure official Jumanji dependencies used only by docs/tests."""
    if "jumanji_oracle_requirements" in native.existing_rules().keys():
        return

    # Keep the official Jumanji oracle on NumPy 1.26/JAX 0.4.35. The v1.1.1
    # Tetris viewer still calls APIs that are incompatible with NumPy 2.4.
    multi_pip_parse(
        name = "jumanji_oracle_requirements",
        default_version = "3.12",
        python_interpreter_target = {
            "3.11": "@python_versions_3_12_host//:python",
            "3.12": "@python_versions_3_12_host//:python",
            "3.13": "@python_versions_3_12_host//:python",
            "3.14": "@python_versions_3_12_host//:python",
        },
        requirements_lock = {
            "3.11": "@envpool//third_party/jumanji:oracle_requirements.txt",
            "3.12": "@envpool//third_party/jumanji:oracle_requirements.txt",
            "3.13": "@envpool//third_party/jumanji:oracle_requirements.txt",
            "3.14": "@envpool//third_party/jumanji:oracle_requirements.txt",
        },
        quiet = False,
    )
