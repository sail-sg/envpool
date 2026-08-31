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
"""Generate pinned Warp's native exports without compiling or loading a JIT."""

import sys
from pathlib import Path

from warp._src.context import export_builtins

exports, version = map(Path, sys.argv[1:])
with exports.open("w") as output:
    export_builtins(output)
version.write_text('#pragma once\n#define WP_VERSION_STRING "1.14.0"\n')
