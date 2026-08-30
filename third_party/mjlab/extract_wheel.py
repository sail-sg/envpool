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
"""Extract declared native library and license members from a pinned wheel."""

import shutil
import sys
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    """Copy only the requested archive members, without importing the package."""
    with ZipFile(sys.argv[1]) as wheel:
        for member, output in zip(sys.argv[2::2], sys.argv[3::2], strict=True):
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with wheel.open(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)


if __name__ == "__main__":
    main()
