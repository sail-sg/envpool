# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the official renderer's expensive constant caches once per Bazel build."""

import argparse
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="craftax-mpl-"))

from craftax.craftax import constants as full
from craftax.craftax_classic import constants as classic


def main() -> None:
    """Generate the requested build artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic", required=True)
    parser.add_argument("--full", required=True)
    args = parser.parse_args()
    for module, path in ((classic, args.classic), (full, args.full)):
        module.TEXTURE_CACHE_FILE = path
        module.load_all_textures()


if __name__ == "__main__":
    main()
