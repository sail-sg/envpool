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

"""Additional Python toolchain versions for rules_python 0.12."""

# Sourced from bazelbuild/rules_python 0.37.0.
PYTHON_TOOL_VERSIONS = {
    "3.12.7": {
        "url": "20241008/cpython-{python_version}+20241008-{platform}-{build}.tar.gz",
        "sha256": {
            "aarch64-apple-darwin": "dd07d467f1d533b93d06e4d2ff88b91f491329510c6434297b88b584641bff5d",
            "aarch64-unknown-linux-gnu": "ce3230da53aacb17ff77e912170786f47db4a446d4acb6cde7c397953a032bca",
            "ppc64le-unknown-linux-gnu": "27d3cba42e94593c49f8610dcadd74f5b731c78f04ebabc2b0e1ba031ec09441",
            "s390x-unknown-linux-gnu": "1e28e0fc9cd1fa0365a149c715c44d3030b2c989ca397fc074809b943449df41",
            "x86_64-apple-darwin": "2347bf53ed3623645bed35adfca950b2c5291e3a759ec6c7765aa707b5dc866b",
            "x86_64-pc-windows-msvc": "4ed1a146c66c7dbd85b87df69b17afc166ea7d70056aaf59a49c3d987a030d3b",
            "x86_64-unknown-linux-gnu": "adbda1f3b77d7b65a551206e34a225375f408f9823e2e11df4c332aaecb8714b",
        },
        "strip_prefix": "python",
    },
}
