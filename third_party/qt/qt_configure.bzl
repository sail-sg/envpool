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

"""Minimal Linux-only Qt repository rule for EnvPool."""

def _get_env_var(repository_ctx, name, default = None):
    for key, value in repository_ctx.os.environ.items():
        if name == key:
            return value
    return default

def _qt_autoconf_impl(repository_ctx):
    os_name = repository_ctx.os.name.lower()
    if "linux" not in os_name:
        fail("EnvPool Qt configure currently only supports Linux, got %s" % repository_ctx.os.name)

    qt_path = "/usr/include/x86_64-linux-gnu/qt5"
    if not repository_ctx.path(qt_path).exists:
        qt_path = "/usr/include/qt"

    env_qt_path = _get_env_var(repository_ctx, "BAZEL_RULES_QT_DIR")
    if env_qt_path:
        qt_path = env_qt_path
        qt_path_with_include = qt_path + "/include"
        if repository_ctx.path(qt_path_with_include).exists:
            qt_path = qt_path_with_include

    if not repository_ctx.path(qt_path).exists:
        fail("Unable to locate Qt headers. Set BAZEL_RULES_QT_DIR or install qtbase5-dev.")

    repository_ctx.file("BUILD", "# empty BUILD file so that bazel sees this as a valid package directory")
    repository_ctx.template(
        "local_qt.bzl",
        repository_ctx.path(Label("//third_party/qt:BUILD.local_qt.tpl")),
        {"%{path}": qt_path},
    )

qt_autoconf = repository_rule(
    implementation = _qt_autoconf_impl,
    configure = True,
)

def qt_configure():
    qt_autoconf(name = "local_config_qt")
