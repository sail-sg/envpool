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

"""Qt repository rule for EnvPool."""

def _get_env_var(repository_ctx, name, default = None):
    for key, value in repository_ctx.os.environ.items():
        if name == key:
            return value
    return default

def _path_exists(repository_ctx, path):
    return path != None and repository_ctx.path(path).exists

def _resolve_qt_include_dir(repository_ctx, raw_path):
    candidates = [
        raw_path,
        raw_path + "/include",
        raw_path + "/include/qt5",
    ]
    for candidate in candidates:
        if _path_exists(repository_ctx, candidate) and repository_ctx.path(candidate + "/QtCore").exists:
            return candidate
    return None

def _resolve_qt_lib_dir(repository_ctx, raw_path, include_dir):
    candidates = [
        raw_path + "/lib",
        include_dir + "/../lib",
    ]
    for candidate in candidates:
        if _path_exists(repository_ctx, candidate):
            return str(repository_ctx.path(candidate))
    return None

def _has_qt_framework(repository_ctx, lib_dir, framework):
    return _path_exists(repository_ctx, lib_dir + "/%s.framework" % framework)

def _generate_build_file(linkopts):
    return """load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "qt_core",
    hdrs = glob(
        ["QtCore/**"],
        allow_empty = True,
    ),
    includes = ["."],
    linkopts = {qt_core_linkopts},
)

cc_library(
    name = "qt_gui",
    hdrs = glob(
        ["QtGui/**"],
        allow_empty = True,
    ),
    includes = ["."],
    linkopts = {qt_gui_linkopts},
    deps = [":qt_core"],
)
""".format(
        qt_core_linkopts = repr(linkopts["qt_core"]),
        qt_gui_linkopts = repr(linkopts["qt_gui"]),
    )

def _symlink_tree(repository_ctx, include_dir):
    include_path = repository_ctx.path(include_dir)
    for entry in include_path.readdir():
        repository_ctx.symlink(entry, entry.basename)

def _qt_autoconf_impl(repository_ctx):
    os_name = repository_ctx.os.name.lower()
    env_qt_path = _get_env_var(repository_ctx, "BAZEL_RULES_QT_DIR")

    qt_candidates = []
    if env_qt_path:
        qt_candidates.append(env_qt_path)

    if "linux" in os_name:
        qt_candidates.extend([
            "/usr/include/x86_64-linux-gnu/qt5",
            "/usr/include/qt",
        ])
    elif "mac" in os_name:
        qt_candidates.extend([
            "/opt/homebrew/opt/qt@5",
            "/usr/local/opt/qt@5",
            "/opt/homebrew/opt/qt",
            "/usr/local/opt/qt",
        ])
    elif "windows" in os_name:
        if env_qt_path:
            qt_candidates.append(env_qt_path)
    else:
        fail("EnvPool Qt configure does not support %s" % repository_ctx.os.name)

    include_dir = None
    lib_dir = None
    for candidate in qt_candidates:
        include_dir = _resolve_qt_include_dir(repository_ctx, candidate)
        if include_dir:
            lib_dir = _resolve_qt_lib_dir(repository_ctx, candidate, include_dir)
            break

    if include_dir:
        _symlink_tree(repository_ctx, include_dir)

    linkopts = {
        "qt_core": [],
        "qt_gui": [],
    }

    if "linux" in os_name:
        linkopts["qt_core"] = ["-lQt5Core"]
        linkopts["qt_gui"] = ["-lQt5Gui"]
    elif include_dir and lib_dir:
        if _has_qt_framework(repository_ctx, lib_dir, "QtCore") and _has_qt_framework(repository_ctx, lib_dir, "QtGui"):
            qt_core_linkopts = [
                "-F%s" % lib_dir,
                "-Wl,-rpath,%s" % lib_dir,
                "-framework",
                "QtCore",
            ]
            qt_gui_linkopts = [
                "-F%s" % lib_dir,
                "-Wl,-rpath,%s" % lib_dir,
                "-framework",
                "QtGui",
            ]
        else:
            qt_core_linkopts = [
                "-L%s" % lib_dir,
                "-Wl,-rpath,%s" % lib_dir,
                "-lQt5Core",
            ]
            qt_gui_linkopts = [
                "-L%s" % lib_dir,
                "-Wl,-rpath,%s" % lib_dir,
                "-lQt5Gui",
            ]
        linkopts["qt_core"] = qt_core_linkopts
        linkopts["qt_gui"] = qt_gui_linkopts

    if "linux" in os_name and not include_dir:
        fail("Unable to locate Qt headers. Set BAZEL_RULES_QT_DIR or install qtbase5-dev.")

    repository_ctx.file("BUILD.bazel", _generate_build_file(linkopts))

qt_autoconf = repository_rule(
    implementation = _qt_autoconf_impl,
    configure = True,
)

def qt_configure(name = "qt"):
    qt_autoconf(name = name)
