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

def _path_exists(repository_ctx, path):
    return path != None and repository_ctx.path(path).exists

def _resolve_qt_include_dir(repository_ctx, raw_path):
    candidates = [
        raw_path,
        raw_path + "/include",
        raw_path + "/include/qt6",
        raw_path + "/include/qt5",
    ]
    for candidate in candidates:
        if _path_exists(repository_ctx, candidate) and repository_ctx.path(candidate + "/QtCore").exists:
            return candidate

    # Homebrew Qt 6 keeps public headers inside its frameworks.
    if _path_exists(repository_ctx, raw_path + "/lib/QtCore.framework/Headers"):
        return raw_path + "/lib"
    return None

def _resolve_qt_lib_dir(repository_ctx, raw_path, include_dir):
    candidates = [
        raw_path + "/lib",
        raw_path + "/lib64",
        include_dir + "/../lib",
    ]
    for candidate in candidates:
        if _path_exists(repository_ctx, candidate):
            return str(repository_ctx.path(candidate))
    return None

def _resolve_qt_dll_dir(repository_ctx, raw_path, lib_dir):
    candidates = [
        raw_path + "/bin",
        lib_dir + "/../bin",
    ]
    for candidate in candidates:
        if _path_exists(repository_ctx, candidate):
            return str(repository_ctx.path(candidate))
    return None

def _has_qt_framework(repository_ctx, lib_dir, framework):
    return _path_exists(repository_ctx, lib_dir + "/%s.framework" % framework)

def _generate_unix_build_file(linkopts):
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

def _generate_windows_build_file(major):
    return """load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "qt_core_import",
    interface_library = "Qt{major}Core.lib",
    shared_library = "Qt{major}Core.dll",
)

cc_import(
    name = "qt_gui_import",
    interface_library = "Qt{major}Gui.lib",
    shared_library = "Qt{major}Gui.dll",
)

cc_library(
    name = "qt_core",
    hdrs = glob(
        ["QtCore/**"],
        allow_empty = True,
    ),
    includes = ["."],
    deps = [":qt_core_import"],
)

cc_library(
    name = "qt_gui",
    hdrs = glob(
        ["QtGui/**"],
        allow_empty = True,
    ),
    includes = ["."],
    deps = [
        ":qt_core",
        ":qt_gui_import",
    ],
)
""".format(major = major)

def _symlink_tree(repository_ctx, include_dir):
    for module in ["QtCore", "QtGui"]:
        source = include_dir + "/" + module
        if not _path_exists(repository_ctx, source):
            source = include_dir + "/" + module + ".framework/Headers"
        repository_ctx.symlink(source, module)

def _qt_major(repository_ctx):
    # qconfig.h can be an architecture-dispatch wrapper in distro packages.
    for line in repository_ctx.read("QtCore/qtcoreversion.h").splitlines():
        fields = [field for field in line.replace("\t", " ").split(" ") if field]
        if len(fields) == 3 and fields[1] == "QTCORE_VERSION_STR":
            major = fields[2].strip('"').split(".")[0]
            if major in ["5", "6"]:
                return major
    fail("EnvPool requires Qt 5 or Qt 6.")

def _symlink_if_exists(repository_ctx, source, destination):
    if _path_exists(repository_ctx, source):
        repository_ctx.symlink(source, destination)
        return True
    return False

def _qt_autoconf_impl(repository_ctx):
    os_name = repository_ctx.os.name.lower()
    env_qt_path = repository_ctx.getenv("BAZEL_RULES_QT_DIR")

    qt_candidates = []
    if env_qt_path:
        qt_candidates.append(env_qt_path)

    if "linux" in os_name:
        qt_candidates.extend([
            "/usr/include/x86_64-linux-gnu/qt6",
            "/usr/include/aarch64-linux-gnu/qt6",
            "/usr/include/qt6",
            "/usr/include/x86_64-linux-gnu/qt5",
            "/usr/include/aarch64-linux-gnu/qt5",
            "/usr/include/qt5",
            "/usr/include/qt",
        ])
    elif "mac" in os_name:
        qt_candidates.extend([
            "/opt/homebrew/opt/qtbase",
            "/usr/local/opt/qtbase",
            "/opt/homebrew/opt/qt",
            "/usr/local/opt/qt",
            "/opt/homebrew/opt/qt@5",
            "/usr/local/opt/qt@5",
        ])
    elif "windows" not in os_name:
        fail("EnvPool Qt configure does not support %s" % repository_ctx.os.name)

    include_dir = None
    lib_dir = None
    dll_dir = None
    for candidate in qt_candidates:
        include_dir = _resolve_qt_include_dir(repository_ctx, candidate)
        if include_dir:
            lib_dir = _resolve_qt_lib_dir(repository_ctx, candidate, include_dir)
            if "windows" in os_name and lib_dir:
                dll_dir = _resolve_qt_dll_dir(repository_ctx, candidate, lib_dir)
            break

    if not include_dir:
        fail("Unable to locate Qt headers. Set BAZEL_RULES_QT_DIR to a Qt install or install qt6-base-dev.")
    _symlink_tree(repository_ctx, include_dir)
    major = _qt_major(repository_ctx)

    if "windows" in os_name:
        if not lib_dir or not dll_dir:
            fail("Unable to locate a Qt MSVC install. Set BAZEL_RULES_QT_DIR to a Qt root like C:/Qt/6.11.1/msvc2022_64.")
        required_files = [
            ("{}/Qt{}Core.lib".format(lib_dir, major), "Qt{}Core.lib".format(major)),
            ("{}/Qt{}Gui.lib".format(lib_dir, major), "Qt{}Gui.lib".format(major)),
            ("{}/Qt{}Core.dll".format(dll_dir, major), "Qt{}Core.dll".format(major)),
            ("{}/Qt{}Gui.dll".format(dll_dir, major), "Qt{}Gui.dll".format(major)),
        ]
        for source, destination in required_files:
            if not _symlink_if_exists(repository_ctx, source, destination):
                fail("Unable to locate {}".format(source))
        repository_ctx.file("BUILD.bazel", _generate_windows_build_file(major))
        return

    linkopts = {
        "qt_core": [],
        "qt_gui": [],
    }

    if "linux" in os_name:
        # Release builds install Qt under a private prefix inside manylinux.
        # The rpath also lets auditwheel locate the libraries for bundling.
        search_opts = ["-L%s" % lib_dir, "-Wl,-rpath,%s" % lib_dir] if lib_dir else []
        linkopts["qt_core"] = search_opts + ["-lQt%sCore" % major]
        linkopts["qt_gui"] = ["-lQt%sGui" % major]
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
                "-lQt%sCore" % major,
            ]
            qt_gui_linkopts = [
                "-L%s" % lib_dir,
                "-Wl,-rpath,%s" % lib_dir,
                "-lQt%sGui" % major,
            ]
        linkopts["qt_core"] = qt_core_linkopts
        linkopts["qt_gui"] = qt_gui_linkopts

    repository_ctx.file("BUILD.bazel", _generate_unix_build_file(linkopts))

qt_autoconf = repository_rule(
    implementation = _qt_autoconf_impl,
    configure = True,
)

def qt_configure(name = "qt"):
    qt_autoconf(name = name)
