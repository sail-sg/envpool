load("@python_versions//3.11:defs.bzl", py_binary_311 = "py_binary")
load("@python_versions//3.12:defs.bzl", py_binary_312 = "py_binary")
load("@python_versions//3.13:defs.bzl", py_binary_313 = "py_binary")
load("@python_versions//3.14:defs.bzl", py_binary_314 = "py_binary")
load("@rules_python//python:defs.bzl", "py_binary")
load("//envpool:requirements.bzl", "requirement")

config_setting(
    name = "linux",
    constraint_values = ["@platforms//os:linux"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

_SETUP_SRCS = [
    "setup.py",
]

_SETUP_DATA = [
    "README.md",
    "setup.cfg",
    "//envpool",
    "//third_party/gfootball:setup_py_data",
]

_SETUP_DEPS = [
    requirement("setuptools"),
    requirement("wheel"),
]

filegroup(
    name = "clang_tidy_config",
    srcs = [".clang-tidy"],
)

py_binary(
    name = "setup",
    srcs = _SETUP_SRCS,
    data = _SETUP_DATA,
    main = "setup.py",
    python_version = "PY3",
    deps = _SETUP_DEPS,
)

py_binary_311(
    name = "setup_py311",
    srcs = _SETUP_SRCS,
    data = _SETUP_DATA,
    main = "setup.py",
    deps = _SETUP_DEPS,
)

py_binary_312(
    name = "setup_py312",
    srcs = _SETUP_SRCS,
    data = _SETUP_DATA,
    main = "setup.py",
    deps = _SETUP_DEPS,
)

py_binary_313(
    name = "setup_py313",
    srcs = _SETUP_SRCS,
    data = _SETUP_DATA,
    main = "setup.py",
    deps = _SETUP_DEPS,
)

py_binary_314(
    name = "setup_py314",
    srcs = _SETUP_SRCS,
    data = _SETUP_DATA,
    main = "setup.py",
    deps = _SETUP_DEPS,
)
