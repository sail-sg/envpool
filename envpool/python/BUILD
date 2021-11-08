load("@pip_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "protocol",
    srcs = ["protocol.py"],
    deps = [
        requirement("dm-env"),
        requirement("gym"),
        requirement("numpy"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        requirement("numpy"),
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        requirement("dm-tree"),
        requirement("dm-env"),
        requirement("gym"),
        requirement("numpy"),
        ":protocol",
    ],
)

py_library(
    name = "env_spec",
    srcs = ["env_spec.py"],
    deps = [
        requirement("dm-env"),
        requirement("gym"),
        ":data",
        ":protocol",
        ":utils",
    ],
)

py_library(
    name = "envpool",
    srcs = ["envpool.py"],
    deps = [
        requirement("dm-tree"),
        requirement("dm-env"),
        requirement("numpy"),
        ":protocol",
    ],
)

py_library(
    name = "dm_envpool",
    srcs = ["dm_envpool.py"],
    deps = [
        requirement("dm-tree"),
        requirement("dm-env"),
        requirement("numpy"),
        ":data",
        ":envpool",
        ":utils",
    ],
)

py_library(
    name = "gym_envpool",
    srcs = ["gym_envpool.py"],
    deps = [
        requirement("dm-tree"),
        requirement("dm-env"),
        requirement("gym"),
        requirement("numpy"),
        ":data",
        ":envpool",
        ":utils",
    ],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    deps = [
        ":dm_envpool",
        ":env_spec",
        ":gym_envpool",
        ":protocol",
    ],
)

py_library(
    name = "python",
    srcs = ["__init__.py"],
    deps = [
        ":api",
    ],
)
