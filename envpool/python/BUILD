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
        requirement("treevalue"),
        requirement("dm-env"),
        requirement("gym"),
        requirement("gymnasium"),
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
        requirement("treevalue"),
        requirement("dm-env"),
        requirement("numpy"),
        requirement("packaging"),
        ":protocol",
    ],
)

py_library(
    name = "xla_template",
    srcs = ["xla_template.py"],
    deps = [
        requirement("jax"),
    ],
)

py_library(
    name = "lax",
    srcs = ["lax.py"],
    deps = [
        requirement("jax"),
        requirement("dm-env"),
        requirement("numpy"),
        requirement("absl-py"),
        ":protocol",
        ":xla_template",
    ],
)

py_library(
    name = "dm_envpool",
    srcs = ["dm_envpool.py"],
    deps = [
        requirement("treevalue"),
        requirement("dm-env"),
        requirement("numpy"),
        ":data",
        ":envpool",
        ":lax",
        ":utils",
    ],
)

py_library(
    name = "gym_envpool",
    srcs = ["gym_envpool.py"],
    deps = [
        requirement("treevalue"),
        requirement("dm-env"),
        requirement("gym"),
        requirement("numpy"),
        ":data",
        ":envpool",
        ":lax",
        ":utils",
    ],
)

py_library(
    name = "gymnasium_envpool",
    srcs = ["gymnasium_envpool.py"],
    deps = [
        requirement("treevalue"),
        requirement("dm-env"),
        requirement("gymnasium"),
        requirement("numpy"),
        ":data",
        ":envpool",
        ":lax",
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
