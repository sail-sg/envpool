load("@pip_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "api",
    srcs = ["api.py"],
    deps = [
        requirement("gym"),
        requirement("dm-env"),
        requirement("numpy"),
        requirement("dm-tree"),
    ],
)

py_library(
    name = "spec_fuzzer",
    srcs = ["spec_fuzzer.py"],
    deps = [
        requirement("numpy"),
    ],
)

py_library(
    name = "python",
    srcs = ["__init__.py"],
    deps = [
        ":api",
        ":spec_fuzzer",
    ],
)
