load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

exports_files(["gfootball.BUILD"])

py_binary(
    name = "generate_scenarios",
    srcs = ["generate_scenarios.py"],
)
