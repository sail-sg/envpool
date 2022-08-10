load("@pip_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

exports_files([
    "workspace0.bzl",
    "workspace1.bzl",
])

py_library(
    name = "registration",
    srcs = ["registration.py"],
)

py_library(
    name = "entry",
    srcs = ["entry.py"],
    deps = [
        "//envpool/atari:atari_registration",
        "//envpool/box2d:box2d_registration",
        "//envpool/classic_control:classic_control_registration",
        "//envpool/mujoco:mujoco_dmc_registration",
        "//envpool/mujoco:mujoco_gym_registration",
        "//envpool/toy_text:toy_text_registration",
        "//envpool/vizdoom:vizdoom_registration",
        "//envpool/football:football_registration",
    ],
)

py_library(
    name = "envpool",
    srcs = ["__init__.py"],
    deps = [
        ":entry",
        ":registration",
        "//envpool/atari",
        "//envpool/box2d",
        "//envpool/classic_control",
        "//envpool/football",
        "//envpool/mujoco:mujoco_dmc",
        "//envpool/mujoco:mujoco_gym",
        "//envpool/python",
        "//envpool/toy_text",
        "//envpool/vizdoom",
    ],
)

py_test(
    name = "make_test",
    srcs = ["make_test.py"],
    deps = [
        ":envpool",
        requirement("absl-py"),
    ],
)
