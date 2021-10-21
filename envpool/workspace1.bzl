"""EnvPool workspace initialization, load after workspace0."""

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
load("@rules_python//python:pip.bzl", "pip_parse")
load("@mypy_integration//repositories:repositories.bzl", mypy_integration_repositories = "repositories")
load("@mypy_integration//repositories:deps.bzl", mypy_integration_deps = "deps")
load("@mypy_integration//:config.bzl", "mypy_configuration")

def workspace():
    """Configure pip requirements and mypy integration."""
    python_configure(
        name = "local_config_python",
        python_version = "3",
    )

    if "pip_requirements" not in native.existing_rules().keys():
        pip_parse(
            name = "pip_requirements",
            python_interpreter = "python3",
            quiet = False,
            requirements_lock = "@envpool//third_party/pip_requirements:requirements_lock.txt",
        )

    mypy_integration_repositories()

    mypy_integration_deps(mypy_requirements_file = "@envpool//tools/typing:mypy_version.txt")

    if "mypy_integration_config" not in native.existing_rules().keys():
        mypy_configuration("@envpool//tools/typing:mypy.ini")

workspace1 = workspace
