load("@rules_python//python:pip.bzl", "pip_install")

def workspace():
    """Configure pip requirements."""

    if "pip_requirements" not in native.existing_rules().keys():
        pip_install(
            name = "pip_requirements",
            python_interpreter = "python3",
            quiet = False,
            requirements = "@envpool//third_party/pip_requirements:requirements.txt",
        )
