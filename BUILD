load("@pip_requirements//:requirements.bzl", "requirement")

py_binary(
    name = "setup",
    srcs = [
        "setup.py",
    ],
    data = [
        "README.md",
        "setup.cfg",
        "//envpool",
    ],
    main = "setup.py",
    python_version = "PY3",
    deps = [
        requirement("setuptools"),
        requirement("wheel"),
    ],
)
