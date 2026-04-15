filegroup(
    name = "myosuite_runtime_assets",
    srcs = glob(
        ["myosuite/envs/myo/assets/**"],
        exclude = [
            "myosuite/envs/myo/assets/**/*.py",
            "myosuite/envs/myo/assets/**/__pycache__/**",
        ],
    ),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "myosuite_registry_sources",
    srcs = [
        ".gitmodules",
        "myosuite/__init__.py",
        "myosuite/envs/myo/myobase/__init__.py",
        "myosuite/envs/myo/myochallenge/__init__.py",
        "myosuite/envs/myo/myodm/__init__.py",
        "myosuite/envs/myo/myoedits/__init__.py",
        "myosuite_init.py",
        "pyproject.toml",
    ],
    visibility = ["//visibility:public"],
)
