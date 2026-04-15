filegroup(
    name = "runtime_assets",
    srcs = glob(
        ["**"],
        exclude = [
            "**/.git/**",
            "**/.github/**",
            "**/.idea/**",
            "**/__pycache__/**",
            "**/*.md",
            "**/*.py",
            "**/LICENSE",
            "**/LICENSE.*",
        ],
    ),
    visibility = ["//visibility:public"],
)
