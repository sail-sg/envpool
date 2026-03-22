load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

licenses(["notice"])

sh_binary(
    name = "clang_tidy_executable",
    srcs = ["clang_tidy_wrapper.sh"],
    visibility = ["//visibility:public"],
)
