package(default_visibility = ["//visibility:public"])

genrule(
    name = "vizdoom",
    srcs = ["@vizdoom//:vizdoom"],
    outs = ["vizdoom"],
    cmd = "cp $< $@",
    executable = True,
)

genrule(
    name = "vizdoom_pk3",
    srcs = ["@vizdoom//:vizdoom_pk3"],
    outs = ["vizdoom_pk3"],
    cmd = "cp $< $@",
    executable = True,
)

genrule(
    name = "freedoom",
    srcs = ["@freedoom"],
    outs = ["freedoom2.wad"],
    cmd = "cp $< $@",
    executable = True,
)
