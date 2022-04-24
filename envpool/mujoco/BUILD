package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_mujoco_xml",
    srcs = ["@mujoco_xml"],
    outs = ["assets"],
    cmd = "mkdir -p $(OUTS) && cp $(SRCS) $(OUTS)",
)

cc_test(
    name = "ant_test",
    srcs = ["ant.cc"],
    data = [
        ":gen_mujoco_xml",
    ],
    deps = [
        "@mujoco//:mujoco_lib",
    ],
)
