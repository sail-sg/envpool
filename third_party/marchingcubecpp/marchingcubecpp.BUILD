package(default_visibility = ["//visibility:public"])

genrule(
    name = "mc_impl",
    outs = ["mc_impl.cc"],
    cmd = "printf '#include <cstdint>\\n#define MC_IMPLEM_ENABLE\\n#include \\\"MC.h\\\"\\n' > $@",
)

cc_library(
    name = "marchingcubecpp",
    srcs = [":mc_impl"],
    hdrs = [
        "MC.h",
        "noise.h",
    ],
    includes = ["."],
)
