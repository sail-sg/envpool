load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:macos"],
)

cc_library(
    name = "freetype",
    hdrs = glob(["include/freetype2/**/*.h"]),
    includes = ["include/freetype2"],
    srcs = select({
        ":darwin": ["lib/libfreetype.dylib"],
        "//conditions:default": glob([
            "lib/libfreetype.so",
            "lib*/libfreetype.so",
            "lib*/**/libfreetype.so",
        ]),
    }),
)
