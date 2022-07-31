load("@envpool//third_party:common.bzl", "template_rule")

package(default_visibility = ["//visibility:public"])

template_rule(
    name = "ccd_config",
    src = "src/ccd/config.h.cmake.in",
    out = "src/ccd/config.h",
    substitutions = {
        "#cmakedefine CCD_SINGLE": "",
        "#cmakedefine CCD_DOUBLE": "#define CCD_DOUBLE",
    },
)

cc_library(
    name = "ccd",
    srcs = glob(["src/*.h"]) + glob(["src/*.c"]) + [":ccd_config"],
    hdrs = glob(["src/ccd/*.h"]),
    includes = ["src"],
)
