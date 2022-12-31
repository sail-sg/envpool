package(default_visibility = ["//visibility:public"])

cc_library(
    name = "procgen",
    srcs = glob(["**/*.cpp"]) + glob(["*.h"]),
    hdrs = glob(["*.h"]),
    copts = [
        "-fpic",
    ],
    deps = [
        "@gym3_libenv//:gym3_libenv_header",
        "@qt//:qt_core",
        "@qt//:qt_gui",
        "@qt//:qt_network",
        "@qt//:qt_qml",
        "@qt//:qt_widgets",
    ],
)
