package(default_visibility = ["//visibility:public"])

cc_library(
    name = "game_env",
    hdrs = ["third_party/gfootball_engine/src/game_env.hpp"],
    srcs = glob(["third_party/gfootball_engine/src/**/*.h",
            "third_party/gfootball_engine/src/**/*.hpp",
            "third_party/gfootball_engine/src/**/*.c",
            "third_party/gfootball_engine/src/**/*.cpp",
            ]),
    includes = [
        "third_party/gfootball_engine/src",
        "third_party/gfootball_engine/src/ai",
        "third_party/gfootball_engine/src/base",
        "third_party/gfootball_engine/src/cmake",
        "third_party/gfootball_engine/src/data",
        "third_party/gfootball_engine/src/hid",
        "third_party/gfootball_engine/src/loaders",
        "third_party/gfootball_engine/src/managers",
        "third_party/gfootball_engine/src/menu",
        "third_party/gfootball_engine/src/misc",
        "third_party/gfootball_engine/src/onthepitch",
        "third_party/gfootball_engine/src/scene",
        "third_party/gfootball_engine/src/systems",
        "third_party/gfootball_engine/src/types",
        "third_party/gfootball_engine/src/utils",
        ],
)

