package(default_visibility = ["//visibility:public"])
load("@com_justbuchanan_rules_qt//:qt.bzl", "qt_cc_library", "qt_ui_library")

cc_library(
    name = "procgen",
    srcs = [
        "assetgen.cpp",
        "basic-abstract-game.cpp",
        "cpp-utils.cpp",
        "entity.cpp",
        "game-registry.cpp",
        "game.cpp",
        "mazegen.cpp",
        "randgen.cpp",
        "resources.cpp",
        "roomgen.cpp",
        "vecgame.cpp",
        "vecoptions.cpp",
        "games/bigfish.cpp",
        "games/bossfight.cpp",
        "games/caveflyer.cpp",
        "games/chaser.cpp",
        "games/climber.cpp",
        "games/coinrun_old.cpp",
        "games/coinrun.cpp",
        "games/dodgeball.cpp",
        "games/fruitbot.cpp",
        "games/heist.cpp",
        "games/jumper.cpp",
        "games/leaper.cpp",
        "games/maze.cpp",
        "games/miner.cpp",
        "games/ninja.cpp",
        "games/plunder.cpp",
        "games/starpilot.cpp",
    ],
    hdrs = [
        "assetgen.h",
        "basic-abstract-game.h",
        "buffer.h",
        "cpp-utils.h",
        "entity.h",
        "game-registry.h",
        "game.h",
        "grid.h",
        "mazegen.h",
        "object-ids.h",
        "qt-utils.h",
        "randgen.h",
        "resources.h",
        "roomgen.h",
        "vecgame.h",
        "vecoptions.h",
        "libenv.h"
    ],
    copts = [
            '-fpic',
        ],
    deps = [
         "@qt//:qt_widgets",
         "@qt//:qt_core",
         "@qt//:qt_gui",
         "@qt//:qt_network",
         "@qt//:qt_qml",
    ],
)
