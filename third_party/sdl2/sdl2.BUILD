load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

# https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=sdl2-static
cmake(
    name = "sdl2",
    generate_args = [
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",  # always compile for release
        "-DSDL_STATIC=ON",
        "-DSDL_STATIC_LIB=ON",
        "-DSDL_DLOPEN=ON",
        "-DARTS=OFF",
        "-DESD=OFF",
        "-DNAS=OFF",
        "-DALSA=ON",
        "-DHIDAPI=ON",
        "-DPULSEAUDIO_SHARED=ON",
        "-DVIDEO_WAYLAND=ON",
        "-DRPATH=OFF",
        "-DCLOCK_GETTIME=ON",
        "-DJACK_SHARED=ON",
        "-DSDL_STATIC_PIC=ON",
    ],
    lib_source = ":srcs",
    out_include_dir = "include",
    out_static_libs = ["libSDL2.a"],
    visibility = ["//visibility:public"],
)
