# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:macos"],
)

config_setting(
    name = "darwin_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:macos",
    ],
)

config_setting(
    name = "linux_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:linux",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

# Shows a standard library using the Ninja generator
cmake(
    name = "opencv",
    generate_args = [
        "-GNinja",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_opencv_calib3d=OFF",
        "-DBUILD_opencv_core=ON",
        "-DBUILD_opencv_features2d=ON",
        "-DBUILD_opencv_flann=ON",
        "-DBUILD_opencv_gapi=ON",
        "-DBUILD_opencv_highgui=OFF",
        "-DBUILD_opencv_imgcodecs=OFF",
        "-DBUILD_opencv_imgproc=ON",
        "-DBUILD_opencv_java_bindings_generator=OFF",
        "-DBUILD_opencv_js=OFF",
        "-DBUILD_opencv_js_bindings_generator=OFF",
        "-DBUILD_opencv_ml=OFF",
        "-DBUILD_opencv_objc_bindings_generator=OFF",
        "-DBUILD_opencv_objdetect=OFF",
        "-DBUILD_opencv_photo=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DBUILD_opencv_python_bindings_generator=OFF",
        "-DBUILD_opencv_python_tests=OFF",
        "-DBUILD_opencv_stitching=OFF",
        "-DBUILD_opencv_ts=OFF",
        "-DBUILD_opencv_video=OFF",
        "-DBUILD_opencv_videoio=OFF",
        "-DBUILD_opencv_world=OFF",
        "-DWITH_CUDA=OFF",
        "-DWITH_EIGEN=ON",
        "-DWITH_FFMPEG=ON",
        "-DWITH_GTK=OFF",
        "-DWITH_GTK_2_X=OFF",
        "-DWITH_IPP=OFF",
        "-DWITH_ITT=OFF",
        "-DWITH_JASPER=ON",
        "-DWITH_JPEG=ON",
        "-DWITH_LAPACK=ON",
        "-DWITH_ONNX=OFF",
        "-DWITH_OPENCL=OFF",
        "-DWITH_OPENGL=OFF",
        "-DWITH_OPENJPEG=OFF",
        "-DWITH_PLAIDML=OFF",
        "-DWITH_PNG=OFF",
        "-DWITH_PROTOBUF=OFF",
        "-DWITH_QT=OFF",
        "-DWITH_TBB=OFF",
        "-DWITH_TIFF=OFF",
    ] + select({
        ":darwin": [
            # Avoid build-time KleidiCV downloads inside the Bazel sandbox on macOS.
            "-DWITH_KLEIDICV=OFF",
        ],
        "@envpool//:windows": [
            "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
            "-DCMAKE_OBJECT_PATH_MAX=200",
            "-DCV_DISABLE_OPTIMIZATION=ON",
            "-DOPENCV_WORKAROUND_CMAKE_20989=ON",
            "-DOPENCV_PYTHON_SKIP_DETECTION=ON",
            "-DWITH_PTHREADS_PF=OFF",
        ],
        "//conditions:default": [],
    }) + select({
        "@envpool//:windows": [],
        "//conditions:default": [
            "-DWITH_PTHREADS_PF=ON",
        ],
    }),
    lib_source = ":srcs",
    linkopts = select({
        "@envpool//:windows": [],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    out_include_dir = "include/opencv4",
    out_static_libs = select({
        "@envpool//:windows": [
            "opencv_imgproc4130.lib",
            "opencv_features2d4130.lib",
            "opencv_flann4130.lib",
            "opencv_core4130.lib",
        ],
        "//conditions:default": [
            "libopencv_imgproc.a",
            "libopencv_features2d.a",
            "libopencv_flann.a",
            "libopencv_core.a",
        ],
    }) + select({
        ":darwin_arm64": [
            "opencv4/3rdparty/libtegra_hal.a",
        ],
        ":linux_arm64": [
            "opencv4/3rdparty/libkleidicv_hal.a",
            "opencv4/3rdparty/libkleidicv_thread.a",
            "opencv4/3rdparty/libkleidicv.a",
            "opencv4/3rdparty/libtegra_hal.a",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
