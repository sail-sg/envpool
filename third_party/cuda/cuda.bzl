_CUDA_DIR = "CUDA_DIR"

def _impl(rctx):
    cuda_dir = rctx.os.environ.get(_CUDA_DIR, default = "/usr/local/cuda")
    rctx.symlink("{}/include".format(cuda_dir), "include")
    rctx.symlink("{}/lib64".format(cuda_dir), "lib64")
    rctx.file("WORKSPACE")
    rctx.file("BUILD", content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudart_static",
    srcs = ["lib64/libcudart_static.a"],
    hdrs = glob([
        "include/*.h",
        "include/**/*.h",
    ]),
    strip_include_prefix = "include",
)
""")

cuda_configure = repository_rule(
    implementation = _impl,
    environ = [
        _CUDA_DIR,
    ],
)
