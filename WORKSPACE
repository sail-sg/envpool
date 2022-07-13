workspace(name = "envpool")

load("//envpool:workspace0.bzl", workspace0 = "workspace")

workspace0()

load("//envpool:workspace1.bzl", workspace1 = "workspace")

workspace1()

load("//envpool:pip.bzl", pip_workspace = "workspace")

pip_workspace()

new_local_repository(
    name = "cuda",
    path = "/usr/local/cuda-11.6",
    build_file = "//third_party/cuda:cuda.BUILD",
)
