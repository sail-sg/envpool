workspace(name = "envpool")

load("//envpool:workspace0.bzl", workspace0 = "workspace")

workspace0()

load("//envpool:workspace1.bzl", workspace1 = "workspace")

workspace1()

load("@pip_requirements//:requirements.bzl", "install_deps")

install_deps()
