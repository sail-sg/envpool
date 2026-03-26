workspace(name = "envpool")

load("//envpool:workspace0.bzl", workspace0 = "workspace")

workspace0()

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

load("@rules_cc//cc:extensions.bzl", "compatibility_proxy_repo")

compatibility_proxy_repo()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@rules_java//java:rules_java_deps.bzl", "rules_java_dependencies")

rules_java_dependencies()

load("@rules_java//java:repositories.bzl", "rules_java_toolchains")

rules_java_toolchains()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("//envpool:workspace1.bzl", workspace1 = "workspace")

workspace1()

load("//envpool:pip.bzl", pip_workspace = "workspace")

pip_workspace()

load("@pip_requirements//:requirements.bzl", "install_deps")

install_deps()
