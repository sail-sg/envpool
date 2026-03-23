# OpenXLA FFI headers

Pinned OpenXLA FFI headers used by `envpool/core/xla_template.h`.

Source archive:

- `openxla/xla@187a5eb58277a85847d1516bd1e20b7faf03d5ef`
- fetched via `http_archive` in `envpool/workspace0.bzl`
- this is the XLA revision pinned by `jax-v0.9.2` in `third_party/xla/revision.bzl`
- only the `xla/ffi/api/` subtree is extracted into the external repo

Only `xla/ffi/api/{api.h,c_api.h,ffi.h}` is exposed to Bazel.
