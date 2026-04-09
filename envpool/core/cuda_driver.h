/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_CORE_CUDA_DRIVER_H_
#define ENVPOOL_CORE_CUDA_DRIVER_H_

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <dlfcn.h>
#endif

// XLA FFI returns a platform stream as an opaque pointer. On CUDA platforms
// this is a CUstream; keep the type opaque so CPU wheels do not depend on
// CUDA runtime headers or libcudart.
using EnvPoolGpuStream = void*;

#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace envpool::cuda_driver {

#if defined(__linux__)

class DriverApi {
 public:
  DriverApi() {
    handle_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    CHECK_NE(handle_, nullptr)
        << "EnvPool XLA GPU backend requires libcuda.so.1: " << dlerror();

    cu_init_ = Load<CuInit>("cuInit");
    cu_get_error_string_ = Load<CuGetErrorString>("cuGetErrorString");
    cu_memcpy_dtoh_async_ = Load<CuMemcpyDtoHAsync>("cuMemcpyDtoHAsync_v2");
    cu_memcpy_htod_async_ = Load<CuMemcpyHtoDAsync>("cuMemcpyHtoDAsync_v2");
    cu_stream_synchronize_ = Load<CuStreamSynchronize>("cuStreamSynchronize");

    Check(cu_init_(0), "cuInit");
  }

  DriverApi(const DriverApi&) = delete;
  DriverApi& operator=(const DriverApi&) = delete;

  void CopyDeviceToHostAsync(void* dst_host, const void* src_device,
                             std::size_t bytes, EnvPoolGpuStream stream) const {
    if (bytes == 0) {
      return;
    }
    Check(cu_memcpy_dtoh_async_(dst_host, DevicePtr(src_device), bytes, stream),
          "cuMemcpyDtoHAsync_v2");
  }

  void CopyHostToDeviceAsync(void* dst_device, const void* src_host,
                             std::size_t bytes, EnvPoolGpuStream stream) const {
    if (bytes == 0) {
      return;
    }
    Check(cu_memcpy_htod_async_(DevicePtr(dst_device), src_host, bytes, stream),
          "cuMemcpyHtoDAsync_v2");
  }

  void SynchronizeStream(EnvPoolGpuStream stream) const {
    Check(cu_stream_synchronize_(stream), "cuStreamSynchronize");
  }

 private:
  using CUdeviceptr = unsigned long long;  // NOLINT
  using CUresult = int;
  using CuInit = CUresult (*)(unsigned int);
  using CuGetErrorString = CUresult (*)(CUresult, const char**);
  using CuMemcpyDtoHAsync = CUresult (*)(void*, CUdeviceptr, std::size_t,
                                         EnvPoolGpuStream);
  using CuMemcpyHtoDAsync = CUresult (*)(CUdeviceptr, const void*, std::size_t,
                                         EnvPoolGpuStream);
  using CuStreamSynchronize = CUresult (*)(EnvPoolGpuStream);

  static_assert(sizeof(CUdeviceptr) >= sizeof(void*));

  void* handle_{nullptr};
  CuInit cu_init_{nullptr};
  CuGetErrorString cu_get_error_string_{nullptr};
  CuMemcpyDtoHAsync cu_memcpy_dtoh_async_{nullptr};
  CuMemcpyHtoDAsync cu_memcpy_htod_async_{nullptr};
  CuStreamSynchronize cu_stream_synchronize_{nullptr};

  template <typename Fn>
  Fn Load(const char* symbol) {
    void* fn = dlsym(handle_, symbol);
    CHECK_NE(fn, nullptr) << "Failed to load CUDA driver symbol " << symbol
                          << ": " << dlerror();
    return reinterpret_cast<Fn>(fn);
  }

  static CUdeviceptr DevicePtr(const void* ptr) {
    return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(ptr));
  }

  std::string ErrorString(CUresult result) const {
    const char* msg = nullptr;
    if (cu_get_error_string_(result, &msg) == 0 && msg != nullptr) {
      return msg;
    }
    return "unknown CUDA driver error";
  }

  void Check(CUresult result, const char* op) const {
    CHECK_EQ(result, 0) << op << " failed: " << ErrorString(result);
  }
};

#else

class DriverApi {
 public:
  void CopyDeviceToHostAsync(void* /*dst_host*/, const void* /*src_device*/,
                             std::size_t /*bytes*/,
                             EnvPoolGpuStream /*stream*/) const {
    Fail();
  }

  void CopyHostToDeviceAsync(void* /*dst_device*/, const void* /*src_host*/,
                             std::size_t /*bytes*/,
                             EnvPoolGpuStream /*stream*/) const {
    Fail();
  }

  void SynchronizeStream(EnvPoolGpuStream /*stream*/) const { Fail(); }

 private:
  [[noreturn]] static void Fail() {
    LOG(FATAL) << "EnvPool XLA GPU backend is unavailable on this platform.";
  }
};

#endif  // defined(__linux__)

inline const DriverApi& Api() {
  static const DriverApi* api = new DriverApi();
  return *api;
}

}  // namespace envpool::cuda_driver

#endif  // ENVPOOL_CORE_CUDA_DRIVER_H_
