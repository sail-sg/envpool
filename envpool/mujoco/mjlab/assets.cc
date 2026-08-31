// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/mujoco/mjlab/assets.h"

#include <zstd.h>

#include <algorithm>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/mujoco/mjlab/physics.h"

namespace mjlab {
namespace {

constexpr std::size_t kMaxAssetBytes = 512 * 1024 * 1024;

std::vector<uint8_t> ReadBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto length = input.tellg();
  if (!input || length <= 0 || length > kMaxAssetBytes) {
    throw std::runtime_error("missing or invalid MJLab asset: " +
                             path.string());
  }
  std::vector<uint8_t> result(static_cast<std::size_t>(length));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char*>(result.data()), length)) {
    throw std::runtime_error("truncated MJLab asset: " + path.string());
  }
  return result;
}

std::vector<uint8_t> ReadBlob(const std::filesystem::path& root,
                              const std::string& digest,
                              std::size_t expected_size) {
  if (digest.size() != 64 ||
      digest.find_first_not_of("0123456789abcdef") != std::string::npos ||
      expected_size == 0 || expected_size > kMaxAssetBytes) {
    throw std::runtime_error("invalid MJLab shared asset reference");
  }
  const auto encoded = ReadBytes(root / (digest + ".zst"));
  if (ZSTD_getFrameContentSize(encoded.data(), encoded.size()) !=
      expected_size) {
    throw std::runtime_error("invalid MJLab shared asset size");
  }
  std::vector<uint8_t> result(expected_size);
  const auto size = ZSTD_decompress(result.data(), result.size(),
                                    encoded.data(), encoded.size());
  if (ZSTD_isError(size) != 0 || size != expected_size) {
    throw std::runtime_error("corrupt MJLab shared asset: " + digest);
  }
  return result;
}

}  // namespace

std::shared_ptr<const std::vector<uint8_t>> LoadAsset(const std::string& name) {
  const auto path = std::filesystem::absolute(name).lexically_normal();
  static std::mutex mutex;
  static std::map<std::filesystem::path,
                  std::weak_ptr<const std::vector<uint8_t>>>
      cache;
  std::scoped_lock lock(mutex);
  if (auto value = cache[path].lock()) {
    return value;
  }
  std::vector<uint8_t> bytes;
  const auto manifest = path.string() + ".json";
  if (std::filesystem::exists(manifest)) {
    const auto index = ReadJson(manifest);
    const std::size_t size = index.at("size").to_number<std::size_t>();
    const auto root = path.parent_path().parent_path() / "shared";
    bytes = ReadBlob(root, String(index.at("remainder")), size);
    std::size_t end = 0;
    for (const auto& piece : index.at("pieces").as_array()) {
      const auto offset = piece.at("offset").to_number<std::size_t>();
      const auto count = piece.at("size").to_number<std::size_t>();
      if (offset < end || offset > size || count > size - offset) {
        throw std::runtime_error("invalid MJLab asset slice");
      }
      auto data = ReadBlob(root, String(piece.at("blob")), count);
      std::copy(data.begin(), data.end(), bytes.begin() + offset);
      end = offset + count;
    }
  } else {
    // Also accept the exporter's unpacked files during development.
    bytes = ReadBytes(path);
  }
  auto result = std::make_shared<const std::vector<uint8_t>>(std::move(bytes));
  cache[path] = result;
  return result;
}

}  // namespace mjlab
