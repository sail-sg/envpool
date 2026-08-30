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

#include "envpool/mujoco/mjlab/physics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "apic.h"
#include "apic_internal.h"
#include "envpool/mujoco/mjlab/assets.h"
#include "warp.h"

namespace mjlab {

using Kernel = void (*)(void*, void*);
Kernel LookupKernel(const std::string& key);

namespace {

std::mutex& ResourceMutex() {
  // Warp's host resource descriptor registries are process-wide. Simulation
  // kernels dereference per-world resources directly and do not take this lock.
  static std::mutex mutex;
  return mutex;
}

template <typename T>
wp::array_t<T> Array(T* data, std::size_t count) {
  wp::array_t<T> result{};
  result.data = data;
  result.ndim = 1;
  result.shape[0] = static_cast<int>(count);
  result.strides[0] = sizeof(T);
  return result;
}

}  // namespace

Json ReadJson(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open MJLab metadata: " + path);
  }
  return boost::json::parse(std::string(std::istreambuf_iterator<char>(file),
                                        std::istreambuf_iterator<char>()));
}

double Number(const Json& value) {
  if (value.is_string()) {
    const auto& text = value.as_string();
    if (text == "Infinity") {
      return std::numeric_limits<double>::infinity();
    }
    if (text == "-Infinity") {
      return -std::numeric_limits<double>::infinity();
    }
    throw std::invalid_argument("expected numeric MJLab metadata");
  }
  return value.to_number<double>();
}

std::string String(const Json& value) { return std::string(value.as_string()); }

std::vector<int> Indices(const Json& value) {
  std::vector<int> result;
  for (const auto& item : value.as_array()) {
    result.push_back(static_cast<int>(Number(item)));
  }
  return result;
}

std::vector<float> Floats(const Json& value) {
  std::vector<float> result;
  const auto append = [&](const auto& self, const Json& item) -> void {
    if (item.is_array()) {
      for (const auto& child : item.as_array()) {
        self(self, child);
      }
    } else {
      result.push_back(static_cast<float>(Number(item)));
    }
  };
  append(append, value);
  return result;
}

struct Physics::Impl {
  Json metadata;
  std::shared_ptr<const std::vector<uint8_t>> model_bytes, graph_bytes;
  std::unique_ptr<APICGraph, decltype(&wp_apic_destroy_graph)> graph{
      nullptr, &wp_apic_destroy_graph};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model{nullptr,
                                                            &mj_deleteModel};
  std::unique_ptr<mjData, decltype(&mj_deleteData)> render_data{nullptr,
                                                                &mj_deleteData};
  uint64_t bvh{0};
  std::map<int, uint64_t> meshes, heightfields;
  std::vector<uint64_t> textures;
  std::map<int, float*> texture_pixels;
  // Runtime terrain meshes own their buffers for the lifetime of the BVH.
  struct Heightfield {
    std::vector<wp::vec3> points;
    std::vector<int> indices;
  };
  std::map<int, Heightfield> terrain;

  ~Impl() {
    std::scoped_lock lock(ResourceMutex());
    if (bvh != 0) {
      wp_bvh_destroy_host(bvh);
    }
    for (auto& entry : meshes) {
      wp_mesh_destroy_host(entry.second);
    }
    for (auto& entry : heightfields) {
      wp_mesh_destroy_host(entry.second);
    }
    for (auto texture : textures) {
      wp_texture_destroy_host(texture);
    }
    graph.reset();
  }
};

Physics::Physics(const std::string& asset_path)
    : impl_(std::make_unique<Impl>()) {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    if (wp_init("1.14.0") != 0) {
      throw std::runtime_error("cannot initialize native Warp 1.14.0");
    }
  });
  impl_->metadata = ReadJson(asset_path + "/task.json");
  impl_->model_bytes = LoadAsset(asset_path + "/model.mjb");
  impl_->graph_bytes = LoadAsset(asset_path + "/physics.wrp");
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  const int added =
      mj_addBufferVFS(&vfs, "model.mjb", impl_->model_bytes->data(),
                      impl_->model_bytes->size());
  if (added == 0) {
    impl_->model.reset(mj_loadModel("model.mjb", &vfs));
  }
  mj_deleteVFS(&vfs);
  if (!impl_->model) {
    throw std::runtime_error("cannot load MJLab model: " + asset_path);
  }
  impl_->render_data.reset(mj_makeData(impl_->model.get()));
  std::scoped_lock lock(ResourceMutex());
  impl_->graph.reset(wp_apic_load_cpu_graph_from_memory(
      impl_->graph_bytes->data(), impl_->graph_bytes->size()));
  auto* graph = impl_->graph.get();
  if (graph == nullptr) {
    throw std::runtime_error("cannot load native MJLab graph: " + asset_path);
  }
  for (int i = 0; i < wp_apic_get_num_kernels(graph); ++i) {
    const char* key = wp_apic_get_kernel_key(graph, i);
    Kernel kernel = LookupKernel(key);
    if (kernel == nullptr) {
      throw std::runtime_error("missing compiled MJLab kernel: " +
                               std::string(key));
    }
    wp_apic_register_loaded_cpu_kernel(
        graph, key, reinterpret_cast<void*>(kernel), nullptr);
  }
  for (const auto& value : Metadata().at("resources").as_array()) {
    const std::string kind = String(value.at("kind"));
    if (kind == "bvh") {
      // Build the same tree as upstream, before its first reset. Rebuilding
      // from a randomized pose changes traversal order and therefore which
      // surface wins an exactly equal float32 ray-distance tie (cube/floor).
      // Sense() subsequently refits these buffers for each live pose.
      for (const std::string axis : {"lower", "upper"}) {
        Set("camera." + axis, Pointer("resource.scene." + axis),
            Bytes("resource.scene." + axis));
      }
      impl_->bvh = wp_bvh_create_host(
          Get<wp::vec3>("camera.lower"), Get<wp::vec3>("camera.upper"),
          static_cast<int>(Number(value.at("count"))), 0,
          Get<int>("camera.group"), 1);
      // APIC's public handle relocation does not expose a setter in 1.14. This
      // narrowly scoped use of the pinned internal ABI relocates a scene BVH;
      // no raw address from the exporting process is used at runtime.
      graph->handle_ptr_remap[1] = impl_->bvh;
      *Get<int>("camera.group_root") = wp::bvh_get_group_root(impl_->bvh, 0);
    } else if (kind == "mesh" || kind == "hfield") {
      const std::string binding = String(value.at("binding"));
      const int index = static_cast<int>(Number(value.at("index")));
      auto points = Array(Get<wp::vec3>(binding + ".points"),
                          Count<wp::vec3>(binding + ".points"));
      auto indices = Array(Get<int>(binding + ".indices"),
                           Count<int>(binding + ".indices"));
      const auto mesh = wp_mesh_create_host(
          points, {}, indices, points.shape[0], indices.shape[0] / 3, 0, 0,
          nullptr, static_cast<int>(Number(value.at("leaf_size"))));
      (kind == "mesh" ? impl_->meshes : impl_->heightfields)[index] = mesh;
      Get<uint64_t>("camera." + kind + "_bvh_id")[index] = mesh;
    } else if (kind == "texture") {
      const std::string binding = String(value.at("binding"));
      const int width = static_cast<int>(Number(value.at("width")));
      const int height = static_cast<int>(Number(value.at("height")));
      const int channels = static_cast<int>(Number(value.at("channels")));
      const int index = static_cast<int>(Number(value.at("index")));
      std::array<int, 3> shape{width, height, 1};
      auto address = Indices(value.at("address"));
      void* pixels = nullptr;
      const auto texture = wp_texture_create_host(
          2, shape.data(), channels,
          static_cast<int>(Number(value.at("dtype"))),
          static_cast<int>(Number(value.at("filter"))), address.data(),
          value.at("normalized").as_bool(), &pixels);
      if (texture == 0 || pixels == nullptr) {
        throw std::runtime_error("cannot allocate native MJLab texture");
      }
      impl_->textures.push_back(texture);
      impl_->texture_pixels[index] = static_cast<float*>(pixels);
      std::memcpy(pixels, Pointer(binding), Bytes(binding));
      Get<wp::texture2d_t>("camera.textures")[index] =
          wp::texture2d_t(texture, width, height, channels);
    } else {
      throw std::runtime_error("unknown MJLab resource: " + kind);
    }
  }
}

Physics::~Physics() = default;
const Json& Physics::Metadata() const { return impl_->metadata; }
mjModel* Physics::Model() const { return impl_->model.get(); }

bool Physics::Has(const std::string& name) const {
  return wp_apic_get_param_ptr(impl_->graph.get(), name.c_str()) != nullptr;
}

std::size_t Physics::Bytes(const std::string& name) const {
  return wp_apic_get_param_size(impl_->graph.get(), name.c_str());
}

void* Physics::Pointer(const std::string& name) const {
  void* result = wp_apic_get_param_ptr(impl_->graph.get(), name.c_str());
  if (result == nullptr) {
    throw std::out_of_range("missing MJLab buffer: " + name);
  }
  return result;
}

void Physics::Set(const std::string& name, const void* value,
                  std::size_t bytes) {
  if (bytes != Bytes(name) ||
      !wp_apic_set_param(impl_->graph.get(), name.c_str(), value, bytes)) {
    throw std::invalid_argument("invalid MJLab buffer size: " + name);
  }
}

void Physics::Run(const std::string& operation) {
  auto* flag = Get<int>("op." + operation);
  *flag = 1;
  const bool success = wp_apic_cpu_replay_graph(impl_->graph.get());
  *flag = 0;
  if (!success) {
    throw std::runtime_error("MJLab native operation failed: " + operation);
  }
}

void Physics::Sense() {
  if (impl_->bvh != 0) {
    Run("bounds");
    wp_bvh_refit_host(impl_->bvh);
    Run("sense");
  }
}

mjData* Physics::RenderData() {
  auto* model = impl_->model.get();
  auto* data = impl_->render_data.get();
  // The pinned viewer copies entire expanded fields from float32 Warp, even
  // elements untouched by randomization. Inertial offsets and subtree masses
  // affect the tracking camera; retaining their original doubles shifts pixels.
  for (const auto& field : impl_->metadata.at("expanded_fields").as_array()) {
    if (field == "body_ipos") {
      std::copy_n(Get("model.body_ipos"), model->nbody * 3, model->body_ipos);
    } else if (field == "body_subtreemass") {
      std::copy_n(Get("model.body_subtreemass"), model->nbody,
                  model->body_subtreemass);
    } else if (field == "geom_rgba") {
      std::copy_n(Get("model.geom_rgba"), model->ngeom * 4, model->geom_rgba);
    }
  }
  std::copy_n(Get("data.qpos"), model->nq, data->qpos);
  std::copy_n(Get("data.qvel"), model->nv, data->qvel);
  if (model->nu != 0) {
    std::copy_n(Get("data.ctrl"), model->nu, data->ctrl);
  }
  if (model->nmocap != 0) {
    std::copy_n(Get("data.mocap_pos"), model->nmocap * 3, data->mocap_pos);
    std::copy_n(Get("data.mocap_quat"), model->nmocap * 4, data->mocap_quat);
  }
  data->time = *Get("data.time");
  mj_forward(model, data);
  return data;
}

void Physics::RebuildHeightfields(const std::vector<int>& ids) {
  // MuJoCo-Warp bvh._optimize_hfield_mesh: merge coplanar rectangles in row
  // order, retaining its vertex and triangle ordering for exact ray results.
  std::scoped_lock lock(ResourceMutex());
  auto* model = Model();
  for (auto& entry : impl_->heightfields) {
    const int id = entry.first;
    if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
      continue;
    }
    wp_mesh_destroy_host(entry.second);
    auto& mesh = impl_->terrain[id];
    mesh.points.clear();
    mesh.indices.clear();
    const int nr = model->hfield_nrow[id];
    const int nc = model->hfield_ncol[id];
    const float* heights = Get("model.hfield_data") + model->hfield_adr[id];
    const float sx = model->hfield_size[id * 4];
    const float sy = model->hfield_size[id * 4 + 1];
    const float sz = model->hfield_size[id * 4 + 2];
    std::vector<bool> visited((nr - 1) * (nc - 1));
    std::map<std::pair<int, int>, int> vertices;
    const auto vertex = [&](int r, int c) {
      const auto key = std::make_pair(r, c);
      auto found = vertices.find(key);
      if (found != vertices.end()) {
        return found->second;
      }
      const int index = static_cast<int>(mesh.points.size());
      vertices[key] = index;
      mesh.points.emplace_back(
          sx * static_cast<float>(static_cast<double>(c) / (0.5 * (nc - 1)) -
                                  1.0),
          sy * static_cast<float>(static_cast<double>(r) / (0.5 * (nr - 1)) -
                                  1.0),
          heights[r * nc + c] * sz);
      return index;
    };
    const auto planar = [&](int r, int c, int row, int col, float dx,
                            float dy) {
      const float z = heights[r * nc + c];
      const float x = heights[r * nc + c + 1];
      const float y = heights[(r + 1) * nc + c];
      const float xy = heights[(r + 1) * nc + c + 1];
      return !visited[r * (nc - 1) + c] &&
             std::abs((z + xy) - (x + y)) < 1.0e-5F &&
             std::abs(z - ((heights[row * nc + col] + (r - row) * dy) +
                           (c - col) * dx)) < 1.0e-5F &&
             std::abs((x - z) - dx) < 1.0e-5F &&
             std::abs((y - z) - dy) < 1.0e-5F;
    };
    for (int r = 0; r < nr - 1; ++r) {
      for (int c = 0; c < nc - 1; ++c) {
        if (visited[r * (nc - 1) + c]) {
          continue;
        }
        const float dx = heights[r * nc + c + 1] - heights[r * nc + c];
        const float dy = heights[(r + 1) * nc + c] - heights[r * nc + c];
        int width = 1;
        int height = 1;
        if (planar(r, c, r, c, dx, dy)) {
          while (c + width < nc - 1 && planar(r, c + width, r, c, dx, dy)) {
            ++width;
          }
          while (r + height < nr - 1) {
            bool extend = true;
            for (int x = c; x < c + width; ++x) {
              extend = extend && planar(r + height, x, r, c, dx, dy);
            }
            if (!extend) {
              break;
            }
            ++height;
          }
        }
        for (int y = r; y < r + height; ++y) {
          for (int x = c; x < c + width; ++x) {
            visited[y * (nc - 1) + x] = true;
          }
        }
        const int tl = vertex(r, c);
        const int tr = vertex(r, c + width);
        const int bl = vertex(r + height, c);
        const int br = vertex(r + height, c + width);
        mesh.indices.insert(mesh.indices.end(), {tl, tr, br, tl, br, bl});
      }
    }
    entry.second = wp_mesh_create_host(
        Array(mesh.points.data(), mesh.points.size()), {},
        Array(mesh.indices.data(), mesh.indices.size()), mesh.points.size(),
        mesh.indices.size() / 3, 0, 0, nullptr, 2);
    Get<uint64_t>("camera.hfield_bvh_id")[id] = entry.second;
  }
}

void Physics::UpdateTexture(int id, const std::vector<uint8_t>& pixels) {
  auto* model = Model();
  const int channels = model->tex_nchannel[id];
  const int count = model->tex_width[id] * model->tex_height[id];
  if (pixels.size() != static_cast<std::size_t>(count) * channels) {
    throw std::invalid_argument("invalid MJLab texture size");
  }
  std::copy(pixels.begin(), pixels.end(), model->tex_data + model->tex_adr[id]);
  const auto found = impl_->texture_pixels.find(id);
  if (found != impl_->texture_pixels.end()) {
    for (int p = 0; p < count; ++p) {
      for (int c = 0; c < 4; ++c) {
        found->second[p * 4 + c] =
            c < channels ? static_cast<float>(pixels[p * channels + c]) / 255.0F
                         : 1.0F;
      }
    }
  }
}

}  // namespace mjlab
