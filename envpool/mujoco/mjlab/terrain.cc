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

#include "envpool/mujoco/mjlab/terrain.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace mjlab {

Terrain::Terrain(Simulation* simulation) : sim_(*simulation) {
  if (!sim_.cfg.as_object().contains("terrain_state")) {
    return;
  }
  const auto& state = sim_.cfg.at("terrain_state");
  const auto& origins = state.at("origins").as_array();
  rows_ = origins.size();
  columns_ = origins[0].as_array().size();
  origins_ = Floats(state.at("origins"));
  const int max_init =
      Param(sim_.cfg.at("terrain"), "max_init_terrain_level", rows_ - 1);
  level_ = sim_.random.Integer(std::min(max_init, rows_ - 1) + 1);
  // The upstream vector scene assigns every terrain column when there are
  // enough worlds, then uses largest remainders for the remaining slots.
  // Exporting a one-world model must not freeze every EnvPool slot on column 0.
  const auto& terrains = sim_.cfg.at("terrain")
                             .at("terrain_generator")
                             .at("sub_terrains")
                             .as_object();
  const int base = sim_.kNumEnvs >= columns_ ? 1 : 0;
  const int remaining = sim_.kNumEnvs - base * columns_;
  double total = 0;
  for (const auto& item : terrains) {
    total += Number(item.value().at("proportion"));
  }
  std::vector<int> counts(columns_, base);
  std::vector<int> order(columns_);
  std::vector<double> remainders(columns_);
  std::iota(order.begin(), order.end(), 0);
  int assigned = base * columns_;
  int column = 0;
  for (const auto& item : terrains) {
    const double ideal =
        Number(item.value().at("proportion")) / total * remaining;
    const int floor = std::floor(ideal);
    counts[column] += floor;
    assigned += floor;
    remainders[column++] = ideal - floor;
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return remainders[a] > remainders[b]; });
  for (int i = 0; i < sim_.kNumEnvs - assigned; ++i) {
    ++counts[order[i]];
  }
  int end = 0;
  for (type_ = 0; type_ < columns_ - 1; ++type_) {
    end += counts[type_];
    if (sim_.kEnvId < end) {
      break;
    }
  }
  sim_.origin = Read<3>(origins_.data() + (level_ * columns_ + type_) * 3);
  Generate();
}

void Terrain::Curriculum(const std::vector<float>& command) {
  if (rows_ == 0) {
    return;
  }
  if (sim_.total_steps > 0) {
    const auto p = sim_.Position(sim_.entities.at("robot").root) - sim_.origin;
    const float distance = Norm(std::array<float, 2>{p[0], p[1]});
    const auto& generator = sim_.cfg.at("terrain").at("terrain_generator");
    if (distance > static_cast<float>(Number(generator.at("size").at(0)) / 2)) {
      ++level_;
    } else if (distance <
               Norm(std::array<float, 2>{command[0], command[1]}) *
                   static_cast<float>(Number(sim_.cfg.at("episode_length_s"))) *
                   0.5F) {
      --level_;
    }
  }
  // torch.where evaluates randint_like even when no environment is promoted.
  const int random_level = sim_.random.Integer(rows_);
  level_ = level_ >= rows_ ? random_level : std::max(level_, 0);
  sim_.origin = Read<3>(origins_.data() + (level_ * columns_ + type_) * 3);
}

bool Terrain::Outside(float margin) const {
  if (rows_ == 0) {
    return false;
  }
  const auto& cfg = sim_.cfg.at("terrain").at("terrain_generator");
  const float x = std::max(0.0, 0.5 * rows_ * Number(cfg.at("size").at(0)) +
                                    Number(cfg.at("border_width")) - margin);
  const float y = std::max(0.0, 0.5 * columns_ * Number(cfg.at("size").at(1)) +
                                    Number(cfg.at("border_width")) - margin);
  const auto pos = sim_.Position(sim_.entities.at("robot").root);
  return std::abs(pos[0]) > x || std::abs(pos[1]) > y;
}

Json Terrain::State() const {
  return boost::json::object{
      {"level", level_}, {"type", type_}, {"origins", ToJson(origins_)}};
}

void Terrain::Generate() {
  // Upstream constructs terrain once, using a separate NumPy Generator. Keep
  // the same categorical height distribution and lifetime, with an independent
  // native stream so terrain size cannot shift the task's Torch-style RNG.
  std::mt19937 rng(sim_.random.Seed());
  auto* model = sim_.physics.Model();
  std::vector<int> changed;
  for (const auto& field :
       sim_.cfg.at("terrain_state").at("random_heightfields").as_array()) {
    const auto& cfg = field.at("cfg");
    const int id = Number(field.at("hfield"));
    const int nr = model->hfield_nrow[id];
    const int nc = model->hfield_ncol[id];
    const double horizontal = Number(cfg.at("horizontal_scale"));
    const double vertical = Number(cfg.at("vertical_scale"));
    const int border = Number(cfg.at("border_width")) / horizontal;
    const int low = Number(cfg.at("noise_range").at(0)) / vertical;
    const int high = Number(cfg.at("noise_range").at(1)) / vertical;
    const int step = Number(cfg.at("noise_step")) / vertical;
    std::uniform_int_distribution<int> choice(0,
                                              (high - low + step - 1) / step);
    std::vector<int> noise(nr * nc);
    for (int r = border; r < nr - border; ++r) {
      for (int c = border; c < nc - border; ++c) {
        noise[r * nc + c] = low + choice(rng) * step;
      }
    }
    const auto [minimum, maximum] =
        std::minmax_element(noise.begin(), noise.end());
    const int range = *maximum == *minimum ? 1 : *maximum - *minimum;
    const double height = range * vertical;
    std::vector<double> physical(noise.size());
    for (std::size_t i = 0; i < noise.size(); ++i) {
      const double normalized =
          static_cast<double>(noise[i] - *minimum) / range;
      model->hfield_data[model->hfield_adr[id] + i] = normalized;
      sim_.physics.Get("model.hfield_data")[model->hfield_adr[id] + i] =
          normalized;
      physical[i] = normalized * height;
    }
    model->hfield_size[id * 4 + 2] = height;
    model->hfield_size[id * 4 + 3] =
        height * Number(cfg.at("base_thickness_ratio"));
    std::copy_n(model->hfield_size + id * 4, 4,
                sim_.physics.Get("model.hfield_size") + id * 4);
    if (sim_.physics.Has("camera.hfield_bounds_size")) {
      sim_.physics.Get("camera.hfield_bounds_size")[id * 3 + 2] = height * 0.5;
    }

    // color_by_height uses scipy.ndimage.zoom(order=1, grid_mode=False),
    // then a fixed HSV map and a vertical flip. Preserve double arithmetic
    // until its final uint8 conversion, independently of float32 physics.
    const int texture = Number(field.at("texture"));
    const int width = model->tex_width[texture];
    const int rows = model->tex_height[texture];
    const int channels = model->tex_nchannel[texture];
    std::vector<uint8_t> pixels(width * rows * channels);
    for (int r = 0; r < rows; ++r) {
      const double y = r * (static_cast<double>(nr - 1) / (rows - 1));
      const int r0 = std::min(static_cast<int>(y), nr - 2);
      const double fy = y - r0;
      for (int c = 0; c < width; ++c) {
        const double x = c * (static_cast<double>(nc - 1) / (width - 1));
        const int c0 = std::min(static_cast<int>(x), nc - 2);
        const double fx = x - c0;
        double z = 0;
        z += physical[r0 * nc + c0] * (1 - fy) * (1 - fx);
        z += physical[r0 * nc + c0 + 1] * (1 - fy) * fx;
        z += physical[(r0 + 1) * nc + c0] * fy * (1 - fx);
        z += physical[(r0 + 1) * nc + c0 + 1] * fy * fx;
        const double signed_height = std::clamp(z / 0.75, -1.0, 1.0);
        const double hue = 0.33 - 0.33 * signed_height;
        const double value = 0.45 + 0.25 * std::abs(signed_height);
        const double chroma = value * value;
        const double secondary =
            chroma * (1 - std::abs(std::fmod(hue * 6, 2) - 1));
        const double m = value - chroma;
        const std::array<std::array<double, 3>, 6> colors{
            {{chroma, secondary, 0},
             {secondary, chroma, 0},
             {0, chroma, secondary},
             {0, secondary, chroma},
             {secondary, 0, chroma},
             {chroma, 0, secondary}}};
        const auto& color = colors[static_cast<int>(hue * 6) % 6];
        const int address = ((rows - 1 - r) * width + c) * channels;
        for (int k = 0; k < 3; ++k) {
          pixels[address + k] = (color[k] + m) * 255;
        }
        if (channels == 4) {
          pixels[address + 3] = 255;
        }
      }
    }
    sim_.physics.UpdateTexture(texture, pixels);
    changed.push_back(id);
  }
  sim_.physics.RebuildHeightfields(changed);
}

}  // namespace mjlab
