/*
 * Copyright 2026 Garena Online Private Limited
 * Copyright 2020 The dm_control Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "envpool/mujoco/locomotion/simulation.h"

namespace mujoco_locomotion {
namespace {

// Cubic B-spline prefilter with mirror boundaries, matching scipy.ndimage.zoom
// (order=3, grid_mode=False). Equations from Unser et al., 1991, as used by
// SciPy's ni_splines.c; its BSD notice is in
// third_party/dmc_locomotion/LICENSE.scipy.
void SplineFilter(std::array<double, 60>* values) {
  auto& c = *values;
  constexpr double pole = -0.267949192431122706472553658494127633;
  const double gain = (1 - pole) * (1 - 1 / pole);
  for (double& value : c) {
    value *= gain;
  }
  const double end = std::pow(pole, 59);
  c[0] += end * c[59];
  double power = pole;
  for (int i = 1; i < 59; ++i) {
    c[0] += power * (c[i] + end * c[59 - i]);
    power *= pole;
  }
  c[0] /= 1 - end * end;
  for (int i = 1; i < 60; ++i) {
    c[i] += pole * c[i - 1];
  }
  c[59] = (pole * c[58] + c[59]) * pole / (pole * pole - 1);
  for (int i = 58; i >= 0; --i) {
    c[i] = pole * (c[i + 1] - c[i]);
  }
}

std::array<double, 4> SplineWeights(double position) {
  const double x = position - std::floor(position);
  const double y = 1 - x;
  std::array<double, 4> weights{y * y * y / 6, (x * x * (x - 2) * 3 + 4) / 6,
                                (y * y * (y - 2) * 3 + 4) / 6, 1};
  for (int i = 0; i < 3; ++i) {
    weights[3] -= weights[i];
  }
  return weights;
}

}  // namespace

void Scene::Bowl() {
  Floor(20, true);
  auto camera = Find(World(), "camera", "top_camera");
  camera.parent().remove_child(camera);
  Set(Find(World(), "geom", "groundplane"), "size", {20, 20, .5});
  auto terrain = Asset().prepend_child("hfield");
  Set(terrain, "name", "terrain");
  Set(terrain, "nrow", 201);
  Set(terrain, "ncol", 201);
  Set(terrain, "size", {6, 6, .5, .1});
  auto geom = World().prepend_child("geom");
  Set(geom, "name", "terrain");
  Set(geom, "type", "hfield");
  Set(geom, "pos", {0, 0, -.01});
  Set(geom, "hfield", "terrain");
  Set(geom, "material", "aesthetic_material");
  ground_geoms.insert(ground_geoms.begin(), "terrain");
  auto map = Child(Child(Root(), "visual"), "map");
  Set(map, "znear", .00025);
  Set(map, "zfar", 50);
}

void Simulation::ResetBowl() {
  std::array<double, 4> quaternion;
  for (double& value : quaternion) {
    value = random_.Normal();
  }
  mju_normalize4(quaternion.data());
  double* qpos = data_->qpos + model_->jnt_qposadr[walkers_[0].freejoint];
  const int flags = model_->opt.disableflags;
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  double height = 0;
  bool clear = false;
  for (int attempt = 0; attempt < 999; ++attempt, height += .01) {
    qpos[0] = qpos[1] = 0;
    qpos[2] = height;
    mju_copy4(qpos + 3, quaternion.data());
    mj_forward(model_.get(), data_.get());
    if (data_->ncon == 0) {
      clear = true;
      break;
    }
  }
  model_->opt.disableflags = flags;
  if (!clear) {
    throw std::runtime_error("Bowl spawn has no contact-free height");
  }
  // The task's initialize hook runs before the arena's. The spawn is found
  // against the initial flat hfield; only then does Bowl generate the terrain.
  GenerateBowl();
}

void Simulation::GenerateBowl() {
  std::array<double, 3600> bumps;
  for (double& value : bumps) {
    value = random_.Uniform(.5, 1);
  }
  std::array<double, 60> line;
  for (int axis = 0; axis < 2; ++axis) {
    for (int i = 0; i < 60; ++i) {
      for (int j = 0; j < 60; ++j) {
        line[j] = bumps[(axis != 0) ? i * 60 + j : j * 60 + i];
      }
      SplineFilter(&line);
      for (int j = 0; j < 60; ++j) {
        bumps[(axis != 0) ? i * 60 + j : j * 60 + i] = line[j];
      }
    }
  }
  std::array<std::array<double, 4>, 201> weights;
  std::array<std::array<int, 4>, 201> indices;
  for (int i = 0; i < 201; ++i) {
    const double coordinate = i * (59.0 / 200);
    weights[i] = SplineWeights(coordinate);
    for (int j = 0; j < 4; ++j) {
      int index = static_cast<int>(std::floor(coordinate)) - 1 + j;
      if (index < 0) {
        index = -index;
      }
      if (index > 59) {
        index = 118 - index;
      }
      indices[i][j] = index;
    }
  }
  for (int y = 0; y < 201; ++y) {
    for (int x = 0; x < 201; ++x) {
      double smooth = 0;
      for (int dy = 0; dy < 4; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
          double value = bumps[indices[y][dy] * 60 + indices[x][dx]];
          value *= weights[y][dy];
          value *= weights[x][dx];
          smooth += value;
        }
      }
      const double px = x * .01 - 1;
      const double py = y * .01 - 1;
      const double radius = std::clamp(std::sqrt(px * px + py * py), .1, 1.);
      const double bowl = .5 - std::cos(2 * std::acos(-1) * radius) / 2;
      model_->hfield_data[y * 201 + x] = bowl * smooth;
    }
  }
}

void Simulation::RandomizeTouchTarget() {
  double* position = model_->geom_pos + 3 * target_geoms_[0];
  position[0] = 1.5 * scene_random_.Uniform(-1, 1);
  position[1] = 1.5 * scene_random_.Uniform(-1, 1);
  position[2] = .14;
  const int texture = Id(mjOBJ_TEXTURE, "target_0_0/target_sphere_init");
  std::fill_n(model_->mat_texid + mjNTEXROLE * target_materials_[0], mjNTEXROLE,
              texture);
  touched_once_ = touched_twice_ = touch_timeout_ = randomize_touch_ = false;
  touch_state_ = 0;
  first_touch_time_ = second_touch_time_ = 0;
  model_dirty_ = true;
}

void Simulation::TouchReward() {
  observed_touch_state_ = touch_state_;
  const double* target = data_->geom_xpos + 3 * target_geoms_[0];
  double closest = 0;
  for (const char* name : {"hand_L", "hand_R"}) {
    const double* hand =
        data_->xpos + 3 * Id(mjOBJ_BODY, walkers_[0].prefix + name);
    double distance = 0;
    for (int i = 0; i < 3; ++i) {
      distance += std::abs(hand[i] - target[i]);
    }
    closest = std::max(closest, std::exp(-3 * distance));
  }
  rewards[0] = .01 * closest * 25;
  if (touch_state_ == 0) {
    if (touched_once_) {
      first_touch_time_ = data_->time;
      touch_state_ = 1;
      rewards[0] += 25;
    }
  } else if (touch_state_ == 1) {
    if (touched_twice_) {
      second_touch_time_ = data_->time;
      touch_state_ = 2;
      const double interval = second_touch_time_ - first_touch_time_;
      if (interval < .8 - .1) {
        touch_timeout_ = true;
        touch_state_ = 3;
      } else if (interval <= .8 + .1) {
        rewards[0] += 25;
      }
    }
    if (data_->time - first_touch_time_ > .8 + .1) {
      touch_timeout_ = true;
      touch_state_ = 4;
      second_touch_time_ = data_->time;
    }
  } else if (touch_timeout_) {
    if (data_->time > second_touch_time_ + 1.2) {
      touch_timeout_ = false;
    }
  } else if (data_->time > second_touch_time_) {
    randomize_touch_ = true;
  }
}

}  // namespace mujoco_locomotion
