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

#ifndef ENVPOOL_MUJOCO_MJLAB_MJLAB_ENV_H_
#define ENVPOOL_MUJOCO_MJLAB_MJLAB_ENV_H_

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mjlab/simulation.h"
#include "envpool/mujoco/offscreen_renderer.h"
#include "third_party/mjlab/generated/registry.h"

namespace mjlab {

inline const Json& TaskInfo(const std::string& name) {
  static const Json registry = boost::json::parse(kRegistry);
  for (const auto& task : registry.as_array()) {
    if (String(task.at("id")) == name) {
      return task;
    }
  }
  throw std::invalid_argument("unknown MJLab task: " + name);
}

inline int ObservationSize(const Json& task) {
  int size = 0;
  for (const auto& entry : task.at("observation_shapes").as_object()) {
    const auto shape = Indices(entry.value());
    size += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  }
  return size;
}

struct MjlabEnvFns {
  static auto DefaultConfig() {
    return MakeDict("task_name"_.Bind(std::string("Mjlab-Cartpole-Balance")),
                    "motion_file"_.Bind(std::string("")));
  }

  template <typename Config>
  static auto StateSpec(const Config& config) {
    return MakeDict("obs"_.Bind(Spec<float>(
                        {-1, ObservationSize(TaskInfo(config["task_name"_]))})),
                    "terminated"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static auto ActionSpec(const Config& config) {
    const int size = Number(TaskInfo(config["task_name"_]).at("action_size"));
    const float infinity = std::numeric_limits<float>::infinity();
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, size}, {-infinity, infinity})));
  }
};

class MjlabEnvSpec : public EnvSpec<MjlabEnvFns> {
  using Base = EnvSpec<MjlabEnvFns>;

 public:
  explicit MjlabEnvSpec(const ConfigValues& values) : Base(Normalize(values)) {}
  MjlabEnvSpec() : MjlabEnvSpec(kDefaultConfig.AllValues()) {}

 private:
  static ConfigValues Normalize(const ConfigValues& values) {
    Config config(values);
    const auto& task = TaskInfo(config["task_name"_]);
    if (config["max_num_players"_] != 1 || config["max_episode_steps"_] <= 0) {
      throw std::invalid_argument(
          "MJLab requires one player and a positive episode limit");
    }
    if (config["task_name"_].find("Mjlab-Tracking-") == 0 &&
        config["motion_file"_].empty()) {
      throw std::invalid_argument(
          "MJLab tracking requires motion_file, an official-format NPZ motion");
    }
    config["max_episode_steps"_] =
        std::min(config["max_episode_steps"_],
                 static_cast<int>(Number(task.at("max_episode_steps"))));
    return config.AllValues();
  }
};

class MjlabEnv : public Env<MjlabEnvSpec>, public RenderableEnv {
  using Base = Env<MjlabEnvSpec>;

 public:
  MjlabEnv(const Spec& spec, int env_id)
      : Base(spec, env_id),
        simulation_(spec.config["base_path"_] + "/mujoco/mjlab/assets/" +
                        String(TaskInfo(spec.config["task_name"_]).at("asset")),
                    seed_, spec.config["max_episode_steps"_],
                    spec.config["motion_file"_], env_id,
                    spec.config["num_envs"_]) {}

  bool IsDone() override {
    return simulation_.terminated || simulation_.truncated;
  }

  void Reset() override {
    std::scoped_lock lock(mutex_);
    simulation_.Reset();
    WriteState();
  }

  void Step(const Action& action) override {
    std::scoped_lock lock(mutex_);
    simulation_.Step(static_cast<const float*>(action["action"_].Data()));
    WriteState();
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    const auto& viewer = simulation_.cfg.at("viewer");
    return {
        width > 0 ? width : static_cast<int>(Number(viewer.at("width"))),
        height > 0 ? height : static_cast<int>(Number(viewer.at("height")))};
  }

  void Render(int width, int height, int camera,
              unsigned char* output) override {
    std::scoped_lock lock(mutex_);
    if (!simulation_.initialized) {
      throw std::runtime_error("reset before rendering");
    }
    if (!renderer_) {
      InitializeRenderer();
    }
    renderer_->Render(simulation_.physics.Model(),
                      simulation_.physics.RenderData(), width, height, camera,
                      output, &camera_, &option_, true, true);
  }

#ifdef ENVPOOL_TEST
  template <typename Function>
  auto Inspect(Function function) {
    std::scoped_lock lock(mutex_);
    if (!simulation_.initialized) {
      throw std::runtime_error("reset before inspecting physics");
    }
    return function(simulation_);
  }
#endif

 private:
  void WriteState() {
    State state = Allocate();
    auto* output = static_cast<float*>(state["obs"_].Data());
    // std::map order is also used by the Python dictionary-view adapter.
    for (const auto& entry : simulation_.observations) {
      output = std::copy(entry.second.begin(), entry.second.end(), output);
    }
    state["reward"_] = simulation_.reward;
    state["discount"_] = simulation_.terminated ? 0.0F : 1.0F;
    state["terminated"_] = simulation_.terminated;
    state["trunc"_] = simulation_.truncated;
  }

  void InitializeRenderer() {
    auto* model = simulation_.physics.Model();
    const auto& viewer = simulation_.cfg.at("viewer");
    // Like the official offscreen viewer, the host model is a render/index
    // copy. Physics continues to use its own immutable Warp model buffers.
    std::fill_n(model->body_sameframe, model->nbody, mjSAMEFRAME_NONE);
    std::fill_n(model->geom_sameframe, model->ngeom, mjSAMEFRAME_NONE);
    std::fill_n(model->site_sameframe, model->nsite, mjSAMEFRAME_NONE);
    std::fill_n(model->body_simple, model->nbody, 0);
    model->stat.extent = std::max(4.0, 1.5 * Number(viewer.at("distance")));
    if (!viewer.at("fovy").is_null()) {
      model->vis.global.fovy = Number(viewer.at("fovy"));
    }
    mjv_defaultFreeCamera(model, &camera_);
    mjv_defaultOption(&option_);
    camera_.type = mjCAMERA_FREE;
    camera_.trackbodyid = -1;
    camera_.fixedcamid = -1;
    const auto origin = String(viewer.at("origin_type"));
    if (origin == "ASSET_ROOT" || origin == "ASSET_BODY") {
      const auto name =
          Name(viewer, "entity_name", simulation_.entities.begin()->first);
      camera_.type = mjCAMERA_TRACKING;
      camera_.trackbodyid =
          origin == "ASSET_ROOT"
              ? simulation_.entities.at(name).root
              : mj_name2id(model, mjOBJ_BODY,
                           (name + "/" + Name(viewer, "body_name")).c_str());
      if (camera_.trackbodyid < 0) {
        throw std::runtime_error("invalid pinned viewer body");
      }
    }
    camera_.distance = Number(viewer.at("distance"));
    camera_.elevation = Number(viewer.at("elevation"));
    camera_.azimuth = Number(viewer.at("azimuth"));
    for (int i = 0; i < 3; ++i) {
      camera_.lookat[i] = Number(viewer.at("lookat").at(i));
    }
    for (int i = 0; i < 6; ++i) {
      option_.geomgroup[i] = Number(viewer.at("geom_group").at(i));
      option_.sitegroup[i] = Number(viewer.at("site_group").at(i));
    }
    renderer_ = std::make_unique<envpool::mujoco::OffscreenRenderer>();
  }

  Simulation simulation_;
  std::mutex mutex_;
  mjvCamera camera_{};
  mjvOption option_{};
  std::unique_ptr<envpool::mujoco::OffscreenRenderer> renderer_;
};

class MjlabEnvPool : public AsyncEnvPool<MjlabEnv> {
  using Base = AsyncEnvPool<MjlabEnv>;

 public:
  using Base::Base;
#ifdef ENVPOOL_TEST
  template <typename Function>
  auto Inspect(int env_id, Function function) {
    if (env_id < 0 || env_id >= static_cast<int>(envs_.size())) {
      throw std::out_of_range("invalid env_id");
    }
    return envs_[env_id]->Inspect(function);
  }
#endif
};

}  // namespace mjlab

#endif  // ENVPOOL_MUJOCO_MJLAB_MJLAB_ENV_H_
