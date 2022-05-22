/*
 * Copyright 2022 Garena Online Private Limited
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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/cartpole.py

#ifndef ENVPOOL_MUJOCO_DMC_CARTPOLE_H_
#define ENVPOOL_MUJOCO_DMC_CARTPOLE_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetCartpoleXML(const std::string& base_path,
                           const std::string& task_name) {
  return GetFileContent(base_path, "cartpole.xml");
}

class CartpoleEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(1),
                    "task_name"_.Bind(std::string("balance")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const std::string task_name = conf["task_name"_];
    int npoles;
    if (task_name == "two_poles") {
      npoles = 2;
    } else if (task_name == "three_poles") {
      npoles = 3;
    } else if (task_name == "balance" ||
               task_name == "balance_sparse" ||
               task_name == "swingup" ||
               task_name == "swingup_sparse" ||) {
      npoles = 1;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc cartpole.");
    }
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({2 * npoles + 1})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({npoles + 1})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({npoles + 1})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 1}, {-1.0, 1.0})));
  }
};

using CartpoleEnvSpec = EnvSpec<CartpoleEnvFns>;

class CartpoleEnv : public Env<CartpoleEnvSpec>, public MujocoEnv {
 protected:
  int id_slider_, id_hinge1_;
  int n_poles_;
  bool is_sparse_;
  bool is_swingup_;

 public:
  CartpoleEnv(const Spec& spec, int env_id)
      : Env<CartpoleEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetCartpoleXML(spec.config["base_path"_],
                                 spec.config["task_name"_]),
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]),
        id_slider_(GetQposId(model_, "slider")),
        id_hinge1_(GetQposId(model_, "hinge1")),
        is_swingup_(spec.config["task_name"_] == "swingup" ||
                    spec.config["task_name"_] == "swingup_sparse" ||
                    spec.config["task_name"_] == "two_poles" ||
                    spec.config["task_name"_] == "three_poles"),
        is_sparse_(spec.config["task_name"_] == "balance_sparse" ||
                   spec.config["task_name"_] == "swingup_sparse") {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "two_poles") {
      n_poles_ = 2;
    } else if (task_name == "three_poles") {
      n_poles_ = 3;
    } else if (task_name == "swingup" || task_name == "swingup_sparse" ||
               task_name == "balance" || task_name == "balance_sparse") {
      n_poles_ = 1;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc cartpole.");
    }
  }

  void TaskInitializeEpisode() override {
    if (is_swingup_) {
      data_->qpos[id_slider_] = RandNormal(0, 1)(gen_) * 0.01;
      data_->qpos[id_hinge1_] = RandNormal(0, 1)(gen_) * 0.01 + M_PI;
      for (int i = 2; i < model_->nv; ++i) {
        data_->qpos[i] = RandUniform(0, 1)(gen_) * 0.01;
      }
    } else {
      data_->qpos[id_slider_] = RandUniform(0, 1)(gen_) * 0.2 - 0.1;
      for (int i = 1; i < model_->nv; ++i) {
        data_->qpos[i] = RandUniform(0, 1)(gen_) * 0.068 - 0.034;
      }
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = RandNormal(0, 1)(gen_);
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    ControlReset();
    WriteState();
  }

  void Step(const Action& action) override {
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState();
  }

  float TaskGetReward() override {
    const auto& pole_angle_cosine = PoleAngleCosine();
    if (is_sparse_) {
      auto cart_in_bounds = RewardTolerance(CartPosition(), -0.25, 0.25);
      mjtNum angle_in_bounds = 1.0;
      for (int i = 0; i < n_poles_; ++i) {
        angle_in_bounds *= RewardTolerance(pole_angle_cosine[i], 0.995, 1);
      }
      return static_cast<float>(cart_in_bounds * angle_in_bounds);
    }
    mjtNum upright = 0.0;
    for (int i = 0; i < n_poles_; ++i) {
      upright += pole_angle_cosine[i];
    }
    upright = (upright / n_poles_ + 1) / 2;
    auto centered = RewardTolerance(CartPosition(), 0.0, 0.0, 2);
    centered = (1 + centered) / 2;
    auto small_control = RewardTolerance(data_->ctrl[0], 0.0, 0.0, 1.0, 0.0,
                                         SigmoidType::kQuadratic);
    small_control = (small_control / model_->nu + 4) / 5;
    mjtNum small_velocity;
    const auto& angular_vel = AngularVel();
    small_velocity = angular_vel[0];
    for (int i = 0; i < n_poles_; ++i) {
      auto x = RewardTolerance(angular_vel[i], 0.0, 0.0, 5.0, 0.1,
                               SigmoidType::kQuadratic);
      small_velocity = std::min(x, small_velocity);
    }
    small_velocity = (small_velocity + 1) / 2;
    return static_cast<float>(upright * small_control * small_velocity *
                              centered);
  }
  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    const auto& position = BoundedPosition();
    state["obs:position"_].Assign(position.begin(), position.size());
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    // info for check alignment
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }

  mjtNum CartPosition() {
    // return self.named.data.qpos['slider'][0]
    return data_->qpos[id_slider_];
  }
  std::array<mjtNum, 3> AngularVel() {
    // return self.data.qvel[1:]
    std::array<mjtNum, 3> angular_vel;
    for (int i = 0; i < n_poles_; ++i) {
      angular_vel[i] = data_->qvel[1 + i];
    }
    for (int i = n_poles_; i < 3; ++i) {
      angular_vel[i] = std::numeric_limits<double>::infinity();
    }
    return angular_vel;
  }
  std::array<mjtNum, 3> PoleAngleCosine() {
    // return self.named.data.xmat[2:, 'zz']
    std::array<mjtNum, 3> pole_angle_cosine;
    for (int i = 0; i < n_poles_; ++i) {
      pole_angle_cosine[i] = data_->xmat[(2 + i) * 9 + 8];
    }
    for (int i = n_poles_; i < 3; ++i) {
      pole_angle_cosine[i] = std::numeric_limits<double>::infinity();
    }
    return pole_angle_cosine;
  }
  std::array<mjtNum, 7> BoundedPosition() {
    // return np.hstack((self.cart_position(),
    //                   self.named.data.xmat[2:, ['zz', 'xz']].ravel()))
    std::array<mjtNum, 7> bounded_position;
    bounded_position[0] = CartPosition();
    for (int i = 0; i < n_poles_; ++i) {
      bounded_position[i * 2] = data_->xmat[(2 + i) * 9 + 8];
      bounded_position[i * 2 + 1] = data_->xmat[(2 + i) * 9 + 2];
    }
    for (int i = n_poles_; i < 3; ++i) {
      bounded_position[i * 2] = std::numeric_limits<double>::infinity();
      bounded_position[i * 2 + 1] = std::numeric_limits<double>::infinity();
    }
    return bounded_position;
  }
};

using CartpoleEnvPool = AsyncEnvPool<CartpoleEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_CARTPOLE_H_
