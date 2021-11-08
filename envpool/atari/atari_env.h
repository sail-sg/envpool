/*
 * Copyright 2021 Garena Online Private Limited
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

#ifndef ENVPOOL_ATARI_ATARI_ENV_H_
#define ENVPOOL_ATARI_ATARI_ENV_H_

#include <algorithm>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ale_interface.hpp"
#include "envpool/core/async_envpool.h"
#include "envpool/utils/resize.h"

namespace atari {

bool TurnOffVerbosity() {
  ale::Logger::setMode(ale::Logger::Error);
  return true;
}

static bool verbosity_off = TurnOffVerbosity();

std::string GetRomPath(std::string task) {
  // task = task.lower()
  std::transform(task.begin(), task.end(), task.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::stringstream rom_path;
  // hardcode path here :(
  rom_path << "envpool/atari/atari_roms/" << task << "/" << task << ".bin";
  return rom_path.str();
}

class AtariEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.bind(25000), "stack_num"_.bind(4),
                    "frame_skip"_.bind(4), "noop_max"_.bind(30),
                    "zero_discount_on_life_loss"_.bind(false),
                    "episodic_life"_.bind(false), "reward_clip"_.bind(false),
                    "img_height"_.bind(84), "img_width"_.bind(84),
                    "task"_.bind(std::string("pong")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict(
        "obs"_.bind(Spec<uint8_t>(
            {conf["stack_num"_], conf["img_height"_], conf["img_width"_]},
            {0, 255})),
        "discount"_.bind(Spec<float>({-1}, {0.0f, 1.0f})),
        "info:lives"_.bind(Spec<int>({-1}, {0, 5})),
        "reward"_.bind(Spec<float>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    ale::ALEInterface env;
    env.loadROM(GetRomPath(conf["task"_]));
    int action_size = env.getMinimalActionSet().size();
    return MakeDict("action"_.bind(Spec<int>({-1}, {0, action_size - 1})));
  }
};

typedef class EnvSpec<AtariEnvFns> AtariEnvSpec;
typedef Spec<uint8_t> FrameSpec;

class AtariEnv : public Env<AtariEnvSpec> {
 protected:
  const int kRawHeight = 210;
  const int kRawWidth = 160;
  const int kRawSize = kRawWidth * kRawHeight;
  std::unique_ptr<ale::ALEInterface> env_;
  ale::ActionVect action_set_;
  FrameSpec raw_spec_, resize_spec_;
  int max_episode_steps_, elapsed_step_, stack_num_, frame_skip_;
  bool fire_reset_, reward_clip_, zero_discount_on_life_loss_, episodic_life_;
  bool done_;
  int lives_;
  std::deque<Array> stack_buf_;
  std::vector<Array> maxpool_buf_;
  std::uniform_int_distribution<> dist_noop_;

 public:
  AtariEnv(const Spec& spec, int env_id)
      : Env<AtariEnvSpec>(spec, env_id),
        env_(new ale::ALEInterface()),
        raw_spec_({kRawHeight, kRawWidth, 1}),
        resize_spec_(
            {spec.config["img_height"_], spec.config["img_width"_], 1}),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        stack_num_(spec.config["stack_num"_]),
        frame_skip_(spec.config["frame_skip"_]),
        fire_reset_(false),
        reward_clip_(spec.config["reward_clip"_]),
        zero_discount_on_life_loss_(spec.config["zero_discount_on_life_loss"_]),
        episodic_life_(spec.config["episodic_life"_]),
        done_(true),
        dist_noop_(0, spec.config["noop_max"_] - 1) {
    env_->setFloat("repeat_action_probability", 0);
    env_->setInt("random_seed", seed_);
    env_->loadROM(GetRomPath(spec.config["task"_]));
    action_set_ = env_->getMinimalActionSet();
    for (auto a : action_set_) {
      if (a == 1) {
        fire_reset_ = true;
      }
    }
    // init buf
    for (int i = 0; i < 2; ++i) {
      maxpool_buf_.push_back(std::move(Array(raw_spec_)));
    }
    for (int i = 0; i < stack_num_; ++i) {
      stack_buf_.push_back(Array(resize_spec_));
    }
    ResetObsBuffer();
  }

  void Reset() override {
    int noop = dist_noop_(gen_) + 1 - fire_reset_;
    if (env_->game_over() || elapsed_step_ >= max_episode_steps_) {
      env_->reset_game();
      elapsed_step_ = 0;
      ResetObsBuffer();
    }
    while (noop--) {
      env_->act((ale::Action)0);
      if (env_->game_over()) {
        env_->reset_game();
      }
    }
    if (fire_reset_) {
      env_->act((ale::Action)1);
    }
    ale::pixel_t* ale_screen_data = env_->getScreen().getArray();
    uint8_t* ptr = static_cast<uint8_t*>(maxpool_buf_[0].data());
    env_->theOSystem->colourPalette().applyPaletteGrayscale(
        ptr, ale_screen_data, kRawSize);
    PushStack();
    done_ = false;
    State state = Allocate();
    state["discount"_] = 1.0f;
    state["reward"_] = 0.0f;
    state["info:lives"_] = lives_ = env_->lives();
    WriteObs(state);
  }

  void Step(const Action& action) override {
    float reward = 0;
    done_ = false;
    int act = action["action"_];
    for (int skip_id = frame_skip_; skip_id > 0 && !done_; --skip_id) {
      reward += env_->act(action_set_[act]);
      done_ = env_->game_over();
      if (skip_id <= 2) {  // put final two frames in to maxpool buffer
        ale::pixel_t* ale_screen_data = env_->getScreen().getArray();
        uint8_t* ptr = static_cast<uint8_t*>(maxpool_buf_[2 - skip_id].data());
        env_->theOSystem->colourPalette().applyPaletteGrayscale(
            ptr, ale_screen_data, kRawSize);
      }
    }
    MaxPool();    // max pool two buffers into the first one
    PushStack();  // push the maxpool outcome to the stack_buf
    ++elapsed_step_;
    if (reward_clip_) {
      if (reward > 0) {
        reward = 1;
      } else if (reward < 0) {
        reward = -1;
      }
    }
    done_ |= (elapsed_step_ >= max_episode_steps_);
    if (episodic_life_ && env_->lives() < lives_) {
      done_ = true;
    }
    State state = Allocate();
    if (zero_discount_on_life_loss_) {
      state["discount"_] = 1.0f * (lives_ == env_->lives() && !done_);
    } else {
      state["discount"_] = 1.0f - done_;
    }
    state["reward"_] = reward;
    state["info:lives"_] = lives_ = env_->lives();
    WriteObs(state);
  }

  bool IsDone() override { return done_; }

 private:
  void WriteObs(State& state) {  // NOLINT
    for (int i = 0; i < stack_num_; ++i) {
      state["obs"_][i].Assign(stack_buf_[i]);
    }
  }

  void ResetObsBuffer() {
    for (int i = 0; i < stack_num_; ++i) {
      stack_buf_[i].Zero();
    }
  }

  void MaxPool() {
    uint8_t* ptr0 = static_cast<uint8_t*>(maxpool_buf_[0].data());
    uint8_t* ptr1 = static_cast<uint8_t*>(maxpool_buf_[1].data());
    for (int i = 0; i < kRawSize; ++i) {
      ptr0[i] = std::max(ptr0[i], ptr1[i]);
    }
  }

  void PushStack() {
    Array tgt = std::move(*stack_buf_.begin());
    stack_buf_.pop_front();
    Resize(maxpool_buf_[0], &tgt);
    stack_buf_.push_back(std::move(tgt));
  }
};

typedef AsyncEnvPool<AtariEnv> AtariEnvPool;

}  // namespace atari

#endif  // ENVPOOL_ATARI_ATARI_ENV_H_
