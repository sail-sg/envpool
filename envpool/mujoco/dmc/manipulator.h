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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/manipulator.py

#ifndef ENVPOOL_MUJOCO_DMC_MANIPULATOR_H_
#define ENVPOOL_MUJOCO_DMC_MANIPULATOR_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetManipulatorXML(const std::string& base_path,
                              const std::string& task_name) {
  return GetFileContent(base_path, "manipulator.xml");
}

class ManipulatorEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(10),
                    "task_name"_.Bind(std::string("bring_ball")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:arm_pos"_.Bind(Spec<mjtNum>({8, 2})),
                    "obs:arm_vel"_.Bind(Spec<mjtNum>({8}),
                    "obs:touch"_.Bind(Spec<mjtNum>({5}),
                    "obs:hand_pos"_.Bind(Spec<mjtNum>({4}),
                    "obs:object_pos"_.Bind(Spec<mjtNum>({4}),
                    "obs:object_vel"_.Bind(Spec<mjtNum>({3}),
                    "obs:target_pos"_.Bind(Spec<mjtNum>({4}),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({14})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 5}, {-1.0, 1.0})));
  }
};

using ManipulatorEnvSpec = EnvSpec<ManipulatorEnvFns>;

class ManipulatorEnv : public Env<ManipulatorEnvSpec>, public MujocoEnv {
 protected:
  const mjtNum kClose = 0.01;
  const mjtNum kPInHand = 0.1;
  const mjtNum kPInTarget = 0.1;
  bool use_peg = false;
  bool insert = false;
  std::set<std::string> kArmJoints = {"arm_root",  "arm_shoulder", "arm_elbow",
                                      "arm_wrist", "finger",       "fingertip",
                                      "thumb",     "thumbtip"};
  std::set<std::string> kAllProps = {"ball", "target_ball", "cup",
                                     "peg",  "target_peg",  "slot"};
  std::set<std::string> kTouchSensors = {"palm_touch", "finger_touch",
                                         "thumb_touch", "fingertip_touch",
                                         "thumbtip_touch"};
  std::uniform_real_distribution<> dist_uniform_;

 public:
  ManipulatorEnv(const Spec& spec, int env_id)
      : Env<ManipulatorEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetManipulatorXML(spec.config["base_path"_],
                                    spec.config["task_name"_]),
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]),
        dist_uniform_(0, 1) {}

  void TaskInitializeEpisode() override {
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

  float TaskGetReward() override {}

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  std::string MakeManipulatorModel(const std::string& task_name) {
    if (task_name == "bring_ball") {
      use_peg = false;
      insert = false;
    } else if (task_name == "bring_peg") {
      use_peg = true;
      insert = false;
    } else if (task_name == "insert_ball") {
      use_peg = false;
      insert = true;
    } else if (task_name == "insert_peg") {
      use_peg = true;
      insert = true;
    } else {
      throw std::runtime_error("Unknown task_name for dmc hopper.");
    }
    std::set<std::string> required_props;
    if (use_peg) {
      required_props.insert("peg");
      required_props.insert("target_peg");
      if (insert) {
        required_props.insert("slot");
      }
    } else {
      required_props.insert("ball");
      required_props.insert("target_ball");
      if (insert) {
        required_props.insert("cup");
      }
    }
    std::string content = GetManipulatorXML();
    for (set<int>::iterator it = kAllProps.begin(); it != kAllProps.end();
         it++) {
      if (required_props.find(*it) == required_props.end()) {
        content = ReplaceRegex(content, *it);
      }
    }
    return content;
  }
  std::string ReplaceRegex(const std::string& content,
                           std::string& unused_prop) {
    std::ostringstream pattern_ss;
    pattern_ss << "<body name=\"" << unused_prop
               << "\"((?!</body>)[\\s\\S])+</body>";
    std::regex pattern(pattern_ss.str());
    std::stringstream ss;
    ss << regex_replace(content, pattern, "");
    return ss.str();
  }

  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }
};

using ManipulatorEnvPool = AsyncEnvPool<ManipulatorEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_MANIPULATOR_H_
