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

#ifndef ENVPOOL_VIZDOOM_VIZDOOM_ENV_H_
#define ENVPOOL_VIZDOOM_VIZDOOM_ENV_H_

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/utils/image_process.h"
#include "utils.h"

namespace vizdoom {

class VizdoomEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.bind(525), "img_height"_.bind(84),
        "img_width"_.bind(84), "stack_num"_.bind(4), "frame_skip"_.bind(4),
        "lmp_save_dir"_.bind(std::string("")), "episodic_life"_.bind(false),
        "force_speed"_.bind(false), "use_raw_action"_.bind(true),
        "use_inter_area_resize"_.bind(true),
        "reward_config"_.bind(std::map<std::string, std::tuple<float, float>>(
            {{'FRAGCOUNT', {1, -1.5}},         {'KILLCOUNT', {1, 0}},
             {'DEATHCOUNT', {-0.75, 0.75}},    {'HITCOUNT', {0.01, -0.01}},
             {'DAMAGECOUNT', {0.003, -0.003}}, {'HEALTH', {0.005, -0.003}},
             {'ARMOR', {0.005, -0.001}},       {'WEAPON0', {0.02, -0.01}},
             {'AMMO0', {0.0002, -0.0001}},     {'WEAPON1', {0.02, -0.01}},
             {'AMMO1', {0.0002, -0.0001}},     {'WEAPON2', {0.02, -0.01}},
             {'AMMO2', {0.0002, -0.0001}},     {'WEAPON3', {0.1, -0.05}},
             {'AMMO3', {0.001, -0.0005}},      {'WEAPON4', {0.1, -0.05}},
             {'AMMO4', {0.001, -0.0005}},      {'WEAPON5', {0.1, -0.05}},
             {'AMMO5', {0.001, -0.0005}},      {'WEAPON6', {0.2, -0.1}},
             {'AMMO6', {0.002, -0.001}},       {'WEAPON7', {0.2, -0.1}},
             {'AMMO7', {0.002, -0.001}}})),
        "selected_weapon_reward_config"_.bind(
            std::map<std::string, float>({{'min_duration', 5.0},
                                          {'SELECTED0', 0.0002},
                                          {'SELECTED1', 0.0002},
                                          {'SELECTED2', 0.0002},
                                          {'SELECTED3', 0.001},
                                          {'SELECTED4', 0.001},
                                          {'SELECTED5', 0.001},
                                          {'SELECTED6', 0.002},
                                          {'SELECTED7', 0.002}})),
        "delta_button_config"_.bind(
            std::map<std::string, std::tuple<int, float, float>>()),
        "cfg_path"_.bind(std::string("")), "wad_path"_.bind(std::string("")),
        "iwad_path"_.bind(std::string("freedoom2")),
        "game_args"_.bind(std::string("")),
        "map_id"_.bind(std::string("map01")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    DoomGame dg;
    dg.loadConfig(conf["cfg_path"_]);
    return MakeDict(
        "obs"_.bind(Spec<uint8_t>({conf["stack_num"_] * dg_.getScreenChannels(),
                                   conf["img_height"_], conf["img_width"_]},
                                  {0, 255})),
        "discount"_.bind(Spec<float>({-1}, {0.0f, 1.0f})),
        "info:AMMO2"_.bind(0.0f), "info:AMMO3"_.bind(0.0f),
        "info:AMMO4"_.bind(0.0f), "info:AMMO5"_.bind(0.0f),
        "info:AMMO6"_.bind(0.0f), "info:AMMO7"_.bind(0.0f),
        "info:ARMOR"_.bind(0.0f), "info:DAMAGECOUNT"_.bind(0.0f),
        "info:DEATHCOUNT"_.bind(0.0f), "info:FRAGCOUNT"_.bind(0.0f),
        "info:HEALTH"_.bind(0.0f), "info:HITCOUNT"_.bind(0.0f),
        "info:KILLCOUNT"_.bind(0.0f), "info:SELECTED_WEAPON"_.bind(0.0f),
        "info:SELECTED_WEAPON_AMMO"_.bind(0.0f), "info:USER2"_.bind(0.0f));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    DoomGame dg;
    dg.loadConfig(conf["cfg_path"_]);
    if (conf["use_raw_action"_]) {
      return MakeDict(
          "action"_.bind(Spec<double>({-1, dg.getAvailableButtons().size()})));
    } else {
      auto button_list = dg->getAvailableButtons();
      std::vector<std::tuple<int, float, float>> delta_config(
          _button_string_list.size());
      for (auto& i : conf["delta_button_config"_]) {
        int button_index = str2button(i.first);
        if (button_index != -1) {
          delta_config[button_index] = i.second;
        }
      }
      auto action_set =
          BuildActionSet(button_list, conf["force_speed"_], delta_config);
      return MakeDict(
          "action"_.bind(Spec<double>({-1}, {0.0f, action_set.size() - 1.0f})));
    }
  }
};

typedef class EnvSpec<VizdoomEnvFns> VizdoomEnvSpec;
typedef Spec<uint8_t> FrameSpec;

class VizdoomEnv : public Env<VizdoomEnvSpec> {
  const std::vector<int> kInfoIndex({0, 3, 4, 5, 7, 9, 10, 15, 16, 19, 20, 21,
                                     22, 23, 24, 73});
  // ({"AMMO2", "AMMO3", "AMMO4", "AMMO5", "AMMO6", "AMMO7", "ARMOR",
  //   "DAMAGECOUNT", "DEATHCOUNT", "FRAGCOUNT", "HEALTH", "HITCOUNT",
  //   "KILLCOUNT", "SELECTED_WEAPON", "SELECTED_WEAPON_AMMO", "USER2"});

 protected:
  std::unique_ptr<DoomGame> dg_;
  FrameSpec raw_spec_, resize_spec_;
  Array raw_buf_;
  std::deque<Array> stack_buf_;
  std::string lmp_dir_;
  bool save_lmp_, episodic_life_, use_raw_action_, use_inter_area_resize_;
  bool done_;
  int max_episode_steps_, elapsed_step_, stack_num_, frame_skip_,
      episode_count_, channel_;
  int deathcount_idx_, hitcount_idx_, damagecount_idx_;  // bugged var
  double last_deathcount_, last_hitcount_, last_damagecount_;
  int selected_weapon_, selected_weapon_count_, weapon_duration_;
  std::vector<vzd_act_t> action_set_;
  std::vector<Button> button_list_;
  std::vector<GameVariable> gv_list_;
  std::vector<double> gvs_, last_gvs_, pos_reward_, neg_reward_, weapon_reward_;

 public:
  VizdoomEnv(const Spec& spec, int env_id)
      : Env<VizdoomEnvSpec>(spec, env_id),
        dg_(new DoomGame()),
        lmp_dir_(spec.config["lmp_save_dir"_]),
        save_lmp_(lmp_dir_.length() > 0),
        episodic_life_(spec.config["episodic_life"_]),
        zero_discount_on_life_loss_(spec.config["zero_discount_on_life_loss"_]),
        use_raw_action_(spec.config["use_raw_action"_]),
        use_inter_area_resize_(spec.config["use_inter_area_resize"_]),
        done_(true),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        stack_num_(spec.config["stack_num"_]),
        frame_skip_(spec.config["frame_skip"_]),
        episode_count_(0),
        last_deathcount_(0),
        last_hitcount_(0),
        last_damagecount_(0),
        weapon_duration_(5),
        weapon_reward_(10) {
    if (save_lmp_) {
      lmp_dir_ = lmp_save_dir + "/env_" + std::to_string(env_id) + "_";
    }
    dg_->setViZDoomPath(spec.config["vzd_path"_]);
    dg_->setDoomGamePath(spec.config["iwad_path"_]);
    dg_->loadConfig(spec.config["cfg_path"_]);
    dg_->setWindowVisible(false);
    dg_->addGameArgs(spec.config["game_args"]);
    dg_->setMode(PLAYER);
    dg_->setEpisodeTimeout((max_episode_steps_ + 1) * frame_skip_);
    if (spec.config["wad_path"_].size()) {
      dg_->setDoomScenarioPath(spec.config["wad_path"_]);
    }
    dg_->setSeed(spec.config["seed"_]);
    dg_->setDoomMap(spec.config["map_id"_]);

    channel_ = dg_->getScreenChannels();
    raw_spec_ = FrameSpec({dg_->getScreenHeight(), dg_->getScreenWidth(), 1});
    raw_buf_ = Array(raw_spec_);
    resize_spec_ = FrameSpec(
        {channel_, spec.config["img_height"_], spec.config["img_width"_]});
    for (int i = 0; i < stack_num_; ++i) {
      stack_buf_.push_back(Array(resize_spec_));
    }

    gv_list_ = dg_->getAvailableGameVariables();
    // handle buggy DEATHCOUNT/HITCOUNT/DAMAGECOUNT in multi-player setting
    auto result = std::find(gv_list_.begin(), gv_list_.end(), DEATHCOUNT);
    if (result == gv_list_.end()) {
      deathcount_idx_ = -1;
    } else {
      deathcount_idx_ = result - gv_list_.begin();
    }
    result = std::find(gv_list_.begin(), gv_list_.end(), HITCOUNT);
    if (result == gv_list_.end()) {
      hitcount_idx_ = -1;
    } else {
      hitcount_idx_ = result - gv_list_.begin();
    }
    result = std::find(gv_list_.begin(), gv_list_.end(), DAMAGECOUNT);
    if (result == gv_list_.end()) {
      damagecount_idx_ = -1;
    } else {
      damagecount_idx_ = result - gv_list_.begin();
    }

    button_list_ = dg_->getAvailableButtons();
    std::vector<std::tuple<int, float, float>> delta_config(
        _button_string_list.size());
    for (auto& i : spec.config["delta_button_config"_]) {
      int button_index = str2button(i.first);
      if (button_index != -1) {
        delta_config[button_index] = i.second;
      }
    }
    action_set_ =
        BuildActionSet(button_list_, spec.config["force_speed"_], delta_config);

    // reward config
    pos_reward_.resize(gv_list_.size(), 0.0);
    neg_reward_.resize(gv_list_.size(), 0.0);
    for (auto& i : spec.config["reward_config"_]) {
      int gv_index = str2gv(i.first);
      if (gv_index == -1) {
        continue;
      }
      auto result = std::find(gv_list_.begin(), gv_list_.end(), gv_index);
      if (result == gv_list_.end()) {
        continue;
      }
      int index = result - gv_list_.begin();
      pos_reward_[index] = std::get<0>(i.second);
      neg_reward_[index] = std::get<1>(i.second);
    }
    // weapon reward config
    auto& weapon_config = spec.config["weapon_config"_];
    if (weapon_config.contains("min_duration")) {
      weapon_duration_ = weapon_config["min_duration"];
    }
    for (int i = 0; i < 8; ++i) {
      std::string key = "SELECTED" + std::to_string(i);
      if (weapon_config.contains(key)) {
        weapon_reward_[i] = weapon_config[key];
      }
    }
  }

  ~VizdoomEnv() { dg_->close(); }

  void Reset() override {
    if (dg_->isEpisodeFinished() || elapsed_step_ >= max_episode_steps_) {
      elapsed_step_ = 0;
      if (episode_count_ > 0) {  // NewEpisode at beginning may hang on MAEnv
        if (save_lmp_) {
          dg_->newEpisode(lmp_dir_ + std::to_string(episode_count_) + ".lmp");
        } else {
          dg_->newEpisode();
        }
      }
    } else {
      ++elapsed_step_;
      dg_->makeAction(action_set_[0], frame_skip_);
    }
    done_ = false;
    ++episode_count_;
    GetState(true);
  }

  void Step(const Action& action) override {
    double* ptr = static_cast<double*>(action["action"_].data());
    if (use_raw_action_) {
      dg_->setAction(std::vector<double>(ptr, ptr + button_list_.size()));
    } else {
      dg_->setAction(action_set_[static_cast<int>(ptr[0])]);
    }
    dg_->advanceAction(frame_skip_, true);
    ++elapsed_step_;
    done_ = (dg_->isEpisodeFinished() | (elapsed_step_ >= max_episode_steps_));
    if (episodic_life_ && dg_->isPlayerDead()) {
      done_ = true;
    }
    GetState(false);
  }

  void GetState(bool is_reset) {
    GameStatePtr state = dg_->getState();
    if (state == nullptr) {  // finish episode
      return;
    }

    // game variables and reward
    if (is_reset) {
      last_gvs_ = state->gameVariables;
      selected_weapon_ = -1;
      selected_weapon_count_ = 0;
    } else {
      last_gvs_ = gvs_;
    }
    gvs_ = state->gameVariables;

    // some variables don't get reset to zero on game.newEpisode().
    // see https://github.com/mwydmuch/ViZDoom/issues/399
    if (hitcount_idx_ >= 0) {
      if (is_reset) {
        last_hitcount_ = gvs_[hitcount_idx];
        last_gvs_[hitcount_idx] = 0;
      }
      gvs_[hitcount_idx] -= last_hitcount_;
    }
    if (damagecount_idx_ >= 0) {
      if (is_reset) {
        last_damagecount_ = gvs_[damagecount_idx];
        last_gvs_[damagecount_idx] = 0;
      }
      gvs_[damagecount_idx] -= last_damagecount_;
    }
    if (deathcount_idx_ >= 0) {
      if (is_reset) {
        last_deathcount_ = gvs_[deathcount_idx];
        last_gvs_[deathcount_idx] = 0;
      }
      gvs_[deathcount_idx] -= last_deathcount_;
    }

    int curr_weapon = -1, curr_weapon_ammo = 0;
    float reward = 0.0f;

    for (int i = 0; i < gvs_.size(); ++i) {
      double delta = gvs_[i] - last_gvs_[i];
      // without this we reward using BFG and shotguns too much
      if (gv_list_[i] == DAMAGECOUNT && delta >= 200) {
        delta = 200;
      } else if (gv_list_[i] == HITCOUNT && delta >= 5) {
        delta = 5;
      } else if (gv_list_[i] == SELECTED_WEAPON) {
        curr_weapon = gvs_[i];
      } else if (gv_list_[i] == SELECTED_WEAPON_AMMO) {
        curr_weapon_ammo = gvs_[i];
      } else if (gv_list_[i] == HEALTH) {  // NOLINT
        // HEALTH -999900: https://github.com/mwydmuch/ViZDoom/issues/202
        if (last_gvs_[i] < 0 && gvs_[i] < 0) {
          last_gvs_[i] = gvs_[i] = 100;
        } else if (gvs_[i] < 0) {
          gvs_[i] = last_gvs_[i];
        } else if (last_gvs_[i] < 0) {
          last_gvs_[i] = gvs_[i];
        }
        delta = gvs_[i] - last_gvs_[i];
      }
      if (delta >= 0) {
        reward += delta * pos_reward_[i];
      } else {
        reward -= delta * neg_reward_[i];
      }
    }

    // update weapon counter
    if (curr_weapon == selected_weapon_) {
      ++selected_weapon_count_;
    } else {
      selected_weapon_ = curr_weapon;
      selected_weapon_count_ = 1;
    }
    if (curr_weapon >= 0 && selected_weapon_count_ >= weapon_duration_ &&
        curr_weapon_ammo > 0) {
      reward += weapon_reward_[selected_weapon_];
    }

    Array tgt = std::move(*stack_buf_.begin());
    uint8_t* ptr = static_cast<uint8_t*>(tgt.data());
    stack_buf_.pop_front();

    // get screen
    uint8_t* raw_ptr = static_cast<uint8_t*>(raw_buf_.data());
    std::size_t size = raw_buf_.size;
    for (int c = 0; c < channel_; ++c) {
      // state->screenBuffer is channel-first image
      memcpy(raw_ptr, state->screenBuffer + c * size, size);
      auto slice = tgt[c];
      Resize(raw_buf_, &slice, use_inter_area_resize_);
    }
    size = tgt.size;
    stack_buf_.push_back(std::move(tgt));
    if (is_reset) {
      for (auto& s : stack_buf_) {
        uint8_t* ptr_s = static_cast<uint8_t*>(s.data());
        if (ptr != ptr_s) {
          memcpy(ptr_s, ptr, size);
        }
      }
    }

    State state = Allocate();
    state["reward"_] = reward;
    for (int i = 0; i < stack_num_; ++i) {
      state["obs"_]
          .Slice(i * channel_, (i + 1) * channel_)
          .Assign(stack_buf_[i]);
    }
    // info
  }
};

}  // namespace vizdoom

#endif  // ENVPOOL_VIZDOOM_VIZDOOM_ENV_H_
