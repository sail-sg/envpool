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

#ifndef ENVPOOL_VIZDOOM_ENGINE_H_
#define ENVPOOL_VIZDOOM_ENGINE_H_

#include "utils.h"
#include "envpool/core/async_envpool.h"
#include "envpool/utils/image_process.h"

namespace vizdoom {

class VizdoomCore {
 protected:
  int raw_height_, raw_width_, raw_screen_size_, raw_channel_, channel_, has_depth_;
  std::deque<Array> stack_buf_;
  std::string lmp_dir_;
  bool save_lmp_, episodic_life_, use_raw_action_, is_cig_;
  int stack_num_, frame_skip_, episode_count_;
  int elapsed_step_, max_episode_steps_;
  int height_, width_, screen_size_, obs_size_;
  int deathcount_idx_, hitcount_idx_, damagecount_idx_;  // bugged var
  double last_deathcount_, last_hitcount_, last_damagecount_;
  int selected_weapon_, selected_weapon_count_, weapon_duration_;
  std::vector<vzd_act_t> action_set_;
  std::vector<Button> button_list_;
  std::vector<GameVariable> gv_list_;
  std::vector<std::string> button_name_, gv_name_;
  std::vector<double> gvs_, last_gvs_, pos_reward_, neg_reward_, weapon_reward_;
  vzd_act_t current_act_;

 public:
  std::unique_ptr<DoomGame> dg_;
  bool done_;

  VizdoomCore(int seed, int env_id, int max_episode_steps, int stack_num,
              int frame_skip, bool episodic_life,
              bool use_raw_action, bool force_speed, int height, int width,
              std::map<std::string, std::tuple<float, float>> reward_config,
              std::map<std::string, float> weapon_config,
              std::map<std::string, std::tuple<int, float, float>> delta_button_config,
              std::string cfg_path, std::string wad_path, std::string vzd_path,
              std::string iwad_path, std::string map_id, std::string game_args,
              std::string lmp_save_dir)
      : height_(height),
        width_(width),
        max_episode_steps_(max_episode_steps),
        elapsed_step_(max_episode_steps + 1),
        stack_num_(stack_num),
        frame_skip_(frame_skip),
        episodic_life_(episodic_life),
        use_raw_action_(use_raw_action),
        save_lmp_(lmp_save_dir.length() > 0),
        episode_count_(0),
        lmp_dir_(lmp_save_dir),
        last_damagecount_(0),
        last_hitcount_(0),
        last_deathcount_(0),
        weapon_duration_(5),
        weapon_reward_(10),
        dg_(new DoomGame()),
        done_(true) {
    if (save_lmp_) {
      lmp_dir_ = lmp_save_dir + "/episode_" + std::to_string(env_id) + "_";
    }
    dg_->setViZDoomPath(vzd_path);
    dg_->setDoomGamePath(iwad_path);
    dg_->loadConfig(cfg_path);
    dg_->setWindowVisible(false);
    dg_->addGameArgs(game_args);
    dg_->setMode(PLAYER);
    dg_->setEpisodeTimeout((max_episode_steps + 1) * frame_skip);
    if (wad_path.size()) {
      dg_->setDoomScenarioPath(wad_path);
    }
    dg_->setSeed(seed);
    dg_->setDoomMap(map_id);

    has_depth_ = dg_->isDepthBufferEnabled();
    raw_height_ = dg_->getScreenHeight();
    raw_width_ = dg_->getScreenWidth();
    raw_channel_ = dg_->getScreenChannels();
    channel_ = raw_channel_ + has_depth_;
    raw_screen_size_ = raw_width_ * raw_height_ * channel_;
    screen_size_ = width_ * height_ * channel_;
    obs_size_ = screen_size_ * stack_num_;

    for (int i = 0; i < stack_num_; ++i) {
      stack_buf_.push_back(Array(Spec<uint8_t>({channel_, width_, height_})));
    }

    gv_list_ = dg_->getAvailableGameVariables();
    for (GameVariable gv : gv_list_) {
      gv_name_.push_back(gv2str(gv));
    }
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
    for (Button b : button_list_) {
      button_name_.push_back(button2str(b));
    }
    std::vector<std::tuple<int, float, float>> delta_config(
        _button_string_list.size());
    for (auto i : delta_button_config) {
      std::string name = i.first.cast<py::str>();
      int button_index = str2button(name);
      if (button_index == -1) continue;
      auto cfg = i.second.cast<py::tuple>();
      delta_config[button_index] = std::make_tuple(
          py::int_(cfg[0]), py::float_(cfg[1]), py::float_(cfg[2]));
    }
    action_set = build_action_set(button_list, force_speed, delta_config);

    // reward config
    pos_reward.resize(gv_list.size(), 0.0);
    neg_reward.resize(gv_list.size(), 0.0);
    for (auto i : reward_config) {
      std::string name = i.first.cast<py::str>();
      int gv_index = str2gv(name);
      if (gv_index == -1) continue;
      auto result = std::find(gv_list.begin(), gv_list.end(), gv_index);
      if (result == gv_list.end()) continue;
      int index = result - gv_list.begin();
      auto cfg = i.second.cast<py::tuple>();
      pos_reward[index] = py::float_(cfg[0]);
      neg_reward[index] = py::float_(cfg[1]);
    }
    // weapon reward config
    if (weapon_config.contains("min_duration")) {
      weapon_duration = weapon_config["min_duration"].cast<double>();
    }
    for (int i = 0; i < 8; ++i) {
      std::string key = "SELECTED" + std::to_string(i);
      if (weapon_config.contains(key))
        weapon_reward[i] = weapon_config[key.c_str()].cast<double>();
    }
  }

  ~VizdoomCore() { close(); }

  std::vector<std::string> get_game_variable_name() { return gv_name; }

  std::vector<std::string> get_button_name() { return button_name; }

  std::vector<vzd_act_t> get_action_set() { return action_set; }

  void new_episode() {
    elapsed_step = 0;
    if (episode_count == 0) { // newEpisode at beginning may hang on MAEnv
      return;
    }
    if (save_lmp) {
      dg_->newEpisode(lmp_dir + std::to_string(episode_count) + ".lmp");
    } else {
      dg_->newEpisode();
    }
  }

  void _reset_meta() {
    this->reward = 0.0;
    this->done = false;
    ++episode_count;
    _reset_buffer();
    _get_state(true);
  }

  int reset() {
    if (dg_->isEpisodeFinished() || elapsed_step >= max_episode_steps) {
      new_episode();
    } else {
      ++elapsed_step;
      dg_->makeAction(action_set[0], frame_skip_);
    }
    _reset_meta();
    return 1;
  }

  // for multiplayer usecase
  void set_action(vzd_act_t act) { current_act = act; }

  void step_before() { dg_->setAction(current_act); }

  void step_after() {
    this->reward = 0.0;
    ++elapsed_step;
    this->done =
        (dg_->isEpisodeFinished() | (elapsed_step >= max_episode_steps));
    if (episodic_life && dg_->isPlayerDead()) this->done = true;
    _get_state(false);
  }

  int step() {
    step_before();
    dg_->advanceAction(frame_skip_, true);
    step_after();
    return 1;
  }

  void _reset_buffer() {
    for (frame_t &buf_ptr : stack_buf) {
      memset(buf_ptr.get(), 0, sizeof(obs_t) * screen_size);
    }
  }

  void _get_state(bool is_reset) {
    GameStatePtr state = dg_->getState();
    if (state == nullptr) {  // finish episode
      return;
    }

    // stack[:-1] = stack[1:]
    stack_buf.push_back(std::move(stack_buf.front()));
    stack_buf.pop_front();

    // game variables and reward
    if (is_reset) {
      last_gvs = state->gameVariables;
      selected_weapon = -1;
      selected_weapon_count = 0;
    } else {
      last_gvs = gvs;
    }
    gvs = state->gameVariables;

    // some variables don't get reset to zero on game.newEpisode().
    // see https://github.com/mwydmuch/ViZDoom/issues/399
    if (hitcount_idx >= 0) {
      if (is_reset) {
        last_hitcount = gvs[hitcount_idx];
        last_gvs[hitcount_idx] = 0;
      }
      gvs[hitcount_idx] -= last_hitcount;
    }
    if (damagecount_idx >= 0) {
      if (is_reset) {
        last_damagecount = gvs[damagecount_idx];
        last_gvs[damagecount_idx] = 0;
      }
      gvs[damagecount_idx] -= last_damagecount;
    }
    if (deathcount_idx >= 0) {
      if (is_reset) {
        last_deathcount = gvs[deathcount_idx];
        last_gvs[deathcount_idx] = 0;
      }
      gvs[deathcount_idx] -= last_deathcount;
    }

    int curr_weapon = -1, curr_weapon_ammo = 0;

    for (int i = 0; i < gvs.size(); ++i) {
      double delta = gvs[i] - last_gvs[i];
      // without this we reward using BFG and shotguns too much
      if (gv_list[i] == DAMAGECOUNT && delta >= 200) {
        delta = 200;
      } else if (gv_list[i] == HITCOUNT && delta >= 5) {
        delta = 5;
      } else if (gv_list[i] == SELECTED_WEAPON) {
        curr_weapon = gvs[i];
      } else if (gv_list[i] == SELECTED_WEAPON_AMMO) {
        curr_weapon_ammo = gvs[i];
      } else if (gv_list[i] == HEALTH) {  // NOLINT
        // HEALTH -999900: https://github.com/mwydmuch/ViZDoom/issues/202
        if (last_gvs[i] < 0 && gvs[i] < 0) {
          last_gvs[i] = gvs[i] = 100;
        } else if (gvs[i] < 0) {
          gvs[i] = last_gvs[i];
        } else if (last_gvs[i] < 0) {
          last_gvs[i] = gvs[i];
        }
        delta = gvs[i] - last_gvs[i];
      }
      if (delta >= 0) {
        this->reward += delta * pos_reward[i];
      } else {
        this->reward -= delta * neg_reward[i];
      }
    }

    // update weapon counter
    if (curr_weapon == selected_weapon) {
      ++selected_weapon_count;
    } else {
      selected_weapon = curr_weapon;
      selected_weapon_count = 1;
    }
    if (curr_weapon >= 0 && selected_weapon_count >= weapon_duration &&
        curr_weapon_ammo > 0)
      this->reward += weapon_reward[selected_weapon];

    // get screen
    obs_t *frame_ptr = stack_buf.back().get();
    if (!depth_only) {
      for (int c = 0; c < raw_channel; ++c) {
        // state->screenBuffer is channel-first image
        resize(state->screenBuffer + c * raw_height * raw_width,
               frame_ptr + c * height * width, raw_height, raw_width, height,
               width, 1);
      }
      if (has_depth) {
        resize(state->depthBuffer, frame_ptr + raw_channel * height * width,
               raw_height, raw_width, height, width, 1);
      }
    } else {
      resize(state->depthBuffer, frame_ptr, raw_height, raw_width, height,
             width, 1);
    }
  }

  void get_obs(obs_t *obs) {
    for (int i = 0; i < stack_num; ++i)
      memcpy(obs + i * screen_size, stack_buf[i].get(),
             sizeof(obs_t) * screen_size);
  }

  void get_info(info_t *info) {
    for (int i = 0; i < gvs.size(); ++i) info[i] = gvs[i];
  }

  int get_action_size() {
    return use_raw_action ? button_list.size() : action_set.size();
  }

  std::vector<int> get_obs_shape() {
    return {stack_num * channel, height, width};
  }

  py::dict cast_info_dict(py::array info) {
    py::dict result = py::dict();
    for (int i = 0; i < gv_name.size(); ++i)
      result[gv_name[i].c_str()] = info[py::make_tuple(py::ellipsis(), i)];
    return result;
  }
};

}  // namespace vizdoom

#endif  // ENVPOOL_VIZDOOM_ENGINE_H_
