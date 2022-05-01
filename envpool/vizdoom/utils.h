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

#ifndef ENVPOOL_VIZDOOM_UTILS_H_
#define ENVPOOL_VIZDOOM_UTILS_H_

#include <string>
#include <tuple>
#include <vector>

#include "ViZDoom.h"

namespace vizdoom {

using VzdActT = std::vector<double>;

void BuildActionSetImpl(
    int button_num,
    const std::vector<std::tuple<int, float, float>>& delta_config,
    int* button_index, bool force_speed, int cur_id, double* current_result,
    std::vector<VzdActT>& result  // NOLINT
) {
  if (cur_id == 43) {
    // get full action set, move to result
    VzdActT current_result_vec;
    for (int i = 0; i < button_num; ++i) {
      current_result_vec.push_back(current_result[i]);
    }
    result.push_back(current_result_vec);
  } else if (38 <= cur_id && cur_id <= 42) {
    // delta button, parse delta_config
    int num;
    float action_min;
    float action_max;
    std::tie(num, action_min, action_max) = delta_config[cur_id];
    if (num <= 1) {
      BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                         cur_id + 1, current_result, result);
    } else {
      float delta = (action_max - action_min) / static_cast<float>(num - 1);
      float a = action_min;
      for (int i = 0; i < num; ++i, a += delta) {
        current_result[button_index[cur_id]] = a;
        BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                           cur_id + 1, current_result, result);
      }
    }
  } else if (21 <= cur_id && cur_id <= 30) {
    // SELECT_WEAPONX
    // select none of weapons
    BuildActionSetImpl(button_num, delta_config, button_index, force_speed, 31,
                       current_result, result);
    // select one weapon
    for (int i = 21; i <= 30; ++i) {
      if (button_index[i] != -1) {
        current_result[button_index[i]] = 1;
        BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                           31, current_result, result);
        current_result[button_index[i]] = 0;
      }
    }
  } else if ((cur_id == 10 || cur_id == 12 || cur_id == 14 || cur_id == 16 ||
              cur_id == 18 || cur_id == 31 || cur_id == 35) &&
             (button_index[cur_id] != -1 && button_index[cur_id + 1] != -1)) {
    // pairwise button, at most one of them can be selected
    // 10, 11: MOVE_RIGHT / MOVE_LEFT
    // 12, 13: MOVE_BACKWARD / MOVE_FORWARD
    // 14, 15: TURN_RIGHT / TURN_LEFT
    // 16, 17: LOOK_UP / LOOK_DOWN
    // 18, 19: MOVE_UP / MOVE_DOWN
    // 31, 32: SELECT_NEXT_WEAPON / SELECT_PREV_WEAPON
    // 34, 35: SELECT_NEXT_ITEM / SELECT_PREV_ITEM
    // False, False
    BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                       cur_id + 2, current_result, result);
    current_result[button_index[cur_id]] = 1;
    // True, False
    BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                       cur_id + 2, current_result, result);
    current_result[button_index[cur_id]] = 0;
    current_result[button_index[cur_id + 1]] = 1;
    // False, True
    BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                       cur_id + 2, current_result, result);
    current_result[button_index[cur_id + 1]] = 0;
  } else if (button_index[cur_id] != -1) {
    // single button
    if (cur_id == 8 && force_speed) {  // SPEED
      current_result[button_index[cur_id]] = 1;
      BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                         cur_id + 1, current_result, result);
      return;
    }
    BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                       cur_id + 1, current_result, result);
    current_result[button_index[cur_id]] = 1;
    BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                       cur_id + 1, current_result, result);
    current_result[button_index[cur_id]] = 0;
  } else {
    BuildActionSetImpl(button_num, delta_config, button_index, force_speed,
                       cur_id + 1, current_result, result);
  }
}

std::vector<VzdActT> BuildActionSet(
    std::vector<Button> button_list, bool force_speed,
    const std::vector<std::tuple<int, float, float>>& delta_config) {
  std::vector<VzdActT> result;
  std::array<int, 43> button_index;
  std::array<double, 43> current_result;
  memset(button_index.begin(), -1, sizeof(int) * 43);
  memset(current_result.begin(), 0, sizeof(double) * 43);
  for (std::size_t i = 0; i < button_list.size(); ++i) {
    button_index[button_list[i]] = i;
  }
  BuildActionSetImpl(button_list.size(), delta_config, button_index.begin(),
                     force_speed, 0, current_result.begin(), result);
  return result;
}

std::vector<std::string> button_string_list({
    "ATTACK",
    "USE",
    "JUMP",
    "CROUCH",
    "TURN180",
    "ALTATTACK",
    "RELOAD",
    "ZOOM",
    "SPEED",
    "STRAFE",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "MOVE_BACKWARD",
    "MOVE_FORWARD",
    "TURN_RIGHT",
    "TURN_LEFT",
    "LOOK_UP",
    "LOOK_DOWN",
    "MOVE_UP",
    "MOVE_DOWN",
    "LAND",
    "SELECT_WEAPON1",
    "SELECT_WEAPON2",
    "SELECT_WEAPON3",
    "SELECT_WEAPON4",
    "SELECT_WEAPON5",
    "SELECT_WEAPON6",
    "SELECT_WEAPON7",
    "SELECT_WEAPON8",
    "SELECT_WEAPON9",
    "SELECT_WEAPON0",
    "SELECT_NEXT_WEAPON",
    "SELECT_PREV_WEAPON",
    "DROP_SELECTED_WEAPON",
    "ACTIVATE_SELECTED_ITEM",
    "SELECT_NEXT_ITEM",
    "SELECT_PREV_ITEM",
    "DROP_SELECTED_ITEM",
    "LOOK_UP_DOWN_DELTA",
    "TURN_LEFT_RIGHT_DELTA",
    "MOVE_FORWARD_BACKWARD_DELTA",
    "MOVE_LEFT_RIGHT_DELTA",
    "MOVE_UP_DOWN_DELTA",
});

std::string Button2Str(Button b) {
  assert(b >= 0 && b < button_string_list.size());
  return button_string_list[b];
}

int Str2Button(const std::string& s) {
  auto result =
      std::find(button_string_list.begin(), button_string_list.end(), s);
  if (result != button_string_list.end()) {
    return result - button_string_list.begin();
  }
  return -1;
}

std::vector<std::string> gv_string_list({
    "KILLCOUNT",
    "ITEMCOUNT",
    "SECRETCOUNT",
    "FRAGCOUNT",
    "DEATHCOUNT",
    "HITCOUNT",
    "HITS_TAKEN",
    "DAMAGECOUNT",
    "DAMAGE_TAKEN",
    "HEALTH",
    "ARMOR",
    "DEAD",
    "ON_GROUND",
    "ATTACK_READY",
    "ALTATTACK_READY",
    "SELECTED_WEAPON",
    "SELECTED_WEAPON_AMMO",
    "AMMO0",
    "AMMO1",
    "AMMO2",
    "AMMO3",
    "AMMO4",
    "AMMO5",
    "AMMO6",
    "AMMO7",
    "AMMO8",
    "AMMO9",
    "WEAPON0",
    "WEAPON1",
    "WEAPON2",
    "WEAPON3",
    "WEAPON4",
    "WEAPON5",
    "WEAPON6",
    "WEAPON7",
    "WEAPON8",
    "WEAPON9",
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "ANGLE",
    "PITCH",
    "ROLL",
    "VIEW_HEIGHT",
    "VELOCITY_X",
    "VELOCITY_Y",
    "VELOCITY_Z",
    "CAMERA_POSITION_X",
    "CAMERA_POSITION_Y",
    "CAMERA_POSITION_Z",
    "CAMERA_ANGLE",
    "CAMERA_PITCH",
    "CAMERA_ROLL",
    "CAMERA_FOV",
    "PLAYER_NUMBER",
    "PLAYER_COUNT",
    "PLAYER1_FRAGCOUNT",
    "PLAYER2_FRAGCOUNT",
    "PLAYER3_FRAGCOUNT",
    "PLAYER4_FRAGCOUNT",
    "PLAYER5_FRAGCOUNT",
    "PLAYER6_FRAGCOUNT",
    "PLAYER7_FRAGCOUNT",
    "PLAYER8_FRAGCOUNT",
    "PLAYER9_FRAGCOUNT",
    "PLAYER10_FRAGCOUNT",
    "PLAYER11_FRAGCOUNT",
    "PLAYER12_FRAGCOUNT",
    "PLAYER13_FRAGCOUNT",
    "PLAYER14_FRAGCOUNT",
    "PLAYER15_FRAGCOUNT",
    "PLAYER16_FRAGCOUNT",
    "USER1",
    "USER2",
    "USER3",
    "USER4",
    "USER5",
    "USER6",
    "USER7",
    "USER8",
    "USER9",
    "USER10",
    "USER11",
    "USER12",
    "USER13",
    "USER14",
    "USER15",
    "USER16",
    "USER17",
    "USER18",
    "USER19",
    "USER20",
    "USER21",
    "USER22",
    "USER23",
    "USER24",
    "USER25",
    "USER26",
    "USER27",
    "USER28",
    "USER29",
    "USER30",
    "USER31",
    "USER32",
    "USER33",
    "USER34",
    "USER35",
    "USER36",
    "USER37",
    "USER38",
    "USER39",
    "USER40",
    "USER41",
    "USER42",
    "USER43",
    "USER44",
    "USER45",
    "USER46",
    "USER47",
    "USER48",
    "USER49",
    "USER50",
    "USER51",
    "USER52",
    "USER53",
    "USER54",
    "USER55",
    "USER56",
    "USER57",
    "USER58",
    "USER59",
    "USER60",
});

std::string GV2Str(GameVariable gv) {
  assert(gv >= 0 && gv < gv_string_list.size());
  return gv_string_list[gv];
}

int Str2GV(const std::string& s) {
  auto result = std::find(gv_string_list.begin(), gv_string_list.end(), s);
  if (result != gv_string_list.end()) {
    return result - gv_string_list.begin();
  }
  return -1;
}

}  // namespace vizdoom

#endif  // ENVPOOL_VIZDOOM_UTILS_H_
