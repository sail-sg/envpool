#include <string>
#include "config.h"
#include "game_env.hpp"

class CoreAction{
  public:
    CoreAction(Action action, std::string name, bool sticky = false, bool directional = false){
      this->action_ = action;
      this->name_ = name;
      this->sticky_ = sticky;
      this->directional_ = directional;
    }
    bool is_in_actionset(Config config){
    };
    Action action_;
    std::string name_;
    bool sticky_;
    bool directional_;
};

bool T = true;
bool F = false;

CoreAction action_idle(Action::game_idle, "idle");
CoreAction action_builtin_ai(Action::game_builtin_ai, "builtin_ai");
CoreAction action_left(Action::game_left, "left", T, T);
CoreAction action_top_left(Action::game_top_left, "top_left", T, T);
CoreAction action_top(Action::game_top, "top", T, T);
CoreAction action_top_right(Action::game_top_right, "top_right", T, T);
CoreAction action_right(Action::game_right, "right", T, T);
CoreAction action_bottom_right(Action::game_bottom_right, "bottom_right", T, T);
CoreAction action_bottom(Action::game_bottom, "bottom", T, T);
CoreAction action_bottom_left(Action::game_bottom_left, "bottom_left", T, T);
CoreAction action_long_pass(Action::game_long_pass, "long_pass");
CoreAction action_high_pass(Action::game_high_pass, "high_pass");
CoreAction action_short_pass(Action::game_short_pass, "short_pass");
CoreAction action_shot(Action::game_shot, "shot");
CoreAction action_keeper_rush(Action::game_keeper_rush, "keeper_rush", T);
CoreAction action_sliding(Action::game_sliding, "sliding");
CoreAction action_pressure(Action::game_pressure, "pressure", T);
CoreAction action_team_pressure(Action::game_team_pressure, "team_pressure", T);
CoreAction action_switch(Action::game_switch, "switch");
CoreAction action_sprint(Action::game_sprint, "sprint", T);
CoreAction action_dribble(Action::game_dribble, "dribble", T);
CoreAction action_release_direction(Action::game_release_direction, "release_direction", F, T);
CoreAction action_release_long_pass(Action::game_release_long_pass, "release_long_pass");
CoreAction action_release_high_pass(Action::game_release_high_pass, "release_high_pass");
CoreAction action_release_short_pass(Action::game_release_short_pass, "release_short_pass");
CoreAction action_release_shot(Action::game_release_shot, "release_shot");
CoreAction action_release_keeper_rush(Action::game_release_keeper_rush, "release_keeper_rush");
CoreAction action_release_sliding(Action::game_release_sliding, "release_sliding");
CoreAction action_release_pressure(Action::game_release_pressure, "release_pressure");
CoreAction action_release_team_pressure(Action::game_release_team_pressure, "release_team_pressure");
CoreAction action_release_switch(Action::game_release_switch, "release_switch");
CoreAction action_release_sprint(Action::game_release_sprint, "release_sprint");
CoreAction action_release_dribble(Action::game_release_dribble, "release_dribble");