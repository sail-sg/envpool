#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <time.h>
#include <ostream>
#include <direct.h>
#include <deque>
// env-specific definition of config and state/action spec

class FootballEnvFns {
 public:
  char* cwd[256];
  getcwd(cwd, 256);
  std::string cwd_string = cwd;
  cwd_string.append("/dump")
  static decltype(auto) DefaultConfig() {
    return MakeDict("action_set"_.Bind(std::string("default")),
                    "custom_display_stats"_.Bind(0),
                    "display_game_stats"_.Bind(true),
                    "dump_full_episodes"_.Bind(false),
                    "dump_scores"_.Bind(false),
                    "players"_.Bind(std::vector<std::string>{"agent:left_players=1"}),
                    "level"_.Bind(std::string("11_vs_11_stochastic")),
                    "physics_steps_per_frame"_.Bind(10),
                    "render_resolution_x"_.Bind(1280),
                    "real_time"_.Bind(false),
                    "tracesdir"_.Bind(cwd_string),
                    "video_format"_.Bind(std::string("avi")),
                    "video_quality_level"_.Bind(0),
                    "write_video"_.Bind(false),
                    "render_resolution_y"_.Bind(0.5625 * 1280)
                    );
  }
  
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(
        Spec<uint8_t>({72, 96, 16}, {0, 255})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // the last argument in Spec is for the range definition
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 18})));
  }
};
// this line will concat common config and common state/action spec
using FootballEnvSpec = EnvSpec<FootballEnvFns>;

class CoreAction{
  public:
    std::string backend_action_;
    std::string name_;
    bool sticky_ = false;
    bool directional_ = false;

    CoreAction(std::string backend_action, std::string name, bool sticky, bool directional):
    backend_action_(backend_action),
    name_(name),
    sticky_(sticky),
    directional_(directional){}

    bool is_in_actionset(const Spec& spec) {
      std::string action_set_name = spec.config["action_set"];
      if (action_set_name == "default") {
        return !(std::find(action_set_dict_default.begin(), action_set_dict_default.end(), *this) == action_set_dict_default.end());
      }
      else if (action_set_name == "v2") {
        return !(std::find(action_set_dict_v2.begin(), action_set_dict_v2.end(), *this) == action_set_dict_v2.end());
      }
      else {
        return !(std::find(action_set_dict_full.begin(), action_set_dict_full.end(), *this) == action_set_dict_full.end());
      }
   }
}

class DumpConfig{
  public:
    int max_count_ = 1;
    int steps_before_ = 100;
    int steps_after_ = 0;
    int min_frequency_ = 10;
    ObservationProcessor active_dump = NULL;
    clock_t last_dump_time = clock() - 2 * min_frequency_;
    DumpConfig(int max_count, int steps_before, int steps_after, int min_frequency) :
    max_count_(max_count), steps_before_(steps_before), steps_after_(steps_after){}

}
class ObservationState{
  public:
    std::map<std::string, >trace_;
    std::vector<int>additional_frames;
    std::vector<std::string>debugs;
    ObservationState(std::map<std::string, >trace):trace_(trace){}
    
}
class ObservationProcessor{
  public:
    float ball_takeover_epsilon = 0.03;
    float ball_lost_epsilon = 0.05;
    int frame = 0;
    Spec spec_self;
    std::deque<ObservationProcessor>trace;
    nullptr_t state = NULL;
    std::map<std::string, DumpConfig> dump_config;
    Spec& spec_self;

    ObservationProcessor(const Spec& spec){
      spec_self = spec;
      int max_count, steps_before = 100, steps_after = 1, min_frequency = 600;
      if(spec.config["dump_socre"]){
        max_count = 100000;
      }
      else{
        max_count = 0;
      }
      dump_config.insert(pair<std::string, DumpConfig>("score", DumpConfig(max_count, steps_before, steps_after, min_frequency)));
      dump_config.insert(pair<std::string, DumpConfig>("lost_score", DumpConfig(max_count, steps_before, steps_after, min_frequency)));
      steps_after = 10000;
      steps_before = 0;
      min_frequency = 10;
      if(spec.config["dump_full_episodes"]){
        max_count = 100000;
      }
      else{
        max_count = 0;
      }
      dump_config.insert(pair<std::string, DumpConfig>("episode_done", DumpConfig(max_count, steps_before, steps_after, min_frequency)));
      max_count = 1;
      steps_before = 100;
      steps_after = 0;
      min_frequency = 10;
      dump_config.insert(pair<std::string, DumpConfig>("shutdown", DumpConfig(max_count, steps_before, steps_after, min_frequency)));
      this->clear_state();
    }
    void clear_state();
    void reset();
    int len();
    void add_frame(int frame);
    void upstate(std::deque<ObservationProcessor>trace_);

    std::vector<DumpConfig> pending_dumps();
}

void ObservationProcessor::clear_state(){
  this->frame = 0;
  this->state = NULL;
  this->trace.clear();
}
void ObservationProcessor::reset(){
  this->clear_state();
}
int ObservationProcessor::len(){
  return(this->trace.size());
}
void ObservationProcessor::add_frame(int frame){
  if (this->trace.size() > 0 && this->spec_self.config["write_video"]){
    this->trace.back().add_frame(frame);
    for(int i = 0; i < this->pending_dumps().size; i++){
      this->pending_dumps()[i].add_frame(frame);
    }
  }
}
void ObservationProcessor::upstate(std::map<std::string, >trace){
  this->frame += 1;
  int frame = trace["frame"];
  if(this->spec_self.config["write_video"] == false && trace["observation"] == "frame"){
    std::map<std::string, >no_video_trace = trace;
    no_video_trace["observation"] = trace["observation"];
    no_video_trace.erase("observation");
    no_video_trace.erase("frame");

  }
}
std::vector<ObservationProcessor> ObservationProcessor::pending_dumps(){
  std::vector<DumpConfig>dumps;
  std::map<std::string, DumpConfig>::iterator iter;
  for(iter = this->dump_config.begin(); iter != _map.end(); iter++) {
      if(iter->second.active_dump != NULL){
        dumps.push_back(iter->second.active_dump);
      }
    }
  return(dumps);
}

action_idle = CoreAction(e_BackendAction.idle, "idle", false, false);
action_builtin_ai = CoreAction(e_BackendAction.builtin_ai, "builtin_ai", false, false);
action_left = CoreAction(
    e_BackendAction.left, "left", true, true);
action_top_left = CoreAction(
    e_BackendAction.top_left, "top_left", true, true);
action_top = CoreAction(
    e_BackendAction.top, "top", true, true);
action_top_right = CoreAction(
    e_BackendAction.top_right, "top_right", true, true);
action_right = CoreAction(
    e_BackendAction.right, "right", true, true);
action_bottom_right = CoreAction(
    e_BackendAction.bottom_right, "bottom_right", true, true);
action_bottom = CoreAction(
    e_BackendAction.bottom, "bottom", true, true);
action_bottom_left = CoreAction(
    e_BackendAction.bottom_left, "bottom_left", true, true);
action_long_pass = CoreAction(e_BackendAction.long_pass, "long_pass", false, false);
action_high_pass = CoreAction(e_BackendAction.high_pass, "high_pass", false, false);
action_short_pass = CoreAction(e_BackendAction.short_pass, "short_pass", false, false);
action_shot = CoreAction(e_BackendAction.shot, "shot", false, false);
action_keeper_rush = CoreAction(
    e_BackendAction.keeper_rush, "keeper_rush", true, false);
action_sliding = CoreAction(e_BackendAction.sliding, "sliding", false, false);
action_pressure = CoreAction(
    e_BackendAction.pressure, "pressure", true, false);
action_team_pressure = CoreAction(
    e_BackendAction.team_pressure, "team_pressure", true, false);
action_switch = CoreAction(e_BackendAction.switch, "switch", false, false);
action_sprint = CoreAction(e_BackendAction.sprint, "sprint", true, false);
action_dribble = CoreAction(
    e_BackendAction.dribble, "dribble", true, false);
action_release_direction = CoreAction(
    e_BackendAction.release_direction, "release_direction", true, false);
action_release_long_pass = CoreAction(e_BackendAction.release_long_pass,
                                      "release_long_pass", false, false);
action_release_high_pass = CoreAction(e_BackendAction.release_high_pass,
                                      "release_high_pass", false, false);
action_release_short_pass = CoreAction(e_BackendAction.release_short_pass,
                                       "release_short_pass", false, false);
action_release_shot = CoreAction(e_BackendAction.release_shot, "release_shot", false, false);
action_release_keeper_rush = CoreAction(e_BackendAction.release_keeper_rush,
                                        "release_keeper_rush", false, false);
action_release_sliding = CoreAction(e_BackendAction.release_sliding,
                                    "release_sliding", false, false);
action_release_pressure = CoreAction(e_BackendAction.release_pressure,
                                     "release_pressure", false, false);
action_release_team_pressure = CoreAction(e_BackendAction.release_team_pressure,
                                          "release_team_pressure", false, false);
action_release_switch = CoreAction(e_BackendAction.release_switch,
                                   "release_switch", false, false);
action_release_sprint = CoreAction(e_BackendAction.release_sprint,
                                   "release_sprint", false, false);
action_release_dribble = CoreAction(e_BackendAction.release_dribble,
                                    "release_dribble", false, false);

std::vector<CoreAction> action_set_dict_default = {action_idle, action_left, action_top_left, action_top,
      action_top_right, action_right, action_bottom_right,
      action_bottom, action_bottom_left, action_long_pass,
      action_high_pass, action_short_pass, action_shot,
      action_sprint, action_release_direction, action_release_sprint,
      action_sliding, action_dribble, action_release_dribble};
std::vector<CoreAction> action_set_dict_v2 = {action_idle, action_left, action_top_left, action_top,
      action_top_right, action_right, action_bottom_right,
      action_bottom, action_bottom_left, action_long_pass,
      action_high_pass, action_short_pass, action_shot,
      action_sprint, action_release_direction, action_release_sprint,
      action_sliding, action_dribble, action_release_dribble, action_builtin_ai};
std::vector<CoreAction> action_set_dict_full = {action_idle, action_left, action_top_left, action_top,
      action_top_right, action_right, action_bottom_right,
      action_bottom, action_bottom_left, action_long_pass,
      action_high_pass, action_short_pass, action_shot,
      action_keeper_rush, action_sliding, action_pressure,
      action_team_pressure, action_switch, action_sprint,
      action_dribble, action_release_direction,
      action_release_long_pass, action_release_high_pass,
      action_release_short_pass, action_release_shot,
      action_release_keeper_rush, action_release_sliding,
      action_release_pressure, action_release_team_pressure,
      action_release_switch, action_release_sprint,
      action_release_dribble, action_builtin_ai};
class FootballEnv : public Env<FootballEnvSpec> {
  protected:
    const std::map<std::string, int> player_config;
    player_config.insert(pair<std::string, int>("index", 0));
    const int agent_index = -1;
    const int agent_left_position = -1;
    const int agent_right_position = -1;

    std::string env_name_, representation_, rewards_, logdir_;
    bool stacked_, write_goal_dumps_, write_full_episode_dumps_, render_, write_video_;
    int dump_frequency_, number_of_left_players_agent_controls_, number_of_right_players_agent_controls_;
    std::tuple channel_dimensions_;
    std::map other_config_option_;
    void* extra_players_;

  public:
    FootballEnv(const Spec& spec, int env_id)
      : Env<FootballEnvSpec>(spec, env_id),
      env_name_(spec.config["env_name"_]),
      representation_(spec.config["representation"_]),
      rewards_(spec.config["rewards"_]),
      logdir_(spec.config["logdir"_]),
      stacked_(spec.config["stacked"_]),
      write_goal_dumps_(spec.config["write_goal_dumps"_]),
      write_full_episode_dumps_(spec.config["write_full_episode_dumps"_]),
      render_(spec.config["render"_]),
      write_video_(spec.config["write_video"_]),
      dump_frequency_(spec.config["dump_frequency"_]),
      number_of_left_players_agent_controls_(spec.config["number_of_left_players_agent_controls"_]),
      number_of_right_players_agent_controls_(spec.config["number_of_right_players_agent_controls"]),
      channel_dimensions_(spec.config["channel_dimensions"_]),
      other_config_option_(spec.config["other_config_option"_]),
      extra_players_(spec.config["extra_players"_]){}
    
    void reset() override {
      clock_t episode_start = clock();
      std::string action_set_name = spec.config["action_set"];
      std::vector<CoreAction> action_set;
      if(action_set_name = "default"){
        action_set = action_set_dict_default;
      }
      else if(action_set_name = "v2"){
        action_set = action_set_dict_v2;
      }
      else{
        action_set = action_set_dict_full;
      }
      
    }

    void step(const Action& action) override {
      
    }
  private:

}
