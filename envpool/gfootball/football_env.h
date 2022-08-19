#ifndef ENVPOOL_FOOTBALL_ENV_H_
#define ENVPOOL_FOOTBALL_ENV_H_

#include <string>
#include <stdio.h>
#include <unistd.h>
#include "game_env.hpp"
#include "config.h"
#include "football_action_set.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace football{

class FootballEnvFns {
  public:
    //static char *tracesdir_pre_char;
    //static std::string tracesdir_pre;
    //static std::string tracesdir;
    static decltype(auto) DefaultConfig(){
      return MakeDict(
        "action_set"_.Bind(std::string("default")), "custom_display_stats"_.Bind(std::vector<std::string>{}),  "display_game_stats"_.Bind(true), 
        "dump_full_episodes"_.Bind(false), "dump_scores"_.Bind(false), "players"_.Bind(std::vector<std::string>{"agent:left_players=1"}), 
        "level"_.Bind(std::string("11_vs_11_stochastic")), "physics_steps_per_frame"_.Bind(10), "render_resolution_x"_.Bind(1280), 
        "render_resolution_y"_.Bind(1280 * 0.5625), "real_time"_.Bind(false), 
        //"tracesdir"_.Bind(tracesdir),
        "video_format"_.Bind(std::string("avi")),
        "video_quality_level"_.Bind(0), "write_video"_.Bind(false)
      );
    }
    template <typename Config>
    static decltype(auto) StateSpec(const Config& conf) {
      return MakeDict(
        "obs"_.Bind(Spec<int>({-1, 72, 96, 16}, {0, 255}),
        "info:episode_reward"_.Bind(Spec<float>({})),
        "info:score"_.Bind(Spec<int>({2})),
        "info:steps"_.Bind(Spec<int>({})),
        "info:FPS"_.Bind(Spec<float>({})),
        "info:gameFPS"_.Bind(Spec<float>({}))
        )
      );
    }
    template <typename Config>
    static decltype(auto) ActionSpec(const Config& conf) {
      return MakeDict(
        "action"_.Bind(Spec<int>({-1}, {0, 32}))
      );
    }
    FootballEnvFns(){
      //tracesdir_pre_char = getcwd(NULL, 0);
      //tracesdir_pre = tracesdir_pre_char;
      //tracesdir = tracesdir_pre + "/dump";
    }
};

using FootballEnvSpec = EnvSpec<FootballEnvFns>;

struct observation{
  int num_players = 11;
  std::vector<int> left_agent_controlled_player = std::vector<int>(1);
  std::vector<int> right_agent_controlled_player = std::vector<int>(1);
  std::vector<float> left_team = std::vector<float>(num_players * 2);
  std::vector<float> left_team_roles = std::vector<float>(num_players);
  std::vector<float> left_team_direction = std::vector<float>(num_players * 2);
  std::vector<float> left_team_tired_factor = std::vector<float>(num_players);
  std::vector<float> left_team_yellow_card = std::vector<float>(num_players);
  std::vector<float> left_team_active = std::vector<float>(1);
  int left_team_designated_player = 0;
  std::vector<float> right_team = std::vector<float>(num_players * 2);
  std::vector<float> right_team_roles = std::vector<float>(num_players);
  std::vector<float> right_team_direction = std::vector<float>(num_players * 2);
  std::vector<float> right_team_tired_factor = std::vector<float>(num_players);
  std::vector<float> right_team_yellow_card = std::vector<float>(num_players);
  std::vector<float> right_team_active = std::vector<float>(1);
  int right_team_designated_player = 0;
  std::vector<int> ball = {0, 0, 0};
  std::vector<float> ball_direction = std::vector<float>(3);
  std::vector<float> ball_rotation = std::vector<float>(3);
  int ball_owned_team = 0;
  int ball_owned_player = 0;
  int game_mode = 0;
  std::vector<float> left_agent_sticky_actions = std::vector<float>(num_players);
  std::vector<float> right_agent_sticky_actions = std::vector<float>(num_players);
  std::vector<int> score = std::vector<int>(2);
  int steps_left = 0;
  observation(){};
};  
  
struct FootballEnvState{
    int previous_score_diff = 0;
    int previous_game_mode = -1;
    int prev_ball_owned_team = -1;
};

class FootballEnv : public Env<FootballEnvSpec> {
  protected:
    int previous_score_diff = 0;
    int previous_game_mode = -1;
    int prev_ball_owned_team = -1;

  public:
    std::string action_set_name_;
    std::string custom_display_stats_;
    bool display_game_stats_;
    bool dump_full_episodes_;
    bool dump_scores_;
    std::vector<std::string>players_;
    std::string level_;
    bool real_time_;
    //std::string tracesdir_;
    std::string video_format_;
    int video_quality_level_;
    bool write_video_;
    int episode_number = 0;
    boost::shared_ptr<ScenarioConfig> scenario_config = ScenarioConfig::make();
    GameContext* context = nullptr;
    boost::shared_ptr<GameConfig> game_config = GameConfig::make();
    GameState state = game_created;
    int waiting_for_game_count = 0;
    GameEnv env_;
    int physics_steps_per_frame_;
    int render_resolution_x_;
    int render_resolution_y_;
    int steps_time = 0;
    int step_count = 0;
    int _step = 0;
    clock_t episode_start = clock();
    std::vector<CoreAction> action_set;
    float cumulative_reward = 0;

    FootballEnv(const Spec& spec, int env_id) : Env<FootballEnvSpec>(spec, env_id),
      env_(GameEnv::GameEnv()),  
      action_set_name(spec.config["action_set"_]), 
      physics_steps_per_frame(spec.config["physics_steps_per_frame"_]), 
      render_resolution_x(spec.config["render_resolution_x"_]), 
      render_resolution_y(spec.config["render_resolution_y"_]), 
      custom_display_stats(spec.config["custom_display_stats"_]), 
      display_game_stats(spec.config["display_game_stats"_]), 
      dump_full_episodes(spec.config["dump_full_episodes"_]),
      dump_scores(spec.config["dump_scores"_]),
      players(spec.config["players"_]),
      level(spec.config["level"_]),
      real_time(spec.config["real_time"_]),
      //tracesdir(spec.config["tracesdir"_]),
      video_format(spec.config["video_format"_]),
      video_quality_level(spec.config["video_quality_level"_]),
      write_video(spec.config["write_video"_])
      {};


    void Step(const Action& action){
      observation obs;
      step_count += 1;
      int action_index = 0;
      std::vector<int> controlled_players;
      for(int left_team = 1; left_team > 0; left_team++){
        auto agents = env_.scenario_config.left_agents;
        for(int j = 0; j < agents; j++){
          auto player_action_index = action["action"_][action_index];
          CoreAction player_action = action_idle;
          switch (player_action_index)
          {
          case 0:
            player_action = action_idle;
            break;
          case 1:
            player_action = action_left;
            break;
          case 2:
            player_action = action_top_left;
            break;
          case 3:
            player_action = action_top;
            break;
          case 4:
            player_action = action_top_right;
            break;
          case 5:
            player_action = action_right;
            break;
          case 6:
            player_action = action_bottom_right;
            break;
          case 7:
            player_action = action_bottom;
            break;
          case 8:
            player_action = action_bottom_left;
            break;
          case 9:
            player_action = action_long_pass;
            break;
          case 10:
            player_action = action_high_pass;
            break;
          case 11:
            player_action = action_short_pass;
            break;
          case 12:
            player_action = action_shot;
            break;
          case 13:
            player_action = action_keeper_rush;
            break;
          case 14:
            player_action = action_sliding;
            break;
          case 15:
            player_action = action_pressure;
            break;
          case 16:
            player_action = action_team_pressure;
            break;
          case 17:
            player_action = action_switch;
            break;
          case 18:
            player_action = action_sprint;
            break;
          case 19:
            player_action = action_dribble;
            break;
          case 20:
            player_action = action_release_direction;
            break;
          case 21:
            player_action = action_release_long_pass;
            break;
          case 22:
            player_action = action_release_high_pass;
            break;
          case 23:
            player_action = action_release_short_pass;
            break;
          case 24:
            player_action = action_release_shot;
            break;
          case 25:
            player_action = action_release_keeper_rush;
            break;
          case 26:
            player_action = action_release_sliding;
            break;
          case 27:
            player_action = action_release_pressure;
            break;
          case 28:
            player_action = action_release_team_pressure;
            break;
          case 29:
            player_action = action_release_switch;
            break;
          case 30:
            player_action = action_release_sprint;
            break;
          case 31:
            player_action = action_release_dribble;
            break;
          case 32:
            player_action = action_builtin_ai;
            break;
          default: 0;
            break;
          }
          if(env_.waiting_for_game_count == 20){
            player_action = action_short_pass;
          }
          else if(env_.waiting_for_game_count > 20){
            player_action = action_idle;
            if(left_team == 0){
              controlled_players = obs.left_agent_controlled_player;
            }
            else{
              controlled_players = obs.right_agent_controlled_player;
            }
            if (obs.ball_owned_team != -1 && controlled_players[j] == obs.ball_owned_player && !(left_team ^ obs.ball_owned_team)){
              if(bool(env_.waiting_for_game_count < 30) != bool(left_team)){
                player_action = action_left;
              }
              else{
                player_action = action_right;
              }
            }
          }
          action_index += 1;
          env_.action(player_action.action_, bool(left_team), j);
        }
      }
      while(true){
        clock_t enter_time = clock();
        env_.step();
        steps_time += clock() - enter_time;
        if (retrieve_observation()){
          break;
        }
      }

      if(env_.scenario_config.end_episode_on_score){
        if(obs.score[0] > 0 || obs.score[1] > 0){
          env_.state = GameState::game_done;
        }
      }

      if(env_.scenario_config.end_episode_on_out_of_play && obs.game_mode != int(e_GameMode::e_GameMode_Normal) && previous_game_mode == int(e_GameMode::e_GameMode_Normal)){
        env_.state = GameState::game_done;
      }
      previous_game_mode = obs.game_mode;

      if(env_.scenario_config.end_episode_on_possession_change && 
        obs.ball_owned_team != -1 &&
        prev_ball_owned_team != -1 &&
        obs.ball_owned_team != prev_ball_owned_team){
          env_.state = GameState::game_done;
      }
      if(obs.ball_owned_team != -1){
        prev_ball_owned_team = obs.ball_owned_team;
      }

      int score_diff = obs.score[0] - obs.score[1];
      int reward = score_diff - previous_score_diff;
      previous_score_diff = score_diff;
      if(reward == 1){

      }
      if(obs.game_mode != int(e_GameMode::e_GameMode_Normal)){
        env_.waiting_for_game_count += 1;
      }
      else{
        env_.waiting_for_game_count = 0;
      }
      if(_step >= env_.scenario_config.game_duration){
        env_.state = GameState::game_done;
      }

      bool episode_done = env_.state == GameState::game_done;
      clock_t end_time = clock();
      cumulative_reward += reward;
      if(episode_done){
        float fps = step_count / (end_time - episode_start);
        float game_fps = step_count / steps_time;
      }
    };

    void setConfig(ScenarioConfig& game_config) {
        DO_VALIDATION;
        scenario_config.ball_position.coords[0] = scenario_config.ball_position.coords[0] * X_FIELD_SCALE;
        scenario_config.ball_position.coords[1] = scenario_config.ball_position.coords[1] * Y_FIELD_SCALE;
        std::vector<SideSelection> setup = GetMenuTask()->GetControllerSetup();
        CHECK(setup.size() == 2 * MAX_PLAYERS);
        int controller = 0;
        for (int x = 0; x < scenario_config.left_agents; x++) {
            DO_VALIDATION;
            setup[controller++].side = -1;
        }
        while (controller < MAX_PLAYERS) {
          DO_VALIDATION;
          setup[controller++].side = 0;
        }
        for (int x = 0; x < scenario_config.right_agents; x++) {
          DO_VALIDATION;
          setup[controller++].side = 1;
        }
        while (controller < 2 * MAX_PLAYERS) {
          DO_VALIDATION;
          setup[controller++].side = 0;
        }
        this->scenario_config = scenario_config;
        GetMenuTask()->SetControllerSetup(setup);
    };
        
    void reset_game(bool animations, int inc) {
        env_.game_config.physics_steps_per_frame = this->physics_steps_per_frame;
        env_.game_config.render_resolution_x = this->render_resolution_x;
        env_.game_config.render_resolution_y = this->render_resolution_y;
        this->config.NewScenario(inc);
        if(env_.state == GameState::game_created){
          env_.start_game();
        }
        env_.state = GameState::game_running;
        auto scenario_config_ = ScenarioConfig::make();
        env_.reset(*scenario_config_, animations);
    };

    void Reset (int inc = 1) {
      episode_start = clock();
      action_set = get_action_set(this->config);
      cumulative_reward = 0;
      step_count = 0;
      reset_game(env_.game_config.render, inc);
    }

    bool retrieve_observation(){
      auto info = env_.get_info();
      return info.is_in_play;
    }

    std::vector<CoreAction> get_action_set(Config config){
      std::vector<CoreAction> action_set;
      if(config.action_set == "default"){
        action_set = {action_idle, action_left, action_top_left, action_top,
          action_top_right, action_right, action_bottom_right,
          action_bottom, action_bottom_left, action_long_pass,
          action_high_pass, action_short_pass, action_shot,
          action_sprint, action_release_direction, action_release_sprint,
          action_sliding, action_dribble, action_release_dribble};
      }
      else if(config.action_set == "v2"){
        action_set = {
          action_idle, action_left, action_top_left, action_top,
          action_top_right, action_right, action_bottom_right,
          action_bottom, action_bottom_left, action_long_pass,
          action_high_pass, action_short_pass, action_shot,
          action_sprint, action_release_direction, action_release_sprint,
          action_sliding, action_dribble, action_release_dribble, action_builtin_ai
        };
      }
      else if(config.action_set == "full"){
        action_set = {
          action_idle, action_left, action_top_left, action_top,
          action_top_right, action_right, action_bottom_right,
          action_bottom, action_bottom_left, action_long_pass,
          action_high_pass, action_short_pass, action_shot,
          action_sprint, action_release_direction, action_release_sprint,
          action_sliding, action_dribble, action_release_dribble, action_builtin_ai,
          action_keeper_rush, action_pressure,
          action_team_pressure, action_switch,
          action_release_long_pass, action_release_high_pass,
          action_release_short_pass, action_release_shot,
          action_release_keeper_rush, action_release_sliding,
          action_release_pressure, action_release_team_pressure,
          action_release_switch,
        };
      }
      return action_set;
    }

};

using FootballEnvPool = AsyncEnvPool<FootballEnv>;
}

#endif //#ifndef ENVPOOL_FOOTBALL_ENV_H_
