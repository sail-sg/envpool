#ifndef ENVPOOL_FOOTBALL_ENV_H_
#define ENVPOOL_FOOTBALL_ENV_H_
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include "game_env.hpp"
#include "config.h"
#include "football_action_set.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "player.h"

int SMM_WIDTH = 96;
int SMM_HEIGHT = 72;
float MINIMAP_NORM_X_MIN = -1.0;
float MINIMAP_NORM_X_MAX = 1.0;
float MINIMAP_NORM_Y_MIN = -1.0 / 2.25;
float MINIMAP_NORM_Y_MAX = 1.0 / 2.25;
int _MARKER_VALUE = 255;

class FootballEnvCore;
class FootballEnvFns;
class PreFootballEnv;

std::vector<std::string>SMM_LAYERS = {"left_team", "right_team", "ball", "active"};

namespace football{

class FootballEnvFns {
  public:
    //static char *tracesdir_pre_char;
    //static std::string tracesdir_pre;
    //static std::string tracesdir;
    static decltype(auto) DefaultConfig(){
      return MakeDict(
        "env_name"_.Bind(std::string("")), 
        "stacked"_.Bind(false), 
        "representation"_.Bind(std::string("extracted")), 
        "rewards"_.Bind(std::string("scoring")),
        "write_goal_dumps"_.Bind(false), 
        "write_full_episode_dumps"_.Bind(false), 
        "render"_.Bind(false), 
        "write_video"_.Bind(false), 
        "dump_frequency"_.Bind(1), 
        "extra_players"_.Bind(0), 
        "number_of_left_players_agent_controls"_.Bind(1), 
        "number_of_right_players_agent_controls"_.Bind(0), 
        "channel_dimensions"_.Bind(std::vector<int>{SMM_WIDTH, SMM_HEIGHT})
      );
    }

    template <typename Config>
    static decltype(auto) StateSpec(const Config& conf) {
      if(conf["representation"_] == "extracted" && conf["stacked"_] == false){
        return MakeDict(
          "obs"_.Bind(Spec<int>({72, 96, 4}, {0, 255})),
          "info:episode_reward"_.Bind(Spec<float>({})),
          "info:score"_.Bind(Spec<int>({2})),
          "info:steps"_.Bind(Spec<int>({})),
          "info:FPS"_.Bind(Spec<float>({})),
          "info:gameFPS"_.Bind(Spec<float>({}))
        );
      }
      else if(conf["representation"_] == "extracted" && conf["stacked"_] == true){
        return MakeDict(
          "obs"_.Bind(Spec<int>({72, 96, 16}, {0, 255})),
          "info:episode_reward"_.Bind(Spec<float>({})),
          "info:score"_.Bind(Spec<int>({2})),
          "info:steps"_.Bind(Spec<int>({})),
          "info:FPS"_.Bind(Spec<float>({})),
          "info:gameFPS"_.Bind(Spec<float>({}))
        );
      }
      else if(conf["representation"_] == "pixels" && conf["stacked"_] == false){
        return MakeDict(
          "obs"_.Bind(Spec<int>({72, 96, 3}, {0, 255})),
          "info:episode_reward"_.Bind(Spec<float>({})),
          "info:score"_.Bind(Spec<int>({2})),
          "info:steps"_.Bind(Spec<int>({})),
          "info:FPS"_.Bind(Spec<float>({})),
          "info:gameFPS"_.Bind(Spec<float>({}))
        );
      }
      else if(conf["representation"_] == "pixels" && conf["stacked"_] == true){
        return MakeDict(
          "obs"_.Bind(Spec<int>({72, 96, 12}, {0, 255})),
          "info:episode_reward"_.Bind(Spec<float>({})),
          "info:score"_.Bind(Spec<int>({2})),
          "info:steps"_.Bind(Spec<int>({})),
          "info:FPS"_.Bind(Spec<float>({})),
          "info:gameFPS"_.Bind(Spec<float>({}))
        );
      }
      else if(conf["representation"_] == "pixels_gray" && conf["stacked"_] == false){
        return MakeDict(
          "obs"_.Bind(Spec<int>({72, 96, 1}, {0, 255})),
          "info:episode_reward"_.Bind(Spec<float>({})),
          "info:score"_.Bind(Spec<int>({2})),
          "info:steps"_.Bind(Spec<int>({})),
          "info:FPS"_.Bind(Spec<float>({})),
          "info:gameFPS"_.Bind(Spec<float>({}))
        );
      }
      else if(conf["representation"_] == "pixels_gray" && conf["stacked"_] == true){
        return MakeDict(
          "obs"_.Bind(Spec<int>({72, 96, 4}, {0, 255})),
          "info:episode_reward"_.Bind(Spec<float>({})),
          "info:score"_.Bind(Spec<int>({2})),
          "info:steps"_.Bind(Spec<int>({})),
          "info:FPS"_.Bind(Spec<float>({})),
          "info:gameFPS"_.Bind(Spec<float>({}))
        );
      }
      else if(conf["representation"_] == "simple115" || conf["representation"_] == "simple115v2"){
        return MakeDict(
          "obs"_.Bind(Spec<float>({115})),
          "info:episode_reward"_.Bind(Spec<float>({})),
          "info:score"_.Bind(Spec<int>({2})),
          "info:steps"_.Bind(Spec<int>({})),
          "info:FPS"_.Bind(Spec<float>({})),
          "info:gameFPS"_.Bind(Spec<float>({}))
        );
      }
    }
    template <typename Config>
    static decltype(auto) ActionSpec(const Config& conf) {
      Config_football c;
      int num_actions = get_action_set(c).size();
      int number_of_players_agent_controls = conf.number_of_left_players_agent_controls + conf.number_of_right_players_agent_controls + conf.extra_players;
      if(number_of_players_agent_controls > 1){
        return MakeDict(
          "action"_.Bind(Spec<int>({number_of_players_agent_controls}, {0, num_actions}))
      );
      }
      return MakeDict(
        "action"_.Bind(Spec<int>({}, {0, num_actions}))
      );
    }

    FootballEnvFns(){
    }
};

using FootballEnvSpec = EnvSpec<FootballEnvFns>;

struct Observation{
  std::vector<float> ball = {0, 0, 0};
  std::vector<float> ball_direction = std::vector<float>(3);
  std::vector<float> ball_rotation = std::vector<float>(3);
  int ball_owned_team = 0;
  int ball_owned_player = 0;
  int game_mode = 0;

  std::vector<std::vector<float> > left_team = std::vector<std::vector<float> >(11, std::vector<float>(2));
  std::vector<e_PlayerRole> left_team_roles = std::vector<e_PlayerRole>(11);
  std::vector<std::vector<float> > left_team_direction = std::vector<std::vector<float> >(11, std::vector<float>(2));
  std::vector<float> left_team_tired_factor = std::vector<float>(11);
  std::vector<bool> left_team_yellow_card = std::vector<bool>(11);
  std::vector<bool> left_team_active = std::vector<bool>(11);
  int left_team_designated_player = 0;

  std::vector<std::vector<float> > right_team = std::vector<std::vector<float> >(11, std::vector<float>(2));
  std::vector<e_PlayerRole> right_team_roles = std::vector<e_PlayerRole>(11);
  std::vector<std::vector<float> > right_team_direction = std::vector<std::vector<float> >(11, std::vector<float>(2));
  std::vector<float> right_team_tired_factor = std::vector<float>(11);
  std::vector<bool> right_team_yellow_card = std::vector<bool>(11);
  std::vector<bool> right_team_active = std::vector<bool>(11);
  int right_team_designated_player = 0;

  std::vector<int> left_agent_controlled_player = std::vector<int>(1);
  std::vector<int> right_agent_controlled_player = std::vector<int>(0);
  std::vector<std::vector<int> > left_agent_sticky_actions = std::vector<std::vector<int> >(1, std::vector<int>(10));
  std::vector<std::vector<int> > right_agent_sticky_actions = std::vector<std::vector<int> >(1, std::vector<int>(10));
  std::vector<int> score = std::vector<int>(2);
  int steps_left = 0;
  Observation(){};
};

struct ConvertObservations : Observation{
  int designated = 0;
  int active = -1;
  std::vector<int>sticky_actions = {};
};

struct FootballEnvState{
    int previous_score_diff = 0;
    int previous_game_mode = -1;
    int prev_ball_owned_team = -1;
    FootballEnvState(){};
};

class FootballEnvCore{
  public:
    Config_football _config;
    Observation _observation;
    std::vector<CoreAction> _sticky_actions;
    std::vector<CoreAction> _action_set = {};
    clock_t _episode_start = 0;
    clock_t _steps_time = 0;
    bool _use_rendering_engine;
    int _cumulative_reward = 0;
    int _step_count = 0;
    int _step = 0;
    int reward = 0;
    float _fps = 0;
    float _game_fps = 0;
    bool _episode_done = false;
    FootballEnvState _state = FootballEnvState();
    GameEnv _env;
    FootballEnvCore(Config_football config){
      _config = config;
      _sticky_actions = get_sticky_actions(config);
      _use_rendering_engine = false;
      _env = _get_new_env();
      reset(0);
      
    }
    GameEnv _get_new_env(){
      auto env = GameEnv();
      env.game_config.physics_steps_per_frame = _config.physics_steps_per_frame;
      env.game_config.render_resolution_x = _config.render_resolution_x;
      env.game_config.render_resolution_y = _config.render_resolution_y;
      return env;
    }

    bool reset(int inc = 1){
      _episode_start = clock();
      _action_set = get_action_set(_config);
      _cumulative_reward = 0;
      _step_count = 0;
      _reset(_env.game_config.render, inc);
      if(!_retrieve_observation()){
        _env.step();
      }
      return true;
    }

    void _reset(bool animations, int inc) {
        if(_env.state == GameState::game_created){
          _env.start_game();
        }
        _env.state = GameState::game_running;
        auto scenario_config = ScenarioConfig::make();
        _env.reset(*scenario_config, animations);
    };

    bool _retrieve_observation(){
      int i = 0;
      auto info = _env.get_info();
      _observation.ball = {info.ball_position.value[0], info.ball_position.value[1], info.ball_position.value[2]};
      _observation.ball_direction = {info.ball_direction.value[0], info.ball_direction.value[1], info.ball_direction.value[2]};
      _observation.ball_rotation = {info.ball_rotation.value[0], info.ball_rotation.value[1], info.ball_rotation.value[2]};
      int left_players_num = info.left_team.size();
      int right_players_num = info.right_team.size();
      std::vector<std::vector<float> >left_team(left_players_num, std::vector<float>(2));
      std::vector<std::vector<float> >right_team(right_players_num, std::vector<float>(2));
      std::vector<float>temp;
      std::vector<std::vector<float> >left_team_direction(left_players_num, std::vector<float>(2));
      std::vector<std::vector<float> >right_team_direction(right_players_num, std::vector<float>(2));
      std::vector<float>left_team_tired_factor;
      std::vector<float>right_team_tired_factor;
      std::vector<bool>left_team_active;
      std::vector<bool>right_team_active;
      std::vector<bool>left_team_yellow_card;
      std::vector<bool>right_team_yellow_card;
      std::vector<e_PlayerRole>left_team_roles;
      std::vector<e_PlayerRole>right_team_roles;
      int left_designated_player = -1;
      int right_designated_player = -1;
      int left_team_designated_player;
      int right_team_designated_player;
      for(i = 0; i < left_players_num; i++){
        temp.push_back(info.left_team[i].player_position.value[0]);
        temp.push_back(info.left_team[i].player_position.value[1]);
        left_team.push_back(temp);
        temp.clear();
        temp.push_back(info.left_team[i].player_direction.value[0]);
        temp.push_back(info.left_team[i].player_direction.value[1]);
        left_team_direction.push_back(temp);
        temp.clear();
        left_team_tired_factor.push_back(info.left_team[i].tired_factor);
        left_team_active.push_back(info.left_team[i].is_active);
        left_team_yellow_card.push_back(info.left_team[i].has_card);
        left_team_roles.push_back(info.left_team[i].role);
        if(info.left_team[i].designated_player){
          left_designated_player = i;
        }
      }
      left_team_designated_player = left_designated_player;
      _observation.left_team.resize(left_players_num);
      _observation.left_team_direction.resize(left_players_num);
      _observation.left_team_tired_factor.resize(left_players_num);
      _observation.left_team_active.resize(left_players_num);
      _observation.left_team_yellow_card.resize(left_players_num);
      _observation.left_team_roles.resize(left_players_num);
      _observation.left_team = left_team;
      _observation.left_team_direction = left_team_direction;
      _observation.left_team_tired_factor = left_team_tired_factor;
      _observation.left_team_active = left_team_active;
      _observation.left_team_yellow_card = left_team_yellow_card;
      _observation.left_team_roles = left_team_roles;
      _observation.left_team_designated_player = left_team_designated_player;

      for(i = 0; i < right_players_num; i++){
        temp.push_back(info.right_team[i].player_position.value[0]);
        temp.push_back(info.right_team[i].player_position.value[1]);
        right_team.push_back(temp);
        temp.clear();
        temp.push_back(info.right_team[i].player_direction.value[0]);
        temp.push_back(info.right_team[i].player_direction.value[1]);
        right_team_direction.push_back(temp);
        temp.clear();
        right_team_tired_factor.push_back(info.right_team[i].tired_factor);
        right_team_active.push_back(info.right_team[i].is_active);
        right_team_yellow_card.push_back(info.right_team[i].has_card);
        right_team_roles.push_back(info.right_team[i].role);
        if(info.right_team[i].designated_player){
          right_designated_player = i;
        }
      }
      right_team_designated_player = right_designated_player;

      _observation.right_team.resize(right_players_num);
      _observation.right_team_direction.resize(right_players_num);
      _observation.right_team_tired_factor.resize(right_players_num);
      _observation.right_team_active.resize(right_players_num);
      _observation.right_team_yellow_card.resize(right_players_num);
      _observation.right_team_roles.resize(right_players_num);
      _observation.right_team = right_team;
      _observation.right_team_direction = right_team_direction;
      _observation.right_team_tired_factor = right_team_tired_factor;
      _observation.right_team_active = right_team_active;
      _observation.right_team_yellow_card = right_team_yellow_card;
      _observation.right_team_roles = right_team_roles;
      _observation.right_team_designated_player = right_team_designated_player;

      std::vector<int>left_agent_controlled_player;
      std::vector<std::vector<int> >left_agent_sticky_actions;
      for(i = 0; i < _env.config().left_agents; i++){
        left_agent_controlled_player.push_back(info.left_controllers[i].controlled_player);
        left_agent_sticky_actions.push_back(sticky_actions_state(true, i));
      }
      std::vector<int>right_agent_controlled_player;
      std::vector<std::vector<int> >right_agent_sticky_actions;
      for(i = 0; i < _env.config().right_agents; i++){
        right_agent_controlled_player.push_back(info.right_controllers[i].controlled_player);
        right_agent_sticky_actions.push_back(sticky_actions_state(false, i));
      }
      _observation.left_agent_controlled_player.resize(left_agent_controlled_player.size());
      _observation.left_agent_controlled_player = left_agent_controlled_player;
      _observation.right_agent_controlled_player.resize(right_agent_controlled_player.size());
      _observation.right_agent_controlled_player = right_agent_controlled_player;
      _observation.left_agent_sticky_actions.resize(left_agent_sticky_actions.size());
      for(int sticky_index = 0; sticky_index < left_agent_sticky_actions.size(); sticky_index++){
        _observation.left_agent_sticky_actions[sticky_index].resize(left_agent_sticky_actions[sticky_index].size());
      }
      _observation.left_agent_sticky_actions = left_agent_sticky_actions;
      _observation.left_agent_sticky_actions.resize(left_agent_sticky_actions.size());
      for(int sticky_index = 0; sticky_index < right_agent_sticky_actions.size(); sticky_index++){
        _observation.right_agent_sticky_actions[sticky_index].resize(right_agent_sticky_actions[sticky_index].size());
      }
      _observation.right_agent_sticky_actions = right_agent_sticky_actions;
      _observation.game_mode = int(info.game_mode);
      _observation.score = {info.left_goals, info.right_goals};
      _observation.ball_owned_team = info.ball_owned_team;
      _observation.ball_owned_player = info.ball_owned_player;
      _observation.steps_left = _env.config().game_duration - info.step;
      _step = info.step;
      return info.is_in_play;
    }

    std::vector<int>sticky_actions_state(bool left_team, int player_id){
      std::vector<int>result;
      for(int a = 0; a < _sticky_actions.size(); a++){
        result.push_back(int(_env.sticky_actions_state(_sticky_actions[a]._backend_action, left_team, player_id)));
      }
      return result;
    }

    void step(std::vector<int> action){
      std::vector<CoreAction>action_set;
      for(int a = 0; a < action.size(); a++){
        action_set.push_back(named_action_from_action_set(_action_set, action[a]));
      }
      _step_count += 1;
      int action_index = 0;
      for(int left_team = 1; left_team >= 0; left_team--){
        int agents = 1;
        if(left_team == 1){
          agents = _env.config().left_agents;
        }
        else{
          agents = _env.config().right_agents;
        }
        for(int i = 0; i < agents; i++){
          CoreAction player_action = action_set[action_index];
          if(_env.waiting_for_game_count == 20){
            player_action = action_short_pass;
          }
          else if(_env.waiting_for_game_count > 20){
            player_action = action_idle;
            std::vector<int>controlled_players;
            if(left_team == 1){
              controlled_players = _observation.left_agent_controlled_player;
            }
            else{
              controlled_players = _observation.right_agent_controlled_player;
            }
            if(_observation.ball_owned_team != -1 && _observation.ball_owned_team ^ left_team && controlled_players[i] == _observation.ball_owned_player){
              if((_env.waiting_for_game_count < 30) != left_team){
                player_action = action_left;
              }
              else{
                player_action = action_right;
              }
            }
          }
          action_index += 1;
          _env.action(player_action._backend_action, left_team, i);
        }
      }
      while(true){
        clock_t enter_time = clock();
        _env.step();
        _steps_time += clock() - enter_time;
        if(_retrieve_observation()){
          break;
        }
      }

      if(_env.config().end_episode_on_score){
        if(_observation.score[0] > 0 || _observation.score[1] > 0){
          _env.state = GameState::game_done;
        }
      }

      if(_env.config().end_episode_on_out_of_play && _observation.game_mode != int(e_GameMode::e_GameMode_Normal) && _state.previous_game_mode == int(e_GameMode::e_GameMode_Normal)){
        _env.state = GameState::game_done;
      }
      _state.previous_game_mode = _observation.game_mode;

      if(_env.config().end_episode_on_possession_change && _observation.ball_owned_team != -1 && _state.prev_ball_owned_team != -1 && _observation.ball_owned_team != _state.prev_ball_owned_team){
        _env.state = GameState::game_done;
      }

      if(_observation.ball_owned_team != -1){
        _state.prev_ball_owned_team = _observation.ball_owned_team;
      }

      int score_diff = _observation.score[0] - _observation.score[1];
      reward = score_diff - _state.previous_score_diff;
      _state.previous_score_diff = score_diff;
      if(_observation.game_mode != int(e_GameMode::e_GameMode_Normal)){
        _env.waiting_for_game_count += 1;
      }
      else{
        _env.waiting_for_game_count = 0;
      }
      if(_step >= _env.config().game_duration){
        _env.state = GameState::game_done;
      }

      _episode_done = _env.state == GameState::game_done;
      clock_t end_time = clock();
      _cumulative_reward += reward;
      if(_episode_done){
        _fps = _step_count / (end_time - _episode_start);
        _game_fps = _step_count / _steps_time;
      }
    }

    Observation observation(){
      return _observation;
    }
};

class PreFootballEnv{
  public:
    Config_football _config;
    PlayerFootball _agent;
    int _agent_index = -1;
    int _agent_left_position = -1;
    int _agent_right_position = -1;
    int player_index = 0;
    std::vector<PlayerFootball> _players;
    FootballEnvCore _env = FootballEnvCore(_config);
    int _num_actions = get_action_set(_config).size();
    PreFootballEnv(Config_football cfg){
      _config = cfg;
      _players = _construct_players(cfg.players, player_index);
    }
    std::vector<PlayerFootball> _construct_players(std::vector<int> definitions, int player_config){
      std::vector<PlayerFootball> result;
      int left_position = 0;
      int right_position = 0;
      auto player = PlayerFootball(definitions, _config);
      result.push_back(player);
      player_config += 1;
      _agent = player;
      _agent_index = player.num_controlled_players();
      _agent_left_position = left_position;
      _agent_right_position = right_position;
      left_position += player.num_controlled_left_players();
      right_position += player.num_controlled_right_players();
      return result;
    }
    std::vector<ConvertObservations> _convert_observations(Observation original, PlayerFootball player, int left_player_position, int right_player_position){
      std::vector<ConvertObservations> observations;
      Observation adopted;
      adopted = original;
      std::string prefix = "left";
      int position = left_player_position;
      ConvertObservations o;
      for(int x = 0; x < player.num_controlled_left_players(); x++){
        o.ball                    = adopted.ball;                
        o.ball_direction          = adopted.ball_direction;         
        o.ball_rotation           = adopted.ball_rotation;          
        o.ball_owned_team         = adopted.ball_owned_team;        
        o.ball_owned_player       = adopted.ball_owned_player;      
        o.left_team               = adopted.left_team;               
        o.left_team_direction     = adopted.left_team_direction;    
        o.left_team_tired_factor  = adopted.left_team_tired_factor; 
        o.left_team_yellow_card   = adopted.left_team_yellow_card;  
        o.left_team_active        = adopted.left_team_active;       
        o.left_team_roles         = adopted.left_team_roles;        
        o.right_team              = adopted.right_team;             
        o.right_team_direction    = adopted.right_team_direction;    
        o.right_team_tired_factor = adopted.right_team_tired_factor; 
        o.right_team_yellow_card  = adopted.right_team_yellow_card;    
        o.right_team_active       = adopted.right_team_active;      
        o.right_team_roles        = adopted.right_team_roles;       
        o.score                   = adopted.score;                   
        o.steps_left              = adopted.steps_left;             
        o.game_mode               = adopted.game_mode; 
        o.designated = adopted.left_team_designated_player;
        if(position + x >= adopted.left_agent_controlled_player.size()){
          o.active = -1;
          o.sticky_actions = {};
        }
        else{
          o.active = adopted.left_agent_controlled_player[position + x];
          o.sticky_actions = adopted.left_agent_sticky_actions[position + x];
        }
        observations.push_back(o);
      }
      return observations;

    }
    std::vector<int>_get_actions(){
      Observation obs = this->_env.observation();
      std::vector<int>actions;
      int left_player_position = 0;
      int right_player_position = 0;
      for(int i = 0; i < _players.size(); i++){
        auto adopted_obs = _convert_observations(obs, _players[i], left_player_position, right_player_position);
        left_player_position += _players[i].num_controlled_left_players();
        right_player_position += _players[i].num_controlled_right_players();
        auto a = _players[i].take_action();
        int j = 0;
        for(j = 0; j < _players[i].num_controlled_left_players(); j++){
          actions.push_back(a[j]);
        }
        for(; j < _players[i].num_controlled_left_players() + _players[i].num_controlled_right_players()){
          actions.push_back(a[j]);
        }
      }
      return actions;
    }
    void reset(){
      _env.reset();
    }
    void step(std::vector<int>action){
      _env.step(_get_actions());
    }

    std::vector<ConvertObservations>observation(){
      return _convert_observations(_env.observation(), _agent, _agent_left_position, _agent_right_position);
    }
    std::vector<std::vector<std::vector<std::vector<int> > > >observation_smm(std::vector<ConvertObservations>obs, std::vector<int>channel_dimensions){
      std::vector<std::vector<std::vector<std::vector<int> > > >frame(obs.size(), std::vector<std::vector<std::vector<int> > >(channel_dimensions[1], std::vector<std::vector<int> >(channel_dimensions[0], std::vector<int>(SMM_LAYERS.size(), 0))));
      for(int i = 0; i < obs.size(); i++){
        for(int j = 0; j < SMM_LAYERS.size(); j++){
          if(SMM_LAYERS[j] == "active"){
            if(obs[i].active == -1){
              continue;
            }
            std::vector<float>left_team_active(2);
            left_team_active = obs[i].left_team[obs[i].active];
            int x = int((left_team_active[0] - MINIMAP_NORM_X_MIN) / (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame[0][0].size());
            int y = int((left_team_active[1] - MINIMAP_NORM_Y_MIN) / (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame[0].size());
            x = std::max(0, std::min(int(frame[0][0].size() - 1), x));
            y = std::max(0, std::min(int(frame[0].size() - 1), y));
            frame[i][y][x][j] = _MARKER_VALUE;
          }
          else if(SMM_LAYERS[j] == "left_team"){
            std::vector<float>left_team(obs[i].left_team.size() * obs[i].left_team[0].size());
            for(int m = 0; m < obs[i].left_team.size(); m++){
              for(int n = 0; n < obs[i].left_team[0].size(); n++){
                left_team[m * obs[i].left_team[0].size() + n] = obs[i].left_team[m][n];
              }
            }
            for(int p = 0; p < (left_team.size() / 2); p++){
              int x = int((left_team[p * 2] - MINIMAP_NORM_X_MIN) / (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame[0][0].size());
              int y = int((left_team[p * 2 + 1] - MINIMAP_NORM_Y_MIN) / (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame[0].size());
              x = std::max(0, std::min(int(frame[0][0].size() - 1), x));
              y = std::max(0, std::min(int(frame[0].size() - 1), y));
              frame[i][y][x][j] = _MARKER_VALUE;
            }
          }
          else if(SMM_LAYERS[j] == "right_team"){
            std::vector<float>right_team(obs[i].right_team.size() * obs[i].right_team[0].size());
            for(int m = 0; m < obs[i].right_team.size(); m++){
              for(int n = 0; n < obs[i].right_team[0].size(); n++){
                right_team[m * obs[i].right_team[0].size() + n] = obs[i].right_team[m][n];
              }
            }
            for(int p = 0; p < (right_team.size() / 2); p++){
              int x = int((right_team[p * 2] - MINIMAP_NORM_X_MIN) / (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame[0][0].size());
              int y = int((right_team[p * 2 + 1] - MINIMAP_NORM_Y_MIN) / (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame[0].size());
              x = std::max(0, std::min(int(frame[0][0].size() - 1), x));
              y = std::max(0, std::min(int(frame[0].size() - 1), y));
              frame[i][y][x][j] = _MARKER_VALUE;
            }
          }
          else if(SMM_LAYERS[j] == "ball"){
            for(int p = 0; p < (obs[i].ball.size() / 2); p++){
              int x = int((obs[i].ball[p * 2] - MINIMAP_NORM_X_MIN) / (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame[0][0].size());
              int y = int((obs[i].ball[p * 2 + 1] - MINIMAP_NORM_Y_MIN) / (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame[0].size());
              x = std::max(0, std::min(int(frame[0][0].size() - 1), x));
              y = std::max(0, std::min(int(frame[0].size() - 1), y));
              frame[i][y][x][j] = _MARKER_VALUE;
            }
          }
        }
      }
      return frame;
    }
    std::vector<std::vector<std::vector<int> > >observation_smm_single_agent(std::vector<std::vector<std::vector<std::vector<int> > > >obs){
      return obs[0];
    }
    std::vector<std::vector<float> >observation_simple115state(std::vector<ConvertObservations>obs, bool fixed_positions){
      std::vector<std::vector<float> >final_obs;
      std::vector<float>o;
      for(int i = 0; i < obs.size(); i++){
        if(fixed_positions){
          for(int j = 0; j < 4; j++){
            if(j == 0){
              o.insert(o.end(), do_flatten(obs[i].left_team).begin(), do_flatten(obs[i].left_team).end());
            }
            else if(j == 1){
              o.insert(o.end(), do_flatten(obs[i].left_team_direction).begin(), do_flatten(obs[i].left_team_direction).end());
            }
            else if(j == 2){
              o.insert(o.end(), do_flatten(obs[i].right_team).begin(), do_flatten(obs[i].right_team).end());
            }
            else if(j == 3){
              o.insert(o.end(), do_flatten(obs[i].right_team_direction).begin(), do_flatten(obs[i].right_team_direction).end());
            }
            if(o.size() < (j + 1) * 22){
              std::vector<float>zero(((j + 1) * 22 - o.size()), -1);
              o.insert(o.end(), zero.begin(), zero.end());
            }
          }
        }
        else{
          o.insert(o.end(), do_flatten(obs[i].left_team).begin(), do_flatten(obs[i].left_team).end());
          o.insert(o.end(), do_flatten(obs[i].left_team_direction).begin(), do_flatten(obs[i].left_team_direction).end());
          o.insert(o.end(), do_flatten(obs[i].right_team).begin(), do_flatten(obs[i].right_team).end());
          o.insert(o.end(), do_flatten(obs[i].right_team_direction).begin(), do_flatten(obs[i].right_team_direction).end());
        }

        if(o.size() < 88){
          std::vector<float>zero((88 - o.size()), -1);
        }
        o.insert(o.end(), obs[i].ball.begin(), obs[i].ball.end());
        o.insert(o.end(), obs[i].ball_direction.begin(), obs[i].ball_direction.end());
        std::vector<float>_ball_owned_team;
        if(obs[i].ball_owned_team == -1){
          _ball_owned_team = {1, 0, 0};
        }
        else if(obs[i].ball_owned_team == 0){
          _ball_owned_team = {0, 1, 0};
        }
        else if(obs[i].ball_owned_team == 1){
          _ball_owned_team = {0, 0, 1};
        }
        o.insert(o.end(), _ball_owned_team.begin(), _ball_owned_team.end());
        std::vector<float>active(11, 0);
        if(obs[i].active != -1){
          active[obs[i].active] = 1;
        }
        o.insert(o.end(), active.begin(), active.end());
        std::vector<float>game_mode(7, 0);
        game_mode[obs[i].game_mode] = 1;
        o.insert(o.end(), game_mode.begin(), game_mode.end());
        final_obs.push_back(o);
      }
      return final_obs;
    }
    std::vector<float>observation_simple115state_single_agent(std::vector<std::vector<float> >obs){
      return obs[0];
    }

    std::vector<float>do_flatten(std::vector<std::vector<float> >obj){
      std::vector<float>result;
      int x = obj.size();
      int y = obj[0].size();
      for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++){
          result[i * y + j] = obj[i][j];
        }
      }
      return result;
    }

};

class FootballEnv : public Env<FootballEnvSpec> {
  public:
    Config_football config;
    std::string env_name_, representation_, rewards_, logdir_;
    bool stacked_, write_goal_dumps_, write_full_episode_dumps_, render_, write_video_;
    int dump_frequency_, extra_players_, number_of_left_players_agent_controls_, number_of_right_players_agent_controls_;
    std::vector<int>channel_dimensions_;
    bool done_;
    PreFootballEnv env;
    FootballEnv(const Spec& spec, int env_id) : Env<FootballEnvSpec>(spec, env_id), 
      env_name_(spec.config["env_name"_]), 
      stacked_(spec.config["stacked"_]), 
      representation_(spec.config["representation"_]), 
      rewards_(spec.config["rewards"_]), 
      write_goal_dumps_(spec.config["write_goal_dumps"_]), 
      write_full_episode_dumps_(spec.config["write_full_episode_dumps"_]), 
      render_(spec.config["render"_]),
      write_video_(spec.config["write_video"_]),
      dump_frequency_(spec.config["dump_frequency"_]),
      logdir_(spec.config["logdir"_]),
      extra_players_(spec.config["extra_players"_]),
      number_of_left_players_agent_controls_(spec.config["number_of_left_players_agent_controls"_]),
      number_of_right_players_agent_controls_(spec.config["number_of_right_players_agent_controls"_]),
      channel_dimensions_(spec.config["channel_dimensions"_])
      {
        config.level = env_name_;
        std::vector<int> players = {number_of_left_players_agent_controls_, number_of_right_players_agent_controls_};
        if(extra_players_ != 0){
          players.push_back(extra_players_);
        }
        config.dump_full_episodes = write_full_episode_dumps_;
        config.dump_scores = write_goal_dumps_;
        config.players = players;
        config.tracesdir = logdir_;
        config.write_video = write_video_;
        env = PreFootballEnv(config);
      }

    void Reset() override{
      env.reset();
      std::vector<std::vector<std::vector<int> > >_observation_smm_single_agent = env.observation_smm_single_agent(env.observation_smm(env.observation(), channel_dimensions_));
      std::vector<float>_observation_simple115state_single_agent = env.observation_simple115state_single_agent(env.observation_simple115state(env.observation(), false));
      std::vector<float>_observation_simple115v2state_single_agent = env.observation_simple115state_single_agent(env.observation_simple115state(env.observation(), true));
      if(representation_ == "extracted" && stacked_ == false){
        WriteState_smm_single(_observation_smm_single_agent, 0, {0, 0}, 0, 0, 0);
      }
      else if(representation_ == "simple115"){
        WriteState_simple115_single(_observation_simple115state_single_agent, 0, {0, 0}, 0, 0, 0);
      }
      else if(representation_ == "simple115v2"){
        WriteState_simple115_single(_observation_simple115v2state_single_agent, 0, {0, 0}, 0, 0, 0);
      }
    }

    void Step(const Action& action) override {
      std::vector<int>act_set = {};
      if(number_of_left_players_agent_controls_ + number_of_right_players_agent_controls_ <= 1){
        int act = action["action"_];
        act_set.push_back(act);
      }
      else{
        std::vector<int>act = action["action"_];
        act_set.resize(act.size());
        act_set = act;
      }
      env.step(act_set);
      std::vector<std::vector<std::vector<int> > >_observation_smm_single_agent = env.observation_smm_single_agent(env.observation_smm(env.observation(), channel_dimensions_));
      std::vector<float>_observation_simple115state_single_agent = env.observation_simple115state_single_agent(env.observation_simple115state(env.observation(), false));
      std::vector<float>_observation_simple115v2state_single_agent = env.observation_simple115state_single_agent(env.observation_simple115state(env.observation(), true));
      if(representation_ == "extracted" && stacked_ == false){
        WriteState_smm_single(_observation_smm_single_agent, env._env._cumulative_reward, env._env._observation.score, env._env._step_count, env._env._fps, env._env._game_fps);
      }
      else if(representation_ == "simple115"){
        WriteState_simple115_single(_observation_simple115state_single_agent, env._env._cumulative_reward, env._env._observation.score, env._env._step_count, env._env._fps, env._env._game_fps);
      }
      else if(representation_ == "simple115v2"){
        WriteState_simple115_single(_observation_simple115v2state_single_agent, env._env._cumulative_reward, env._env._observation.score, env._env._step_count, env._env._fps, env._env._game_fps);
      }
    };

  private:
    void WriteState_smm_single(std::vector<std::vector<std::vector<int> > >obs, float episode_reward, std::vector<int>score, int steps, float fps, float game_fps) {
      State state = Allocate();
      for(int i = 0; i < channel_dimensions_[1]; i++){
        for(int j = 0; j < channel_dimensions_[0]; j++){
          for(int k = 0; k < 4; k++){
            state["obs"_][i][j][k] = obs[i][j][k];
          }
        }
      }
      state["info:episode_reward"_] = episode_reward;
      state["info:score"_][0] = score[0];
      state["info:score"_][1] = score[1];
      state["info:steps"_] = steps;
      state["info:fps"_] = fps;
      state["info:game_fps"_] = game_fps;
    }
    void WriteState_simple115_single(std::vector<float>obs, float episode_reward, std::vector<int>score, int steps, float fps, float game_fps){
      State state = Allocate();
      for(int i = 0; i < 115; i++){
        state["obs"_][i] = obs[i];
      }
      state["info:episode_reward"_] = episode_reward;
      state["info:score"_][0] = score[0];
      state["info:score"_][1] = score[1];
      state["info:steps"_] = steps;
      state["info:fps"_] = fps;
      state["info:game_fps"_] = game_fps;
    }
};

using FootballEnvPool = AsyncEnvPool<FootballEnv>;

}

#endif //#ifndef ENVPOOL_FOOTBALL_ENV_H_