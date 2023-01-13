#ifndef ENVPOOL_FOOTBALL_CONFIG_H_
#define ENVPOOL_FOOTBALL_CONFIG_H_

#include <string>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include "game_env.hpp"

class Config_football{
    public:
        std::string action_set = "default";
        std::string custom_display_stats = "";
        bool display_game_stats = true;
        bool dump_full_episodes = false;
        bool dump_scores = false;
        std::vector<std::string>players = {"agent:left_players=1"};
        std::string level = "11_vs_11_stochastic";
        int physics_steps_per_frame = 10;
        float render_resolution_x = 1280;
        float render_resolution_y = 0.5625 * render_resolution_x;
        bool real_time = false;
        char* tracesdir_pre_char = getcwd(NULL, 0);
        std::string tracesdir_pre = tracesdir_pre_char;
        std::string tracesdir = tracesdir_pre + "/dump";
        std::string video_format = "avi";
        int video_quality_level = 0;
        bool write_video = false;
        int episode_number = 0;
        Config_football(){ };
        void NewScenario(int inc = 1){
          this->episode_number += inc;
          auto scenario_config = ScenarioConfig::make();
        };
        int number_of_left_players = players[0];
        int number_of_right_players = players[1];
        int number_of_players_agent_controls();
};

int Config_football::number_of_players_agent_controls(){
  int sum = 0;
  for(int i = 0; i < this->players.size(); i++){
    sum += players[i];
  }
  return sum;
}

#endif  // ENVPOOL_FOOTBALL_CONFIG_H_
