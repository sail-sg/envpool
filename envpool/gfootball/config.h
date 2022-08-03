#include <string>
#include <vector>
#include <direct.h>
#include "game_env.hpp"

class Config{
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
        std::string tracesdir_pre = getcwd(NULL, 0);
        std::string tracesdir = tracesdir_pre + "/dump";
        std::string video_format = "avi";
        int video_quality_level = 0;
        bool write_video = false;
        int episode_number = 0;
        Config();
        void NewScenario(int inc = 1){
          this->episode_number += inc;
          ScenarioConfig scenario_config;
        };

};