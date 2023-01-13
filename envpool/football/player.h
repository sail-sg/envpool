#ifndef ENVPOOL_FOOTBALL_PLAYER_H_
#define ENVPOOL_FOOTBALL_PLAYER_H_

#include <string>
#include <vector>
#include <stdio.h>
#include "config.h"

class PlayerFootball{
    private:
        int _num_left_controlled_players = 1;
        int _num_right_controlled_players = 0;
        bool _can_play_right = false;
        std::vector<int> _action = {};
    public:
        PlayerFootball(){};
        PlayerFootball(std::vector<int> player_config, Config_football env_config){
            if(player_config.size() >= 2){
                _num_left_controlled_players = player_config[0];
                _num_right_controlled_players = player_config[1];
            }
        }
        int num_controlled_left_players(){
            return _num_left_controlled_players;
        }
        int num_controlled_right_players(){
            return _num_right_controlled_players;
        }
        int num_controlled_players(){
            return (_num_left_controlled_players + _num_right_controlled_players);
        }
        bool can_play_right(){
            return _can_play_right;
        }   
        void set_action(std::vector<int>action){
            _action = action;
        }  
        std::vector<int>take_action(){
            return _action;
        }
};

#endif 