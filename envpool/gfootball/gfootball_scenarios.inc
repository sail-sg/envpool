inline void BuildScenarioConfig(const std::string& env_name, int episode_number, unsigned int game_engine_random_seed, ScenarioConfig* cfg) {
  cfg->left_agents = 1;
  cfg->right_agents = 0;
  cfg->use_magnet = true;
  cfg->real_time = false;
  cfg->reverse_team_processing = false;
  cfg->second_half = 999999999;
  cfg->game_engine_random_seed = game_engine_random_seed;
  const bool even_episode = (episode_number % 2) == 0;
  if (env_name == "11_vs_11_competition") {
    if (even_episode) {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 1.0f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    } else {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 1.0f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    }
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "11_vs_11_easy_stochastic") {
    if (even_episode) {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 0.05f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    } else {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 0.05f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    }
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "11_vs_11_hard_stochastic") {
    if (even_episode) {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 0.95f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    } else {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 0.95f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    }
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "11_vs_11_kaggle") {
    if (even_episode) {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 1.0f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->second_half = 1500;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->left_team.emplace_back(0.0f, 0.02f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(0.0f, -0.02f, e_PlayerRole_RM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_RM, false, true);
    } else {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 1.0f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->second_half = 1500;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->left_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_RM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.emplace_back(0.0f, 0.02f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(0.0f, -0.02f, e_PlayerRole_RM, false, true);
    }
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "11_vs_11_stochastic") {
    if (even_episode) {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 0.6f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    } else {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 1.0f;
    cfg->right_team_difficulty = 0.6f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->left_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    cfg->right_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
    cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
    cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
    cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
    }
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "1_vs_1_easy") {
  cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
  cfg->offsides = true;
  cfg->left_team_difficulty = 0.0f;
  cfg->right_team_difficulty = 0.0f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = false;
  cfg->end_episode_on_possession_change = false;
  cfg->end_episode_on_out_of_play = false;
  cfg->game_duration = 500;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "5_vs_5") {
    if (even_episode) {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 0.05f;
    cfg->right_team_difficulty = 0.05f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, false);
    cfg->left_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.1f, -0.1f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.1f, 0.1f, e_PlayerRole_CB, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, false);
    cfg->right_team.emplace_back(-0.04f, 0.04f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(-0.04f, -0.04f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.1f, -0.1f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.1f, 0.1f, e_PlayerRole_CB, false, true);
    } else {
    cfg->ball_position = Vector3(0.0f, 0.0f, 0.0f);
    cfg->offsides = true;
    cfg->left_team_difficulty = 0.05f;
    cfg->right_team_difficulty = 0.05f;
    cfg->deterministic = false;
    cfg->end_episode_on_score = false;
    cfg->end_episode_on_possession_change = false;
    cfg->end_episode_on_out_of_play = false;
    cfg->game_duration = 3000;
    cfg->control_all_players = false;
    cfg->left_team.clear();
    cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, false);
    cfg->left_team.emplace_back(-0.04f, 0.04f, e_PlayerRole_RM, false, true);
    cfg->left_team.emplace_back(-0.04f, -0.04f, e_PlayerRole_CF, false, true);
    cfg->left_team.emplace_back(-0.1f, -0.1f, e_PlayerRole_LB, false, true);
    cfg->left_team.emplace_back(-0.1f, 0.1f, e_PlayerRole_CB, false, true);
    cfg->right_team.clear();
    cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, false);
    cfg->right_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
    cfg->right_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
    cfg->right_team.emplace_back(-0.1f, -0.1f, e_PlayerRole_LB, false, true);
    cfg->right_team.emplace_back(-0.1f, 0.1f, e_PlayerRole_CB, false, true);
    }
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_3_vs_1_with_keeper") {
  cfg->ball_position = Vector3(0.62f, 0.0f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(0.6f, 0.0f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.7f, 0.2f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.7f, -0.2f, e_PlayerRole_CM, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(-0.75f, 0.0f, e_PlayerRole_CB, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_corner") {
  cfg->ball_position = Vector3(0.99f, 0.41f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = false;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(1.0f, 0.42f, e_PlayerRole_LB, false, true);
  cfg->left_team.emplace_back(0.7f, 0.15f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(0.7f, 0.05f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(0.7f, -0.05f, e_PlayerRole_RB, false, true);
  cfg->left_team.emplace_back(0.0f, 0.0f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.6f, 0.35f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.8f, 0.07f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.8f, -0.03f, e_PlayerRole_LM, false, true);
  cfg->left_team.emplace_back(0.8f, -0.13f, e_PlayerRole_RM, false, true);
  cfg->left_team.emplace_back(0.7f, -0.3f, e_PlayerRole_CF, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(-0.75f, -0.18f, e_PlayerRole_LB, false, true);
  cfg->right_team.emplace_back(-0.75f, -0.08f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(-0.75f, 0.02f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(-1.0f, -0.1f, e_PlayerRole_RB, false, true);
  cfg->right_team.emplace_back(-0.8f, -0.25f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(-0.88f, -0.07f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(-0.88f, 0.03f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(-0.88f, 0.13f, e_PlayerRole_LM, false, true);
  cfg->right_team.emplace_back(-0.75f, 0.25f, e_PlayerRole_RM, false, true);
  cfg->right_team.emplace_back(-0.2f, 0.0f, e_PlayerRole_CF, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_counterattack_easy") {
  cfg->ball_position = Vector3(0.26f, -0.11f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(-0.672f, -0.19576f, e_PlayerRole_LB, false, true);
  cfg->left_team.emplace_back(-0.75f, -0.06356f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(-0.75f, 0.063559f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(-0.672f, 0.19576f, e_PlayerRole_RB, false, true);
  cfg->left_team.emplace_back(-0.434f, -0.10568f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(-0.434f, 0.10568f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.5f, -0.3161f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.25f, -0.1f, e_PlayerRole_LM, false, true);
  cfg->left_team.emplace_back(0.25f, 0.1f, e_PlayerRole_RM, false, true);
  cfg->left_team.emplace_back(0.35f, 0.316102f, e_PlayerRole_CF, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(0.128f, -0.19576f, e_PlayerRole_LB, false, true);
  cfg->right_team.emplace_back(0.4f, -0.06356f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(-0.4f, 0.063559f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(0.128f, -0.19576f, e_PlayerRole_RB, false, true);
  cfg->right_team.emplace_back(0.365f, -0.10568f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(0.282f, 0.0f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(0.365f, 0.10568f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(0.54f, -0.3161f, e_PlayerRole_LM, false, true);
  cfg->right_team.emplace_back(0.51f, 0.0f, e_PlayerRole_RM, false, true);
  cfg->right_team.emplace_back(0.54f, 0.316102f, e_PlayerRole_CF, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_counterattack_hard") {
  cfg->ball_position = Vector3(0.26f, -0.11f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(-0.672f, -0.19576f, e_PlayerRole_LB, false, true);
  cfg->left_team.emplace_back(-0.75f, -0.06356f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(-0.75f, 0.063559f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(-0.672f, 0.19576f, e_PlayerRole_RB, false, true);
  cfg->left_team.emplace_back(-0.434f, -0.10568f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(-0.434f, 0.10568f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.5f, -0.3161f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(0.25f, -0.1f, e_PlayerRole_LM, false, true);
  cfg->left_team.emplace_back(0.25f, 0.1f, e_PlayerRole_RM, false, true);
  cfg->left_team.emplace_back(0.35f, 0.316102f, e_PlayerRole_CF, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(0.128f, -0.19576f, e_PlayerRole_LB, false, true);
  cfg->right_team.emplace_back(-0.4f, -0.06356f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(-0.4f, 0.063559f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(0.128f, -0.19576f, e_PlayerRole_RB, false, true);
  cfg->right_team.emplace_back(0.365f, -0.10568f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(0.282f, 0.0f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(0.365f, 0.10568f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(0.54f, -0.3161f, e_PlayerRole_LM, false, true);
  cfg->right_team.emplace_back(0.51f, 0.0f, e_PlayerRole_RM, false, true);
  cfg->right_team.emplace_back(0.54f, 0.316102f, e_PlayerRole_CF, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_empty_goal") {
  cfg->ball_position = Vector3(0.02f, 0.0f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(0.0f, 0.0f, e_PlayerRole_CB, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(1.0f, 0.0f, e_PlayerRole_GK, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_empty_goal_close") {
  cfg->ball_position = Vector3(0.77f, 0.0f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(0.75f, 0.0f, e_PlayerRole_CB, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(1.0f, 0.0f, e_PlayerRole_GK, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_pass_and_shoot_with_keeper") {
  cfg->ball_position = Vector3(0.7f, -0.28f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(0.7f, 0.0f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(0.7f, -0.3f, e_PlayerRole_CB, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(-0.75f, 0.3f, e_PlayerRole_CB, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_run_pass_and_shoot_with_keeper") {
  cfg->ball_position = Vector3(0.7f, -0.28f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(0.7f, 0.0f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(0.7f, -0.3f, e_PlayerRole_CB, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(-0.75f, 0.1f, e_PlayerRole_CB, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_run_to_score") {
  cfg->ball_position = Vector3(0.02f, 0.0f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(0.0f, 0.0f, e_PlayerRole_CB, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(0.12f, 0.2f, e_PlayerRole_LB, false, true);
  cfg->right_team.emplace_back(0.12f, 0.1f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(0.12f, 0.0f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(0.12f, -0.1f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(0.12f, -0.2f, e_PlayerRole_RB, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_run_to_score_with_keeper") {
  cfg->ball_position = Vector3(0.02f, 0.0f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 400;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(0.0f, 0.0f, e_PlayerRole_CB, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(0.12f, 0.2f, e_PlayerRole_LB, false, true);
  cfg->right_team.emplace_back(0.12f, 0.1f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(0.12f, 0.0f, e_PlayerRole_CM, false, true);
  cfg->right_team.emplace_back(0.12f, -0.1f, e_PlayerRole_CB, false, true);
  cfg->right_team.emplace_back(0.12f, -0.2f, e_PlayerRole_RB, false, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  if (env_name == "academy_single_goal_versus_lazy") {
  cfg->ball_position = Vector3(-0.48f, -0.06356f, 0.0f);
  cfg->offsides = false;
  cfg->left_team_difficulty = 1.0f;
  cfg->right_team_difficulty = 0.6f;
  cfg->deterministic = false;
  cfg->end_episode_on_score = true;
  cfg->end_episode_on_possession_change = true;
  cfg->end_episode_on_out_of_play = true;
  cfg->game_duration = 3000;
  cfg->control_all_players = false;
  cfg->left_team.clear();
  cfg->left_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->left_team.emplace_back(0.0f, 0.02f, e_PlayerRole_RM, false, true);
  cfg->left_team.emplace_back(0.0f, -0.02f, e_PlayerRole_CF, false, true);
  cfg->left_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, false, true);
  cfg->left_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, false, true);
  cfg->left_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, false, true);
  cfg->left_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, false, true);
  cfg->left_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, false, true);
  cfg->right_team.clear();
  cfg->right_team.emplace_back(-1.0f, 0.0f, e_PlayerRole_GK, false, true);
  cfg->right_team.emplace_back(-0.05f, 0.0f, e_PlayerRole_RM, true, true);
  cfg->right_team.emplace_back(-0.01f, 0.216102f, e_PlayerRole_CF, true, true);
  cfg->right_team.emplace_back(-0.422f, -0.19576f, e_PlayerRole_LB, true, true);
  cfg->right_team.emplace_back(-0.5f, -0.06356f, e_PlayerRole_CB, true, true);
  cfg->right_team.emplace_back(-0.5f, 0.063559f, e_PlayerRole_CB, true, true);
  cfg->right_team.emplace_back(-0.422f, 0.19576f, e_PlayerRole_RB, true, true);
  cfg->right_team.emplace_back(-0.184212f, -0.10568f, e_PlayerRole_CM, true, true);
  cfg->right_team.emplace_back(-0.267574f, 0.0f, e_PlayerRole_CM, true, true);
  cfg->right_team.emplace_back(-0.184212f, 0.10568f, e_PlayerRole_CM, true, true);
  cfg->right_team.emplace_back(-0.01f, -0.2161f, e_PlayerRole_LM, true, true);
    if (!cfg->deterministic) {
      cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
    }
    return;
  }
  throw std::invalid_argument("Unknown gfootball scenario: " + env_name);
}
