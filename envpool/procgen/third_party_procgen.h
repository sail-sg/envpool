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

#ifndef ENVPOOL_PROCGEN_THIRD_PARTY_H_
#define ENVPOOL_PROCGEN_THIRD_PARTY_H_
#include "assetgen.h"
#include "basic-abstract-game.h"
#include "buffer.h"
#include "cpp-utils.h"
#include "entity.h"
#include "game-registry.h"
#include "game.h"
#include "grid.h"
#include "libenv.h"
#include "mazegen.h"
#include "object-ids.h"
#include "qt-utils.h"
#include "randgen.h"
#include "resources.h"
#include "roomgen.h"
#include "vecgame.h"
#include "vecoptions.h"
#include "procgen_games.h"

/**
 * @brief helper function that acts as a Procgen game factory
 * @param name the name of the Procgen game we want to retrieve
 * @return a shared pointer to the game
 */
std::shared_ptr<Game> make_game(std::string name) {
    if (name == "bigfish") {
        return make_bigfish();
    } else if (name == "bossfight") {
        return make_bossfight();
    } else if (name == "caveflyer") {
        return make_caveflyer();
    } else if (name == "chaser") {
        return make_chaser();
    } else if (name == "climber") {
        return make_climber();
    } else if (name == "coinrun") {
        return make_coinrun();
    } else if (name == "dodgeball") {
        return make_dodge();
    } else if (name == "fruitbot") {
        return make_fruitbot();
    } else if (name == "heist") {
        return make_heist();
    } else if (name == "jumper") {
        return make_jumper();
    } else if (name == "leaper") {
        return make_leaper();
    } else if (name == "maze") {
        return make_maze();
    } else if (name == "miner") {
        return make_miner();
    } else if (name == "ninja") {
        return make_ninja();
    } else if (name == "plunder") {
        return make_plunder();
    } else if (name == "starpilot") {
        return make_starpilot();
    } else {
        // default fallback to bigfish
        return make_bigfish();
    }
}
#endif