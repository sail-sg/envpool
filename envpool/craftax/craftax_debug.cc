// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <vector>

#include "envpool/craftax/renderer.h"
#include "envpool/craftax/state_io.h"

namespace py = pybind11;
using craftax::Game;
using craftax::Params;

PYBIND11_MODULE(craftax_debug, module) {
  auto params = py::class_<Params>(module, "Params");
  params.def(py::init<bool>(), py::arg("classic") = false);
#define CRAFTAX_PARAM(field) params.def_readwrite(#field, &Params::field)
  CRAFTAX_PARAM(height);
  CRAFTAX_PARAM(width);
  CRAFTAX_PARAM(levels);
  CRAFTAX_PARAM(symbolic);
  CRAFTAX_PARAM(fractal_noise_angles);
  CRAFTAX_PARAM(max_timesteps);
  CRAFTAX_PARAM(day_length);
  CRAFTAX_PARAM(always_diamond);
  CRAFTAX_PARAM(god_mode);
  CRAFTAX_PARAM(mob_despawn_distance);
  CRAFTAX_PARAM(max_attribute);
  CRAFTAX_PARAM(max_melee_mobs);
  CRAFTAX_PARAM(max_passive_mobs);
  CRAFTAX_PARAM(max_ranged_mobs);
  CRAFTAX_PARAM(max_mob_projectiles);
  CRAFTAX_PARAM(max_player_projectiles);
  CRAFTAX_PARAM(max_growing_plants);
  CRAFTAX_PARAM(zombie_health);
  CRAFTAX_PARAM(cow_health);
  CRAFTAX_PARAM(skeleton_health);
  CRAFTAX_PARAM(spawn_cow_chance);
  CRAFTAX_PARAM(spawn_zombie_base_chance);
  CRAFTAX_PARAM(spawn_zombie_night_chance);
  CRAFTAX_PARAM(spawn_skeleton_chance);
#undef CRAFTAX_PARAM
  py::class_<Game>(module, "Game")
      .def(py::init<Params>())
      .def("reset", &Game::Reset)
      .def("step", &Game::Step)
      .def("done", &Game::Done)
      .def("obs",
           [](const Game& game) {
             auto values = game.Symbolic();
             py::array_t<float> out(values.size());
             std::copy(values.begin(), values.end(), out.mutable_data());
             return out;
           })
      .def("encode_state",
           [](Game& game) {
             const auto values = craftax::EncodeState(&game);
             py::array_t<double> out(values.size());
             std::copy(values.begin(), values.end(), out.mutable_data());
             return out;
           })
      .def("pixels",
           [](const Game& game, int tile) {
             const int rows = game.params.classic ? 9 : 13;
             const int cols = game.params.classic ? 9 : 11;
             py::array_t<float> array({rows * tile, cols * tile, 3});
             auto pixels = craftax::Pixels(game, tile);
             std::copy(pixels.begin(), pixels.end(), array.mutable_data());
             return array;
           })
      .def("set_state",
           [](Game& game, const py::dict& state) {
             craftax::VisitState(&game, [&](const std::string& key,
                                            std::vector<double>* values) {
               auto array = py::array_t<double, py::array::c_style |
                                                    py::array::forcecast>::
                   ensure(state[py::str(key)]);
               if (!array ||
                   static_cast<std::size_t>(array.size()) != values->size()) {
                 throw std::invalid_argument("invalid state field: " + key);
               }
               std::copy_n(array.data(), values->size(), values->begin());
             });
           })
      .def("get_state", [](Game& game) {
        py::dict out;
        craftax::VisitState(
            &game, [&](const std::string& key, std::vector<double>* values) {
              py::array_t<double> array(values->size());
              std::copy(values->begin(), values->end(), array.mutable_data());
              out[py::str(key)] = array;
            });
        return out;
      });
  module.def("split", &craftax::Split);
  module.def("bits", &craftax::Bits);
  module.def("uniform", &craftax::Uniform);
  module.def("randint", &craftax::RandInt);
  module.def("choice", &craftax::Choice);
}
