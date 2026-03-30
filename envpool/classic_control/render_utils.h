// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_CLASSIC_CONTROL_RENDER_UTILS_H_
#define ENVPOOL_CLASSIC_CONTROL_RENDER_UTILS_H_

#include <cstdint>

namespace classic_control::rendering {

void RenderCartPole(double x, double theta, int width, int height,
                    std::uint8_t* rgb);

void RenderPendulum(double theta, bool has_last_u, double last_u, int width,
                    int height, std::uint8_t* rgb);

void RenderMountainCar(double pos, double goal_pos, int width, int height,
                       std::uint8_t* rgb);

void RenderAcrobot(double theta1, double theta2, int width, int height,
                   std::uint8_t* rgb);

}  // namespace classic_control::rendering

#endif  // ENVPOOL_CLASSIC_CONTROL_RENDER_UTILS_H_
