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

#ifndef ENVPOOL_HIGHWAY_OFFICIAL_TASK_H_
#define ENVPOOL_HIGHWAY_OFFICIAL_TASK_H_

#include <string>

#include "envpool/highway/official_scene.h"

namespace highway {
namespace official {

[[nodiscard]] Road MakeMergeRoad();
[[nodiscard]] int ResetMergeVehicles(Road* road, double position_noise0,
                                     double position_noise1,
                                     double position_noise2,
                                     double speed_noise0, double speed_noise1,
                                     double speed_noise2);
[[nodiscard]] Road MakeRoundaboutRoad();
[[nodiscard]] int ResetRoundaboutVehicles(Road* road);
[[nodiscard]] Road MakeTwoWayRoad();
[[nodiscard]] int ResetTwoWayVehicles(Road* road);
[[nodiscard]] Road MakeUTurnRoad();
[[nodiscard]] int ResetUTurnVehicles(Road* road);
[[nodiscard]] Road MakeParkingRoad();
[[nodiscard]] int ResetParkingVehicles(Road* road, double ego_x,
                                       double ego_heading, int goal_spot,
                                       bool add_parked_vehicles);
[[nodiscard]] Road MakeExitRoad();
[[nodiscard]] int ResetExitVehicles(Road* road);
[[nodiscard]] Road MakeIntersectionRoad();
[[nodiscard]] int ResetIntersectionVehicles(Road* road);
[[nodiscard]] int ResetMultiAgentIntersectionVehicles(Road* road);
[[nodiscard]] Road MakeLaneKeepingRoad();
[[nodiscard]] int ResetLaneKeepingVehicle(Road* road);
[[nodiscard]] Road MakeRacetrackRoad(const std::string& scenario);
[[nodiscard]] int ResetRacetrackVehicles(Road* road, double longitudinal,
                                         int lane);

}  // namespace official
}  // namespace highway

#endif  // ENVPOOL_HIGHWAY_OFFICIAL_TASK_H_
