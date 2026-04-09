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

#ifndef ENVPOOL_HIGHWAY_OFFICIAL_SCENE_H_
#define ENVPOOL_HIGHWAY_OFFICIAL_SCENE_H_

#include <optional>
#include <utility>
#include <vector>

#include "envpool/highway/official_lane.h"
#include "envpool/highway/official_road.h"
#include "envpool/highway/official_vehicle.h"

namespace highway {
namespace official {

enum class RoadObjectKind : int {
  kObstacle = 0,
  kLandmark = 1,
};

struct RoadObject {
  RoadObjectKind kind{RoadObjectKind::kObstacle};
  Vec2 position{};
  double heading{0.0};
  double speed{0.0};
  double length{2.0};
  double width{2.0};
  bool collidable{true};
  bool solid{true};
  bool check_collisions{true};
  bool crashed{false};
  bool hit{false};
  LaneIndex lane_index{};
};

class Road {
 public:
  RoadNetwork network;
  std::vector<Vehicle> vehicles;
  std::vector<RoadObject> objects;
  bool record_history{false};

  [[nodiscard]] std::pair<int, int> NeighbourVehicles(
      const Vehicle& vehicle,
      std::optional<LaneIndex> lane_index = std::nullopt) const;
  void Act();
  void Step(double dt);

 private:
  void CheckCollisions(double dt);
};

Vehicle MakeIDMVehicle(
    const RoadNetwork& network, Vec2 position, double heading = 0.0,
    double speed = 0.0,
    std::optional<LaneIndex> target_lane_index = std::nullopt,
    std::optional<double> target_speed = std::nullopt, Route route = {},
    bool enable_lane_change = true, std::optional<double> timer = std::nullopt);
Vehicle MakeIDMVehicleFrom(const Vehicle& vehicle);

void ActIDM(Road* road, int vehicle_index);
void ChangeLanePolicy(Road* road, int vehicle_index);

[[nodiscard]] double IDMAcceleration(const Road& road, const Vehicle* ego,
                                     const Vehicle* front);
[[nodiscard]] double DesiredGap(const Vehicle& ego, const Vehicle& front);
[[nodiscard]] bool Mobil(const Road& road, int vehicle_index,
                         const LaneIndex& lane_index);
[[nodiscard]] double LaneDistanceTo(const RoadNetwork& network,
                                    const Vehicle& self, const Vehicle& other);

}  // namespace official
}  // namespace highway

#endif  // ENVPOOL_HIGHWAY_OFFICIAL_SCENE_H_
