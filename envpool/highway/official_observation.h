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

#ifndef ENVPOOL_HIGHWAY_OFFICIAL_OBSERVATION_H_
#define ENVPOOL_HIGHWAY_OFFICIAL_OBSERVATION_H_

#include <optional>
#include <vector>

#include "envpool/highway/official_scene.h"

namespace highway {
namespace official {

enum class KinematicFeature : int {
  kPresence = 0,
  kX = 1,
  kY = 2,
  kVx = 3,
  kVy = 4,
  kHeading = 5,
  kCosH = 6,
  kSinH = 7,
  kCosD = 8,
  kSinD = 9,
  kLongOff = 10,
  kLatOff = 11,
  kAngOff = 12,
};

struct KinematicObservationConfig {
  std::vector<KinematicFeature> features{
      KinematicFeature::kPresence, KinematicFeature::kX, KinematicFeature::kY,
      KinematicFeature::kVx, KinematicFeature::kVy};
  int vehicles_count{5};
  bool absolute{false};
  bool normalize{true};
  bool clip{true};
  bool see_behind{false};
  bool include_obstacles{true};
  double x_min{-5.0 * kVehicleMaxSpeed};
  double x_max{5.0 * kVehicleMaxSpeed};
  double y_min{0.0};
  double y_max{0.0};
  double vx_min{-2.0 * kVehicleMaxSpeed};
  double vx_max{2.0 * kVehicleMaxSpeed};
  double vy_min{-2.0 * kVehicleMaxSpeed};
  double vy_max{2.0 * kVehicleMaxSpeed};
  std::optional<double> ego_x_override;
};

struct FeatureRow {
  double presence{1.0};
  double x{0.0};
  double y{0.0};
  double vx{0.0};
  double vy{0.0};
  double heading{0.0};
  double cos_h{1.0};
  double sin_h{0.0};
  double cos_d{0.0};
  double sin_d{0.0};
  double long_off{0.0};
  double lat_off{0.0};
  double ang_off{0.0};
};

[[nodiscard]] FeatureRow VehicleToFeatureRow(
    const Road& road, const Vehicle& vehicle,
    const Vehicle* origin_vehicle = nullptr, bool observe_intentions = true);

[[nodiscard]] std::vector<float> ObserveKinematics(
    const Road& road, const Vehicle& observer,
    const KinematicObservationConfig& config);
[[nodiscard]] std::vector<float> ObserveTimeToCollision(
    const Road& road, const Vehicle& observer, double time_quantization,
    double horizon);

[[nodiscard]] double FeatureValue(const FeatureRow& row,
                                  KinematicFeature feature);

}  // namespace official
}  // namespace highway

#endif  // ENVPOOL_HIGHWAY_OFFICIAL_OBSERVATION_H_
