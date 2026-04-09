/*
 * Copyright 2026 Garena Online Private Limited
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
// Straight-road highway-v0 port:
// https://github.com/Farama-Foundation/HighwayEnv

#ifndef ENVPOOL_HIGHWAY_HIGHWAY_ENV_H_
#define ENVPOOL_HIGHWAY_HIGHWAY_ENV_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace highway {

[[maybe_unused]] constexpr double kPi = 3.14159265358979323846;
[[maybe_unused]] constexpr double kLaneWidth = 4.0;
[[maybe_unused]] constexpr double kLaneLength = 10000.0;
[[maybe_unused]] constexpr double kVehicleLength = 5.0;
[[maybe_unused]] constexpr double kVehicleWidth = 2.0;
[[maybe_unused]] constexpr double kVehicleMaxSpeed = 40.0;
[[maybe_unused]] constexpr double kVehicleMinSpeed = -40.0;
[[maybe_unused]] constexpr double kPerceptionDistance = 5.0 * kVehicleMaxSpeed;

struct HighwayVehicleDebugState {
  int kind{0};
  std::string lane_from{"0"};
  std::string lane_to{"1"};
  int lane_index{0};
  std::string target_lane_from{"0"};
  std::string target_lane_to{"1"};
  int target_lane_index{0};
  int speed_index{0};
  double x{0.0};
  double y{0.0};
  double heading{0.0};
  double speed{0.0};
  double target_speed{0.0};
  double target_speed0{20.0};
  double target_speed1{25.0};
  double target_speed2{30.0};
  bool has_goal{false};
  double goal_x{0.0};
  double goal_y{0.0};
  double goal_heading{0.0};
  double goal_speed{0.0};
  double idm_delta{4.0};
  double timer{0.0};
  bool crashed{false};
  bool on_road{true};
  bool check_collisions{true};
  bool enable_lane_change{true};
  std::vector<std::string> route_from;
  std::vector<std::string> route_to;
  std::vector<int> route_id;
};

struct HighwayLaneDebugState {
  std::string from;
  std::string to;
  int index{0};
  int kind{0};
  double start_x{0.0};
  double start_y{0.0};
  double end_x{0.0};
  double end_y{0.0};
  double center_x{0.0};
  double center_y{0.0};
  double width{kLaneWidth};
  int line_type0{1};
  int line_type1{1};
  bool forbidden{false};
  double speed_limit{20.0};
  int priority{0};
  double amplitude{0.0};
  double pulsation{0.0};
  double phase{0.0};
  double radius{1.0};
  double start_phase{0.0};
  double end_phase{0.0};
  bool clockwise{true};
};

struct HighwayRoadObjectDebugState {
  int kind{0};
  double x{0.0};
  double y{0.0};
  double heading{0.0};
  double speed{0.0};
  double length{2.0};
  double width{2.0};
  bool collidable{true};
  bool solid{true};
  bool check_collisions{true};
  bool crashed{false};
  bool hit{false};
};

struct HighwayDebugState {
  std::string scenario{"highway"};
  int lanes_count{0};
  int simulation_frequency{0};
  int policy_frequency{0};
  int elapsed_step{0};
  double time{0.0};
  std::vector<HighwayLaneDebugState> road_lanes;
  std::vector<HighwayRoadObjectDebugState> road_objects;
  std::vector<HighwayVehicleDebugState> vehicles;
};

class HighwayEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "lanes_count"_.Bind(4), "vehicles_count"_.Bind(50),
        "observation_vehicles_count"_.Bind(5), "controlled_vehicles"_.Bind(1),
        "initial_lane_id"_.Bind(-1), "duration"_.Bind(40),
        "simulation_frequency"_.Bind(15), "policy_frequency"_.Bind(1),
        "ego_spacing"_.Bind(2.0), "vehicles_density"_.Bind(1.0),
        "collision_reward"_.Bind(-1.0), "right_lane_reward"_.Bind(0.1),
        "high_speed_reward"_.Bind(0.4), "lane_change_reward"_.Bind(0.0),
        "reward_speed_low"_.Bind(20.0), "reward_speed_high"_.Bind(30.0),
        "normalize_reward"_.Bind(true), "offroad_terminal"_.Bind(false),
        "other_vehicles_check_collisions"_.Bind(true),
        "screen_width"_.Bind(600), "screen_height"_.Bind(150),
        "centering_position_x"_.Bind(0.3), "centering_position_y"_.Bind(0.5),
        "scaling"_.Bind(5.5));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>(
                        {conf["observation_vehicles_count"_], 5}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 4}))));
  }
};

using HighwayEnvSpec = EnvSpec<HighwayEnvFns>;

enum class VehicleKind : std::uint8_t { kMDP = 0, kIDM = 1 };

struct Vehicle {
  VehicleKind kind{VehicleKind::kIDM};
  double x{0.0};
  double y{0.0};
  double heading{0.0};
  double speed{0.0};
  double steering{0.0};
  double acceleration{0.0};
  double target_speed{0.0};
  int lane_index{0};
  int target_lane_index{0};
  int speed_index{1};
  std::array<double, 3> target_speeds{20.0, 25.0, 30.0};
  bool crashed{false};
  bool check_collisions{true};
  bool enable_lane_change{true};
  double timer{0.0};
  double idm_delta{4.0};

  [[nodiscard]] double Vx() const { return speed * std::cos(heading); }
  [[nodiscard]] double Vy() const { return speed * std::sin(heading); }
};

class HighwayEnv : public Env<HighwayEnvSpec>, public RenderableEnv {
 public:
  HighwayEnv(const Spec& spec, int env_id);

  bool IsDone() override;
  void Reset() override;
  void Step(const Action& action) override;
  [[nodiscard]] std::pair<int, int> RenderSize(int width,
                                               int height) const override;
  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override;

  [[nodiscard]] HighwayDebugState DebugState() const;

 protected:
  [[nodiscard]] int CurrentMaxEpisodeSteps() const override;

 private:
  int lanes_count_;
  int traffic_vehicle_count_;
  int obs_vehicle_count_;
  int initial_lane_id_;
  int max_episode_steps_;
  int simulation_frequency_;
  int policy_frequency_;
  int elapsed_step_{0};
  double time_{0.0};
  double ego_spacing_;
  double vehicles_density_;
  double collision_reward_;
  double right_lane_reward_;
  double high_speed_reward_;
  double reward_speed_low_;
  double reward_speed_high_;
  bool normalize_reward_;
  bool offroad_terminal_;
  bool other_vehicles_check_collisions_;
  bool done_{true};

  std::vector<Vehicle> vehicles_;

  std::uniform_real_distribution<double> uniform01_;

  [[nodiscard]] double Uniform(double low, double high);
  [[nodiscard]] int RandomLane();
  void ResetRoad();
  Vehicle CreateRandomVehicle(VehicleKind kind, std::optional<double> speed,
                              int lane_id, double spacing);
  void CreateVehicles();

  [[nodiscard]] int ClosestLane(double y) const;
  [[nodiscard]] double LaneCenter(int lane_index) const;
  [[nodiscard]] std::pair<double, double> LaneCoordinates(
      const Vehicle& vehicle, int lane_index) const;
  [[nodiscard]] bool LaneIsReachableFrom(const Vehicle& vehicle,
                                         int lane_index) const;
  [[nodiscard]] bool LaneOnRoad(const Vehicle& vehicle) const;
  void UpdateVehicleLane(Vehicle* vehicle);

  void ApplyMetaAction(int action);
  void RoadAct();
  void ActMDP(Vehicle* vehicle, const std::optional<std::string>& action);
  void ActControlled(Vehicle* vehicle,
                     const std::optional<std::string>& action);
  void ActIDM(int vehicle_index);
  void StepVehicle(Vehicle* vehicle, double dt);
  void RoadStep(double dt);
  void CheckCollisions();

  [[nodiscard]] double SteeringControl(const Vehicle& vehicle,
                                       int target_lane_index) const;
  [[nodiscard]] double SpeedControl(const Vehicle& vehicle,
                                    double target_speed) const;
  [[nodiscard]] int SpeedToIndex(const Vehicle& vehicle, double speed) const;
  [[nodiscard]] double IndexToSpeed(const Vehicle& vehicle, int index) const;

  [[nodiscard]] std::pair<int, int> NeighbourVehicles(const Vehicle& vehicle,
                                                      int lane_index) const;
  [[nodiscard]] double LaneDistanceTo(const Vehicle& self,
                                      const Vehicle& other) const;
  [[nodiscard]] double IDMDistanceTo(const Vehicle& ego,
                                     const Vehicle& front) const;
  [[nodiscard]] double IDMAcceleration(int idm_index, const Vehicle* ego,
                                       const Vehicle* front) const;
  [[nodiscard]] double DesiredGap(const Vehicle& ego,
                                  const Vehicle& front) const;
  void ChangeLanePolicy(int vehicle_index);
  [[nodiscard]] bool Mobil(int vehicle_index, int lane_index) const;

  [[nodiscard]] bool RectanglesIntersect(const Vehicle& a,
                                         const Vehicle& b) const;
  [[nodiscard]] double Reward(int action) const;
  [[nodiscard]] bool EgoOnRoad() const;
  void WriteState(float reward);
};

class HighwayEnvPool : public AsyncEnvPool<HighwayEnv> {
 public:
  using AsyncEnvPool<HighwayEnv>::AsyncEnvPool;

  [[nodiscard]] std::vector<HighwayDebugState> DebugStates(
      const std::vector<int>& env_ids) const;
};

}  // namespace highway

#endif  // ENVPOOL_HIGHWAY_HIGHWAY_ENV_H_
