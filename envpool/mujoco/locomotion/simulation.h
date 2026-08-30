/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MUJOCO_LOCOMOTION_SIMULATION_H_
#define ENVPOOL_MUJOCO_LOCOMOTION_SIMULATION_H_

#include <mujoco.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "envpool/mujoco/locomotion/mocap.h"
#include "envpool/mujoco/locomotion/scene.h"
#include "envpool/mujoco/offscreen_renderer.h"

namespace mujoco_locomotion {

struct Observation {
  std::string name;
  std::vector<int> shape;
  // 0 = float64, 1 = int64, 2 = uint8. Boolean observations use int64 storage
  // and are exposed as bool by the API's dtype-only view conversion.
  int storage{0};
  bool boolean{false};
  int offset{0};
  int size{1};
};

std::vector<Observation> ObservationLayout(const std::string& name,
                                           int team_size);
int StorageSize(const std::vector<Observation>& layout, int storage);
int ActionSize(Walker walker);

struct Options {
  std::string task;
  std::string asset_path;
  std::string labmaze_asset_path;
  std::string mocap_asset_path;
  int seed{0};
  int team_size{2};
  int max_episode_steps{1000};
  double time_limit{30};
  bool disable_walker_contacts{false};
  bool enable_field_box{false};
  bool keep_aspect_ratio{false};
  bool terminate_on_goal{true};
};

class Simulation {
 public:
  explicit Simulation(Options options);
  ~Simulation();
  void Reset();
  void Step(const double* actions);
  void Observe();
  void Render(int width, int height, int camera, unsigned char* output,
              const mjvOption* option = nullptr);
  const mjModel* Model() const { return model_.get(); }
  const mjData* Data() const { return data_.get(); }
  const Scene& GetScene() const { return scene_; }
  bool Done() const { return done_; }
  bool Truncated() const { return truncated_; }
  bool Terminated() const {
    return failure_ || (success_ && task_.task != Task::kTracking);
  }
  int Players() const { return players_; }
  const std::vector<Observation>& Layout() const { return layout_; }
#ifdef ENVPOOL_TEST
  void SetResetState(const std::vector<double>& qpos,
                     const std::vector<double>& qvel,
                     const std::map<std::string, std::array<double, 3>>& geoms);
#endif

  std::vector<double> continuous;
  std::vector<int64_t> discrete;
  std::vector<uint8_t> pixels;
  std::vector<double> rewards;
  double discount{1};

 private:
  using Features = std::map<std::string, std::vector<double>>;

  struct WalkerIds {
    std::string prefix;
    int root, frame, head, pelvis, camera, freejoint;
    std::vector<int> effectors, joints, actuators, tendons, bodies;
    std::map<int, std::vector<int>> sensors;
    std::vector<int> effector_sensors;
    std::vector<int> ground_contact_geoms;
  };

  void Compile();
  void CacheIds();
  void ResetWalker(int index);
  void ShiftWalker(int index, const std::array<double, 3>& position,
                   double rotation = 0);
  bool DisallowedContact() const;
  void AfterStep();
  void AfterSubstep();
  void RespawnMaze();
  void ResetBowl();
  void GenerateBowl();
  void RandomizeTouchTarget();
  void TouchReward();
  void InitializeSoccer();
  void BeforeSoccerStep();
  void SoccerDetections();
  void AfterSoccerStep();
  std::vector<double> SoccerObservation(int player,
                                        const std::string& key) const;
  std::vector<double> SensorValues(const std::string& name) const;
  void SelectTrackingClip();
  void ResetTracking();
  void AfterTrackingStep();
  void UpdateTrackingObservations();
  Features TrackingFeatures() const;
  double TrackingError() const;
  std::vector<double> WalkerObservation(const WalkerIds& walker,
                                        const std::string& key) const;
  std::vector<double> RelativePositions(const WalkerIds& walker,
                                        const std::vector<int>& bodies) const;
  int Id(mjtObj object, const std::string& name) const;

  Options options_;
  TaskConfig task_;
  RandomState random_, scene_random_;
  Scene scene_;
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model_{nullptr,
                                                             mj_deleteModel};
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data_{nullptr,
                                                          mj_deleteData};
  std::unique_ptr<envpool::mujoco::OffscreenRenderer> renderer_;
  std::vector<Observation> layout_;
  std::vector<WalkerIds> walkers_;
  std::vector<int> ground_geoms_;
  std::vector<double> previous_actions_;
  std::vector<int> target_geoms_, target_materials_;
  std::vector<bool> target_activated_, target_rewarded_;
  std::vector<int> hand_geoms_;
  int touch_state_{0}, observed_touch_state_{0};
  double touch_time_{0}, first_touch_time_{0}, second_touch_time_{0};
  bool touched_once_{false}, touched_twice_{false};
  bool touch_timeout_{false}, randomize_touch_{false}, model_dirty_{false};
  int ball_joint_{-1}, ball_geom_{-1};
  std::vector<int> geom_player_;
  std::array<bool, 2> goals_{}, goals_now_{};
  bool off_court_{false};
  int scoring_team_{-1};
  std::shared_ptr<const MocapData> mocap_;
  int clip_id_{0}, reference_step_{0}, reference_start_{0};
  std::vector<std::pair<int, int>> possible_starts_;
  std::vector<double> start_cdf_;
  Features tracking_features_, previous_features_, tracking_observations_;
  int players_, steps_{0};
  bool done_{true}, truncated_{false}, failure_{false}, success_{false};
};

}  // namespace mujoco_locomotion

#endif  // ENVPOOL_MUJOCO_LOCOMOTION_SIMULATION_H_
