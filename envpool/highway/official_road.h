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

#ifndef ENVPOOL_HIGHWAY_OFFICIAL_ROAD_H_
#define ENVPOOL_HIGHWAY_OFFICIAL_ROAD_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "envpool/highway/official_lane.h"

namespace highway::official {

[[maybe_unused]] constexpr int kUnknownLaneId = -1;

struct LaneIndex {
  std::string from;
  std::string to;
  int id{0};

  [[nodiscard]] bool SameRoad(const LaneIndex& other) const {
    return from == other.from && to == other.to;
  }
};

using Route = std::vector<LaneIndex>;

struct PositionHeading {
  Vec2 position;
  double heading{0.0};
};

class RoadNetwork {
 public:
  void AddLane(const std::string& from, const std::string& to,
               const Lane& lane);

  [[nodiscard]] const Lane& GetLane(const LaneIndex& index) const;
  [[nodiscard]] LaneIndex GetClosestLaneIndex(
      Vec2 position, std::optional<double> heading = std::nullopt) const;
  [[nodiscard]] LaneIndex NextLane(LaneIndex current_index, Route* route,
                                   Vec2 position) const;
  [[nodiscard]] std::vector<LaneIndex> AllSideLanes(
      const LaneIndex& index) const;
  [[nodiscard]] std::vector<LaneIndex> SideLanes(const LaneIndex& index) const;
  [[nodiscard]] std::vector<LaneIndex> LaneIndexes() const;
  [[nodiscard]] std::vector<const Lane*> Lanes() const;
  [[nodiscard]] bool IsConnectedRoad(const LaneIndex& start,
                                     const LaneIndex& target, Route route = {},
                                     bool same_lane = false,
                                     int depth = 0) const;
  [[nodiscard]] std::vector<std::string> ShortestPath(
      const std::string& start, const std::string& goal) const;
  [[nodiscard]] PositionHeading PositionHeadingAlongRoute(
      Route route, double longitudinal, double lateral,
      const LaneIndex& current_lane_index) const;

  static RoadNetwork StraightRoadNetwork(
      int lanes = 4, double start = 0.0, double length = 10000.0,
      double angle = 0.0, double speed_limit = 30.0,
      std::pair<std::string, std::string> nodes = {"0", "1"});

 private:
  struct Edge {
    std::string from;
    std::string to;
    std::vector<Lane> lanes;
  };

  std::vector<Edge> edges_;

  [[nodiscard]] const Edge& GetEdge(const std::string& from,
                                    const std::string& to) const;
  [[nodiscard]] const Edge* FindEdge(const std::string& from,
                                     const std::string& to) const;
  [[nodiscard]] Edge* FindEdge(const std::string& from, const std::string& to);
  [[nodiscard]] std::vector<std::string> OutgoingNodes(
      const std::string& from) const;
  [[nodiscard]] std::pair<int, double> NextLaneGivenNextRoad(
      const LaneIndex& current_index, const std::string& next_to, int next_id,
      Vec2 projected_position) const;
};

[[nodiscard]] bool SameRoad(const LaneIndex& lhs, const LaneIndex& rhs,
                            bool same_lane = false);
[[nodiscard]] bool LeadingToRoad(const LaneIndex& lhs, const LaneIndex& rhs,
                                 bool same_lane = false);

}  // namespace highway::official

#endif  // ENVPOOL_HIGHWAY_OFFICIAL_ROAD_H_
