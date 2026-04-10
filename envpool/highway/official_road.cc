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

#include "envpool/highway/official_road.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace highway::official {

namespace {

Vec2 RotateRoadPoint(Vec2 point, double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {c * point.x + s * point.y, -s * point.x + c * point.y};
}

}  // namespace

bool SameRoad(const LaneIndex& lhs, const LaneIndex& rhs, bool same_lane) {
  return lhs.from == rhs.from && lhs.to == rhs.to &&
         (!same_lane || lhs.id == rhs.id);
}

bool LeadingToRoad(const LaneIndex& lhs, const LaneIndex& rhs, bool same_lane) {
  return lhs.to == rhs.from && (!same_lane || lhs.id == rhs.id);
}

void RoadNetwork::AddLane(const std::string& from, const std::string& to,
                          const Lane& lane) {
  Edge* edge = FindEdge(from, to);
  if (edge == nullptr) {
    edges_.push_back({from, to, {}});
    edge = &edges_.back();
  }
  edge->lanes.push_back(lane);
}

const Lane& RoadNetwork::GetLane(const LaneIndex& index) const {
  const Edge& edge = GetEdge(index.from, index.to);
  int lane_id = index.id;
  if (lane_id == kUnknownLaneId && edge.lanes.size() == 1) {
    lane_id = 0;
  }
  if (lane_id < 0 || static_cast<std::size_t>(lane_id) >= edge.lanes.size()) {
    throw std::out_of_range("invalid highway lane id");
  }
  return edge.lanes[lane_id];
}

LaneIndex RoadNetwork::GetClosestLaneIndex(
    Vec2 position, std::optional<double> heading) const {
  if (edges_.empty()) {
    throw std::runtime_error("cannot find a lane in an empty road network");
  }
  LaneIndex best;
  double best_distance = 0.0;
  bool found = false;
  for (const auto& edge : edges_) {
    for (int i = 0; static_cast<std::size_t>(i) < edge.lanes.size(); ++i) {
      const double distance =
          heading.has_value()
              ? edge.lanes[i].DistanceWithHeading(position, *heading)
              : edge.lanes[i].Distance(position);
      if (!found || distance < best_distance) {
        best = {edge.from, edge.to, i};
        best_distance = distance;
        found = true;
      }
    }
  }
  return best;
}

LaneIndex RoadNetwork::NextLane(LaneIndex current_index, Route* route,
                                Vec2 position) const {
  const auto& current_lane = GetLane(current_index);
  std::string next_to;
  int next_id = kUnknownLaneId;
  if (route != nullptr && !route->empty()) {
    if (SameRoad((*route)[0], current_index)) {
      route->erase(route->begin());
    }
    if (!route->empty() && (*route)[0].from == current_index.to) {
      next_to = (*route)[0].to;
      next_id = (*route)[0].id;
    }
  }
  const LaneCoordinates local = current_lane.LocalCoordinates(position);
  const Vec2 projected = current_lane.Position(local.longitudinal, 0.0);
  if (next_to.empty()) {
    const std::vector<std::string> outgoing = OutgoingNodes(current_index.to);
    if (outgoing.empty()) {
      return current_index;
    }
    bool found = false;
    double best_distance = 0.0;
    std::string best_to;
    int best_id = 0;
    for (const auto& candidate_to : outgoing) {
      const auto [candidate_id, candidate_distance] = NextLaneGivenNextRoad(
          current_index, candidate_to, kUnknownLaneId, projected);
      if (!found || candidate_distance < best_distance) {
        best_to = candidate_to;
        best_id = candidate_id;
        best_distance = candidate_distance;
        found = true;
      }
    }
    return {current_index.to, best_to, best_id};
  }
  const auto [resolved_id, unused_distance] =
      NextLaneGivenNextRoad(current_index, next_to, next_id, projected);
  (void)unused_distance;
  return {current_index.to, next_to, resolved_id};
}

std::vector<LaneIndex> RoadNetwork::AllSideLanes(const LaneIndex& index) const {
  const Edge& edge = GetEdge(index.from, index.to);
  std::vector<LaneIndex> lanes;
  lanes.reserve(edge.lanes.size());
  for (int id = 0; static_cast<std::size_t>(id) < edge.lanes.size(); ++id) {
    lanes.push_back({index.from, index.to, id});
  }
  return lanes;
}

std::vector<LaneIndex> RoadNetwork::SideLanes(const LaneIndex& index) const {
  const Edge& edge = GetEdge(index.from, index.to);
  std::vector<LaneIndex> lanes;
  if (index.id > 0) {
    lanes.push_back({index.from, index.to, index.id - 1});
  }
  const int next_id = index.id + 1;
  if (next_id >= 0 && static_cast<std::size_t>(next_id) < edge.lanes.size()) {
    lanes.push_back({index.from, index.to, next_id});
  }
  return lanes;
}

std::vector<LaneIndex> RoadNetwork::LaneIndexes() const {
  std::vector<LaneIndex> indexes;
  for (const auto& edge : edges_) {
    for (int id = 0; static_cast<std::size_t>(id) < edge.lanes.size(); ++id) {
      indexes.push_back({edge.from, edge.to, id});
    }
  }
  return indexes;
}

bool RoadNetwork::IsConnectedRoad(const LaneIndex& start,
                                  const LaneIndex& target, Route route,
                                  bool same_lane, int depth) const {
  if (SameRoad(target, start, same_lane) ||
      LeadingToRoad(target, start, same_lane)) {
    return true;
  }
  if (depth <= 0) {
    return false;
  }
  if (!route.empty() && SameRoad(route[0], start)) {
    route.erase(route.begin());
    return IsConnectedRoad(start, target, std::move(route), same_lane, depth);
  }
  if (!route.empty() && route[0].from == start.to) {
    const LaneIndex next = route[0];
    route.erase(route.begin());
    return IsConnectedRoad(next, target, std::move(route), same_lane,
                           depth - 1);
  }
  const std::vector<std::string> next_nodes = OutgoingNodes(start.to);
  return std::any_of(
      next_nodes.begin(), next_nodes.end(), [&](const std::string& next_to) {
        return IsConnectedRoad({start.to, next_to, start.id}, target, route,
                               same_lane, depth - 1);
      });
}

std::vector<const Lane*> RoadNetwork::Lanes() const {
  std::vector<const Lane*> lanes;
  for (const auto& edge : edges_) {
    for (const auto& lane : edge.lanes) {
      lanes.push_back(&lane);
    }
  }
  return lanes;
}

std::vector<std::string> RoadNetwork::ShortestPath(
    const std::string& start, const std::string& goal) const {
  std::queue<std::vector<std::string>> queue;
  queue.push({start});
  while (!queue.empty()) {
    auto path = queue.front();
    queue.pop();
    const std::string& node = path.back();
    std::vector<std::string> outgoing = OutgoingNodes(node);
    std::sort(outgoing.begin(), outgoing.end());
    for (const auto& next : outgoing) {
      if (std::find(path.begin(), path.end(), next) != path.end()) {
        continue;
      }
      auto next_path = path;
      next_path.push_back(next);
      if (next == goal) {
        return next_path;
      }
      queue.push(std::move(next_path));
    }
  }
  return {};
}

PositionHeading RoadNetwork::PositionHeadingAlongRoute(
    Route route, double longitudinal, double lateral,
    const LaneIndex& current_lane_index) const {
  if (route.empty()) {
    const Lane& lane = GetLane(current_lane_index);
    return {lane.Position(longitudinal, lateral), lane.HeadingAt(longitudinal)};
  }
  auto route_head = [&](const Route& value) {
    LaneIndex lane_index = value.front();
    if (lane_index.id == kUnknownLaneId) {
      const auto& current_edge =
          GetEdge(current_lane_index.from, current_lane_index.to);
      lane_index.id = static_cast<std::size_t>(current_lane_index.id) <
                              current_edge.lanes.size()
                          ? current_lane_index.id
                          : 0;
    }
    return lane_index;
  };
  LaneIndex lane_index = route_head(route);
  while (route.size() > 1 && longitudinal > GetLane(lane_index).Length()) {
    longitudinal -= GetLane(lane_index).Length();
    route.erase(route.begin());
    lane_index = route_head(route);
  }
  const Lane& lane = GetLane(lane_index);
  return {lane.Position(longitudinal, lateral), lane.HeadingAt(longitudinal)};
}

RoadNetwork RoadNetwork::StraightRoadNetwork(
    int lanes, double start, double length, double angle, double speed_limit,
    const std::pair<std::string, std::string>& nodes) {
  RoadNetwork network;
  for (int lane = 0; lane < lanes; ++lane) {
    const Vec2 origin =
        RotateRoadPoint({start, lane * kDefaultLaneWidth}, angle);
    const Vec2 end =
        RotateRoadPoint({start + length, lane * kDefaultLaneWidth}, angle);
    const std::array<LineType, 2> line_types = {
        lane == 0 ? LineType::kContinuousLine : LineType::kStriped,
        lane == lanes - 1 ? LineType::kContinuousLine : LineType::kNone};
    network.AddLane(nodes.first, nodes.second,
                    Lane::Straight(origin, end, kDefaultLaneWidth, line_types,
                                   false, speed_limit));
  }
  return network;
}

const RoadNetwork::Edge& RoadNetwork::GetEdge(const std::string& from,
                                              const std::string& to) const {
  const Edge* edge = FindEdge(from, to);
  if (edge == nullptr) {
    throw std::out_of_range("invalid highway road edge");
  }
  return *edge;
}

const RoadNetwork::Edge* RoadNetwork::FindEdge(const std::string& from,
                                               const std::string& to) const {
  const auto it =
      std::find_if(edges_.begin(), edges_.end(),
                   [&](const Edge& e) { return e.from == from && e.to == to; });
  return it == edges_.end() ? nullptr : &*it;
}

RoadNetwork::Edge* RoadNetwork::FindEdge(const std::string& from,
                                         const std::string& to) {
  const auto it =
      std::find_if(edges_.begin(), edges_.end(),
                   [&](const Edge& e) { return e.from == from && e.to == to; });
  return it == edges_.end() ? nullptr : &*it;
}

std::vector<std::string> RoadNetwork::OutgoingNodes(
    const std::string& from) const {
  std::vector<std::string> nodes;
  for (const auto& edge : edges_) {
    if (edge.from == from) {
      nodes.push_back(edge.to);
    }
  }
  return nodes;
}

std::pair<int, double> RoadNetwork::NextLaneGivenNextRoad(
    const LaneIndex& current_index, const std::string& next_to, int next_id,
    Vec2 projected_position) const {
  const Edge& current_edge = GetEdge(current_index.from, current_index.to);
  const Edge& next_edge = GetEdge(current_index.to, next_to);
  if (next_id == kUnknownLaneId &&
      current_edge.lanes.size() == next_edge.lanes.size()) {
    next_id = current_index.id;
  }
  if (next_id != kUnknownLaneId) {
    const LaneIndex next_lane_index{current_index.to, next_to, next_id};
    return {next_id, GetLane(next_lane_index).Distance(projected_position)};
  }
  int best_id = 0;
  double best_distance = next_edge.lanes[0].Distance(projected_position);
  for (int id = 1; static_cast<std::size_t>(id) < next_edge.lanes.size();
       ++id) {
    const double distance = next_edge.lanes[id].Distance(projected_position);
    if (distance < best_distance) {
      best_id = id;
      best_distance = distance;
    }
  }
  return {best_id, best_distance};
}

}  // namespace highway::official
