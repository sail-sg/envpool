/*
 * Copyright 2026 Garena Online Private Limited
 * Copyright 2019-2021 The dm_control Authors.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "envpool/mujoco/locomotion/scene.h"
#include "third_party/dmc_locomotion/metadata.h"

namespace mujoco_locomotion {
namespace {

std::string PlayerName(int player, int team_size) {
  return std::string(player < team_size ? "home" : "away") +
         std::to_string(player % team_size) + "/";
}

std::array<double, 3> HoardingColor(double x, double y) {
  const double pi = std::acos(-1);
  const double angle = pi + std::atan2(x, -std::abs(y));
  double hue = .5 + angle / (2 * pi) - .25;
  hue -= std::floor(hue);
  const int sector = static_cast<int>(hue * 6);
  const double fraction = hue * 6 - sector;
  const double p = 1 - .7;
  const double q = 1 - .7 * fraction;
  const double t = 1 - .7 * (1 - fraction);
  const std::array<std::array<double, 3>, 6> colors{
      {{1, t, p}, {q, 1, p}, {p, 1, t}, {p, q, 1}, {t, p, 1}, {1, p, q}}};
  return {colors[sector % 6][0], colors[sector % 6][1], colors[sector % 6][2]};
}

}  // namespace

void Scene::Detector(const std::string& name,
                     const std::array<double, 3>& position,
                     const std::array<double, 3>& size, bool goal,
                     int direction) {
  pugi::xml_document document;
  auto model = document.append_child("mujoco");
  auto body = Child(model, "worldbody");
  const std::array<double, 4> rgba{
      !goal || direction < 0 ? 1 : .2, goal ? .2 : 1,
      !goal || direction > 0 ? 1 : .2, goal ? .5 : 1};
  auto zone = body.append_child("site");
  Set(zone, "name", "detection_zone");
  Set(zone, "type", "box");
  Set(zone, "pos", Numbers(position.data(), 3));
  Set(zone, "size", Numbers(size.data(), 3));
  Set(zone, "rgba", Numbers(rgba.data(), 4));
  Set(zone, "group", 4);
  const std::array<std::string, 3> corner_names{"lower", "mid", "upper"};
  for (int corner = -1; corner <= 1; ++corner) {
    const auto& key = corner_names[corner + 1];
    auto site = body.append_child("site");
    Set(site, "name", key);
    Set(site, "pos",
        {position[0] + corner * size[0], position[1] + corner * size[1],
         goal ? position[2] + corner * size[2] : 0});
    Set(site, "size", {.05});
    Set(site, "rgba", Numbers(rgba.data(), 4));
    Set(site, "group", 4);
    auto sensor = Child(model, "sensor").append_child("framepos");
    Set(sensor, "name", std::string(name).append("_").append(key));
    Set(sensor, "objtype", "site");
    Set(sensor, "objname", key);
  }
  if (goal) {
    // Python 3.12+ sum() compensates roundoff. The two equal, smaller goal
    // dimensions add exactly before the longer dimension is accumulated.
    const double radius = .07 * ((size[0] + size[2]) + size[1]) / 3;
    for (const auto& post : kGoalPosts) {
      std::array<double, 6> fromto;
      for (int i = 0; i < 6; ++i) {
        fromto[i] =
            post.fromto[i] * (i % 3 == 2 ? 1 : direction) * size[i % 3] +
            position[i % 3];
      }
      double post_radius = radius;
      if (post.name.find("top") != std::string_view::npos) {
        post_radius *= 1.01;
      }
      if (post.name.find("support") != std::string_view::npos) {
        post_radius *= .75;
      }
      auto geom = body.append_child("geom");
      Set(geom, "name", std::string(post.name));
      Set(geom, "type", "capsule");
      Set(geom, "size", {post_radius});
      Set(geom, "fromto", Numbers(fromto.data(), 6));
      Set(geom, "rgba", {rgba[0], rgba[1], rgba[2], 1});
    }
  }
  Attach(model, name + "/", RootJoint::kFixed);
}

void Scene::SoccerBall(bool humanoid, bool field_box) {
  const double radius = humanoid ? .117 : .35;
  pugi::xml_document document;
  auto model = document.append_child("mujoco");
  auto geom = Child(model, "worldbody").append_child("geom");
  Set(geom, "name", "geom");
  Set(geom, "type", "sphere");
  Set(geom, "size", {radius});
  Set(geom, "pos", {0, 0, radius});
  Set(geom, "condim", 6);
  Set(geom, "priority", 1);
  Set(geom, "mass", humanoid ? .45 : .045);
  Set(geom, "friction",
      humanoid ? Numbers({.7, .05, .04}) : Numbers({.7, .075, .075}));
  Set(geom, "solref", {.02, humanoid ? .4 : 1});
  Set(geom, "material", "soccer_ball");
  if (field_box) {
    Set(geom, "contype", 129);
  }
  const std::array types = {"framepos", "framequat", "framelinvel",
                            "frameangvel"};
  const std::array names = {"position", "orientation", "linear_velocity",
                            "angular_velocity"};
  for (int i = 0; i < 4; ++i) {
    auto sensor = Child(model, "sensor").append_child(types[i]);
    Set(sensor, "name", names[i]);
    Set(sensor, "objtype", "geom");
    Set(sensor, "objname", "geom");
  }
  auto texture = Child(model, "asset").append_child("texture");
  Set(texture, "name", "soccer_ball");
  Set(texture, "type", "cube");
  for (const char* face : {"up", "down", "front", "back", "left", "right"}) {
    Set(texture, (std::string("file") + face).c_str(),
        asset_path_ + "/locomotion/soccer/assets/soccer_ball/" + face + ".png");
  }
  auto material = Child(model, "asset").append_child("material");
  Set(material, "name", "soccer_ball");
  Set(material, "texture", "soccer_ball");
  const std::array<double, 3> distances{2, 7, 10};
  const std::array camera_names{"ball_cam_near", "ball_cam", "ball_cam_far"};
  for (int i = 0; i < 3; ++i) {
    const double distance = distances[i];
    auto camera = Child(model, "worldbody").append_child("camera");
    Set(camera, "name", camera_names[i]);
    Set(camera, "pos", {0, -distance, distance});
    Set(camera, "zaxis", {0, -1, 1});
    Set(camera, "fovy", 70);
    Set(camera, "mode", "trackcom");
  }
  Attach(model, "soccer_ball/");
}

void Scene::Soccer(const TaskConfig& task, int team_size, bool field_box,
                   bool keep_aspect_ratio, bool disable_contacts,
                   RandomState* random) {
  const bool humanoid = task.walker == Walker::kCmu2019;
  std::array<double, 2> minimum{32, 24};
  std::array<double, 2> maximum{48, 36};
  if (humanoid) {
    minimum = {std::sqrt(100. * team_size * 2 / .75) / 2,
               std::sqrt(100. * team_size * 2 * .75) / 2};
    maximum = {std::sqrt(350. * team_size * 2 / .75) / 2,
               std::sqrt(350. * team_size * 2 * .75) / 2};
  }
  // Soccer.Task calls the arena hook explicitly, then Composer calls the root
  // entity hook. Both resizes consume RNG, although only the second is
  // compiled.
  for (int hook = 0; hook < 2; ++hook) {
    const double x = random->Uniform();
    const double y = keep_aspect_ratio ? x : random->Uniform();
    pitch_size = {minimum[0] + x * (maximum[0] - minimum[0]),
                  minimum[1] + y * (maximum[1] - minimum[1])};
  }
  goal_size = humanoid ? std::array<double, 3>{1.22 / 2, 3.66 / 2, 1.22 / 2}
                       : std::array<double, 3>{
                             (32. / 6) / 2, pitch_size[1] * .33, (32. / 6) / 2};
  field_size = {pitch_size[0] - 2 * goal_size[0],
                pitch_size[1] - 2 * goal_size[0]};
  const double extent = .1 * std::max(maximum[0], maximum[1]);
  Set(Child(Root(), "statistic"), "extent", extent);
  Set(Child(Root(), "statistic"), "center", {0, 0, extent});
  auto visual = Child(Root(), "visual");
  Set(Child(visual, "map"), "zfar", 50);
  Set(Child(visual, "map"), "znear", .1 / extent);
  Set(Child(visual, "quality"), "shadowsize", 8192);
  auto camera = World().append_child("camera");
  Set(camera, "name", "top_down");
  Set(camera, "pos", {0, 0, 95});
  Set(camera, "zaxis", {0, 0, 1});
  Set(camera, "fovy",
      360 / std::acos(-1) *
          std::atan2(1.1 * std::max(pitch_size[0], pitch_size[1]), 95));
  auto sky = Asset().append_child("texture");
  Set(sky, "name", "skybox");
  Set(sky, "type", "skybox");
  Set(sky, "builtin", "gradient");
  Set(sky, "rgb1", {.7, .9, .9});
  Set(sky, "rgb2", {.03, .09, .27});
  Set(sky, "width", 400);
  Set(sky, "height", 400);
  int light_index = 0;
  for (int sx : {-1, 1}) {
    for (int sy : {-1, 1}) {
      const double height = .5 * (field_size[0] + field_size[1]) * 2 / 3;
      auto light = World().append_child("light");
      Set(light, "name", "//unnamed_light_" + std::to_string(light_index++));
      Set(light, "cutoff", 60);
      Set(light, "pos", {sx * field_size[0], sy * field_size[1], height});
      Set(light, "dir",
          {-sx * field_size[0], -sy * field_size[1], -height * 2});
    }
  }
  auto texture = Asset().append_child("texture");
  Set(texture, "name", "fieldplane");
  Set(texture, "type", "2d");
  Set(texture, "file",
      asset_path_ + "/locomotion/soccer/assets/pitch/pitch_nologo_l.png");
  auto material = Asset().append_child("material");
  Set(material, "name", "fieldplane");
  Set(material, "texture", "fieldplane");
  const double grid = std::max(pitch_size[0], pitch_size[1]) * .01;
  auto ground = World().append_child("geom");
  Set(ground, "name", "ground");
  Set(ground, "type", "plane");
  Set(ground, "material", "fieldplane");
  Set(ground, "size", {field_size[0], field_size[1], grid});
  ground_geoms.emplace_back("ground");
  const std::array<std::array<double, 6>, 4> axes{{{-1, 0, 0, 0, 0, 1},
                                                   {1, 0, 0, 0, 0, 1},
                                                   {0, 1, 0, 0, 0, 1},
                                                   {0, -1, 0, 0, 0, 1}}};
  for (int i = 0; i < 4; ++i) {
    auto wall = World().append_child("geom");
    Set(wall, "name", "//unnamed_geom_" + std::to_string(i + 1));
    Set(wall, "type", "plane");
    Set(wall, "rgba", {.1, .1, .1, .8});
    Set(wall, "size", {1e-7, 1e-7, 1e-7});
    Set(wall, "pos",
        {i < 2 ? 0 : (i == 2 ? -1 : 1) * pitch_size[0],
         i > 1 ? 0 : (i == 0 ? -1 : 1) * pitch_size[1], 0});
    Set(wall, "xyaxes", Numbers(axes[i].data(), 6));
  }
  Detector("home_goal", {-pitch_size[0] + goal_size[0], 0, goal_size[2]},
           goal_size, true, 1);
  Detector("away_goal", {pitch_size[0] - goal_size[0], 0, goal_size[2]},
           goal_size, true, -1);
  Detector("field", {0, 0, 0}, {field_size[0], field_size[1], .01}, false, 0);
  int geom_index = 5;
  for (int x : {-1, 0, 1}) {
    for (int y : {-1, 0, 1}) {
      if (x == 0 && y == 0) {
        continue;
      }
      auto geom = World().append_child("geom");
      Set(geom, "name", "//unnamed_geom_" + std::to_string(geom_index++));
      Set(geom, "type", "plane");
      Set(geom, "rgba", {.306, .682, .223, 1});
      Set(geom, "contype", 0);
      Set(geom, "conaffinity", 0);
      Set(geom, "size",
          {(x != 0) ? goal_size[0] : field_size[0],
           (y != 0) ? goal_size[0] : field_size[1], grid});
      Set(geom, "pos",
          {x * (pitch_size[0] - goal_size[0]),
           y * (pitch_size[1] - goal_size[0]), 0});
    }
  }
  if (field_box) {
    const double corner = .5 * (field_size[1] + goal_size[1]);
    const double half = .5 * (field_size[1] - goal_size[1]);
    const std::array<std::array<double, 3>, 8> poses{
        {{0, -field_size[1] - 1, 20},
         {0, field_size[1] + 1, 20},
         {-field_size[0] - 1, -corner, 20},
         {-field_size[0] - 1, 0, 20 + goal_size[2]},
         {-field_size[0] - 1, corner, 20},
         {field_size[0] + 1, -corner, 20},
         {field_size[0] + 1, 0, 20 + goal_size[2]},
         {field_size[0] + 1, corner, 20}}};
    for (int i = 0; i < 8; ++i) {
      auto geom = World().append_child("geom");
      Set(geom, "name", "//unnamed_geom_" + std::to_string(geom_index++));
      Set(geom, "type", "box");
      Set(geom, "rgba", {.3, .3, .3, 0});
      Set(geom, "contype", 128);
      Set(geom, "conaffinity", 128);
      Set(geom, "pos", Numbers(poses[i].data(), 3));
      double half_width = half;
      if (i < 2) {
        half_width = 1;
      } else if (i == 3 || i == 6) {
        half_width = goal_size[1];
      }
      Set(geom, "size",
          {i < 2 ? field_size[0] : 1, half_width,
           i == 3 || i == 6 ? 20 - goal_size[2] : 20});
    }
  }
  int site_index = 0;
  for (int dim = 0; dim < 2; ++dim) {
    const double width = goal_size[2] / 8;
    const double height = goal_size[2] / 2;
    const double length = pitch_size[dim] + ((dim != 0) ? 2 * width : 0);
    for (int sign : {-1, 1}) {
      for (int i = 0; i < 30; ++i) {
        std::array<double, 3> position{0, 0, width};
        std::array<double, 3> half{height, height, height};
        position[dim] = i * ((2 * length) / 30) - length + length / 30;
        position[1 - dim] = sign * (pitch_size[1 - dim] + width);
        half[dim] = length / 30;
        half[1 - dim] = width;
        auto color = HoardingColor(position[0], position[1]);
        auto site = World().append_child("site");
        Set(site, "name", "//unnamed_site_" + std::to_string(site_index++));
        Set(site, "type", "box");
        Set(site, "pos", Numbers(position.data(), 3));
        Set(site, "size", Numbers(half.data(), 3));
        Set(site, "rgba", {color[0], color[1], color[2], 1});
      }
    }
  }
  SoccerBall(humanoid, field_box);
  for (int player = 0; player < team_size * 2; ++player) {
    const auto prefix = PlayerName(player, team_size);
    AddWalker(task.walker, prefix, player % team_size, player >= team_size);
    if (disable_contacts) {
      for (auto entry : Find(World(), "body", prefix).select_nodes(".//geom")) {
        Set(entry.node(), "contype", 0);
      }
    }
  }
  SoccerSensors(task.walker, team_size);
  // Soccer sets these limits after attaching the walkers; do not add each
  // CMU model's standalone allocation limits to the team-wide limits.
  Set(Child(Root(), "size"), "njmax", 400 * team_size * 2);
  Set(Child(Root(), "size"), "nconmax", 200 * team_size * 2);
}

void Scene::SoccerSensors(Walker walker, int team_size) {
  auto sensors = Child(Root(), "sensor");
  std::string torso = "root";
  std::vector<std::string> effectors{"rradius", "lradius", "rfoot", "lfoot"};
  if (walker == Walker::kBoxhead) {
    torso = "head_body";
    effectors = {"head_body"};
  } else if (walker == Walker::kAnt) {
    torso = "torso";
    effectors = {"front_left_foot", "front_right_foot", "back_right_foot",
                 "back_left_foot"};
  }
  for (int player = 0; player < team_size * 2; ++player) {
    const auto prefix = PlayerName(player, team_size);
    pugi::xml_node last;
    for (auto sensor : sensors.children()) {
      if (std::string(sensor.attribute("name").value()).find(prefix) == 0) {
        last = sensor;
      }
    }
    auto add = [&](const std::string& type, const std::string& name,
                   const std::string& target) {
      auto sensor = sensors.insert_child_after(type.c_str(), last);
      last = sensor;
      Set(sensor, "name", prefix + name);
      Set(sensor, "objtype", "body");
      Set(sensor, "objname", target);
      Set(sensor, "reftype", "body");
      Set(sensor, "refname", prefix + torso);
    };
    add("frameangvel", "ball_ego_angvel", "soccer_ball/");
    add("framepos", "ball_ego_pos", "soccer_ball/");
    add("framelinvel", "ball_ego_linvel", "soccer_ball/");
    int teammate = 0;
    int opponent = 0;
    for (int other = 0; other < team_size * 2; ++other) {
      if (other == player) {
        continue;
      }
      const bool same = (other < team_size) == (player < team_size);
      const auto key = std::string(same ? "teammate_" : "opponent_") +
                       std::to_string(same ? teammate++ : opponent++);
      const auto other_prefix = PlayerName(other, team_size);
      for (const auto& effector : effectors) {
        add("framepos",
            std::string(effector).append("_").append(key).append(
                "_end_effector"),
            other_prefix + effector);
      }
      add("framelinvel", key + "_ego_linear_velocity", other_prefix + torso);
      add("framepos", key + "_ego_position", other_prefix + torso);
      for (char axis : std::string("xyz")) {
        add(std::string("frame") + axis + "axis",
            key + "_ego_orientation_" + axis, other_prefix + torso);
      }
    }
  }
}

}  // namespace mujoco_locomotion
