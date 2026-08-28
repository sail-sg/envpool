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
// Native translation of dm_control 1.0.44's Composer arenas and walkers.

#include "envpool/mujoco/locomotion/scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lodepng.h"
#include "third_party/dmc_locomotion/metadata.h"

namespace mujoco_locomotion {
namespace {

void Load(pugi::xml_document* document, const std::string& path) {
  const auto result = document->load_file(path.c_str());
  if (!result) {
    throw std::runtime_error("Cannot load locomotion asset " + path + ": " +
                             result.description());
  }
}

void Namespace(pugi::xml_node node, const std::string& prefix) {
  static const std::set<std::string_view> references{
      "name",    "class",   "childclass", "material",  "texture",
      "mesh",    "skin",    "joint",      "joint1",    "joint2",
      "site",    "site1",   "site2",      "tendon",    "tendon1",
      "tendon2", "body",    "body1",      "body2",     "target",
      "objname", "refname", "actuator",   "cranksite", "slidersite"};
  for (auto attribute : node.attributes()) {
    if ((references.count(attribute.name()) != 0u) &&
        ((*attribute.value()) != 0)) {
      attribute.set_value((prefix + attribute.value()).c_str());
    }
  }
  for (auto child : node.children()) {
    Namespace(child, prefix);
  }
}

std::vector<double> Values(pugi::xml_attribute attribute) {
  std::istringstream input(attribute.value());
  std::vector<double> values;
  double value;
  while (input >> value) {
    values.push_back(value);
  }
  return values;
}

void AddFrameSensor(pugi::xml_node sensors, const char* type,
                    const std::string& body, const std::string& name,
                    const std::string& torso) {
  auto sensor = sensors.append_child(type);
  Set(sensor, "name", name);
  Set(sensor, "objtype", "xbody");
  Set(sensor, "objname", body);
  Set(sensor, "reftype", "xbody");
  Set(sensor, "refname", torso);
}

void MergeSettings(pugi::xml_node destination, pugi::xml_node source) {
  for (auto attribute : source.attributes()) {
    if (!destination.attribute(attribute.name())) {
      Set(destination, attribute.name(), attribute.value());
    }
  }
  for (auto node : source.children()) {
    MergeSettings(Child(destination, node.name()), node);
  }
}

}  // namespace

TaskConfig GetTaskConfig(const std::string& name) {
  if (std::find(kTaskNames.begin(), kTaskNames.end(), name) ==
      kTaskNames.end()) {
    throw std::invalid_argument("Unknown dm_control locomotion task: " + name);
  }
  TaskConfig result{Task::kTarget, Walker::kCmu2019, 0.005, 0.03, 30};
  if (name.find("rodent_") == 0) {
    result.walker = Walker::kRodent;
    result.physics_timestep = 0.001;
    result.control_timestep = 0.02;
  }
  if (name.find("run_walls") != std::string::npos) {
    result.task = Task::kWalls;
  }
  if (name.find("run_gaps") != std::string::npos) {
    result.task = Task::kGaps;
  }
  if (name.find("maze_forage") != std::string::npos) {
    result.task = Task::kForage;
  }
  if (name.find("heterogeneous") != std::string::npos) {
    result.task = Task::kHeterogeneous;
    result.physics_timestep = 0.001;
    result.time_limit = 25;
  }
  if (name == "rodent_escape_bowl") {
    result.task = Task::kBowl;
    result.time_limit = 20;
  }
  if (name == "rodent_two_touch") {
    result.task = Task::kTwoTouch;
  }
  if (name == "cmu_humanoid_tracking") {
    result.task = Task::kTracking;
    result.walker = Walker::kCmu2020;
    result.control_timestep = 0.03;
  }
  if (name.find("soccer_") == 0) {
    result.task = Task::kSoccer;
    result.physics_timestep = 0.005;
    result.control_timestep = 0.025;
    result.time_limit = 45;
    if (name == "soccer_boxhead") {
      result.walker = Walker::kBoxhead;
    }
    if (name == "soccer_ant") {
      result.walker = Walker::kAnt;
    }
  }
  return result;
}

std::string Numbers(const double* values, int size) {
  std::ostringstream output;
  output << std::setprecision(17);
  for (int i = 0; i < size; ++i) {
    if (i != 0) {
      output << ' ';
    }
    output << values[i];
  }
  return output.str();
}

std::string Numbers(std::initializer_list<double> values) {
  return Numbers(values.begin(), values.size());
}

void Set(pugi::xml_node node, const char* key, const std::string& value) {
  auto attribute = node.attribute(key);
  if (!attribute) {
    attribute = node.append_attribute(key);
  }
  attribute.set_value(value.c_str());
}

void Set(pugi::xml_node node, const char* key, double value) {
  Set(node, key, Numbers({value}));
}

void Set(pugi::xml_node node, const char* key,
         std::initializer_list<double> values) {
  Set(node, key, Numbers(values));
}

pugi::xml_node Child(pugi::xml_node parent, const char* tag) {
  auto child = parent.child(tag);
  return (child != nullptr) ? child : parent.append_child(tag);
}

pugi::xml_node Find(pugi::xml_node root, const char* tag,
                    const std::string& name) {
  const std::string query = std::string(".//") + tag + "[@name='" + name + "']";
  return root.select_node(query.c_str()).node();
}

Scene::Scene(std::string asset_path, std::string labmaze_asset_path)
    : asset_path_(std::move(asset_path)),
      // cpplint misclassifies an empty multiline constructor as a namespace.
      // NOLINTNEXTLINE(whitespace/indent_namespace)
      labmaze_asset_path_(std::move(labmaze_asset_path)) {}

void Scene::LoadArena(const std::string& name, double timestep) {
  document_.reset();
  ground_geoms.clear();
  target_positions.clear();
  spawn_positions.clear();
  targets.clear();
  target_types.clear();
  virtual_files_.clear();
  Load(&document_, asset_path_ + "/composer/arena.xml");
  Root().child("compiler").remove_attribute("coordinate");
  Set(Root(), "model", name);
  Set(Child(Root(), "option"), "timestep", timestep);
}

void Scene::OutdoorTexture() {
  for (auto node = Asset().first_child(); node != nullptr;) {
    const auto next = node.next_sibling();
    if (std::string(node.attribute("type").value()) == "skybox") {
      Asset().remove_child(node);
    }
    node = next;
  }
  const auto path = asset_path_ + "/locomotion/arenas/assets/outdoor_natural/";
  auto texture = Asset().append_child("texture");
  Set(texture, "name", "aesthetic_texture");
  Set(texture, "type", "2d");
  Set(texture, "file", path + "OutdoorGrassFloorD.png");
  auto material = Asset().append_child("material");
  Set(material, "name", "aesthetic_material");
  Set(material, "texture", "aesthetic_texture");
  Set(material, "texuniform", "true");
  auto sky = Asset().append_child("texture");
  Set(sky, "name", "aesthetic_skybox");
  Set(sky, "type", "skybox");
  Set(sky, "file", path + "OutdoorSkybox2048.png");
  Set(sky, "gridsize", "3 4");
  Set(sky, "gridlayout", ".U..LFRB.D..");
}

void Scene::Floor(double size, bool outdoor) {
  auto headlight = Child(Child(Root(), "visual"), "headlight");
  Set(headlight, "ambient", {.4, .4, .4});
  Set(headlight, "diffuse", {.8, .8, .8});
  Set(headlight, "specular", {.1, .1, .1});
  if (outdoor) {
    OutdoorTexture();
  } else {
    auto texture = Asset().append_child("texture");
    Set(texture, "name", "groundplane");
    Set(texture, "type", "2d");
    Set(texture, "builtin", "checker");
    Set(texture, "rgb1", {.2, .3, .4});
    Set(texture, "rgb2", {.1, .2, .3});
    Set(texture, "width", 200);
    Set(texture, "height", 200);
    Set(texture, "mark", "edge");
    Set(texture, "markrgb", {.8, .8, .8});
    auto material = Asset().append_child("material");
    Set(material, "name", "groundplane");
    Set(material, "texture", "groundplane");
    Set(material, "texrepeat", {2, 2});
    Set(material, "texuniform", "true");
    Set(material, "reflectance", .2);
  }
  auto ground = World().append_child("geom");
  Set(ground, "name", "groundplane");
  Set(ground, "type", "plane");
  Set(ground, "material", outdoor ? "aesthetic_material" : "groundplane");
  Set(ground, "size", {size, size, .25});
  ground_geoms.emplace_back("groundplane");
  auto camera = World().append_child("camera");
  Set(camera, "name", "top_camera");
  Set(camera, "pos", {0, 0, 100});
  Set(camera, "quat", {1, 0, 0, 0});
  Set(camera, "fovy", 360 / std::acos(-1) * std::atan2(1.1 * size, 100));
}

void Scene::Corridor(const TaskConfig& task, RandomState* random) {
  const bool rodent = task.walker == Walker::kRodent;
  const double length = rodent ? 40 : 100;
  const double width = rodent ? 2 : 10;
  auto walls = World().append_child("body");
  Set(walls, "name", "walls");
  auto visual = Child(Root(), "visual");
  Set(Child(visual, "map"), "znear", .00025);
  Set(Child(visual, "map"), "zfar", 4);
  auto headlight = Child(visual, "headlight");
  Set(headlight, "ambient", {.4, .4, .4});
  Set(headlight, "diffuse", {.8, .8, .8});
  Set(headlight, "specular", {.1, .1, .1});
  auto sky = Asset().append_child("texture");
  Set(sky, "type", "skybox");
  Set(sky, "builtin", "gradient");
  Set(sky, "rgb1", {.4, .6, .8});
  Set(sky, "rgb2", {0, 0, 0});
  Set(sky, "width", 100);
  Set(sky, "height", 600);
  auto ground = World().append_child("geom");
  Set(ground, "name", "//unnamed_geom_0");
  Set(ground, "type", "plane");
  Set(ground, "rgba", {.5, .5, .5, 1});
  Set(ground, "pos", {length / 2, 0, 0});
  Set(ground, "size", {length / 2 + 2, width / 2, 1});
  ground_geoms.emplace_back("//unnamed_geom_0");
  const std::array<std::array<double, 6>, 4> axes{{{1, 0, 0, 0, 0, 1},
                                                   {-1, 0, 0, 0, 0, 1},
                                                   {0, 1, 0, 0, 0, 1},
                                                   {0, -1, 0, 0, 0, 1}}};
  const std::array<std::array<double, 3>, 4> positions{
      {{length / 2, width / 2, 2},
       {length / 2, -width / 2, 2},
       {-2, 0, 2},
       {length + 2, 0, 2}}};
  for (int i = 0; i < 4; ++i) {
    auto plane = World().append_child("geom");
    Set(plane, "name", "//unnamed_geom_" + std::to_string(i + 1));
    Set(plane, "type", "plane");
    Set(plane, "pos", Numbers(positions[i].data(), 3));
    Set(plane, "xyaxes", Numbers(axes[i].data(), 6));
    Set(plane, "size", {i < 2 ? length / 2 + 2 : width / 2, 2, 1});
    Set(plane, "rgba", {1, 0, 0, 0});
    Set(plane, "group", 3);
  }
  if (task.task == Task::kWalls) {
    int index = 0;
    for (int distance = 2; distance < length; distance += 4, ++index) {
      const double x = distance;
      const double wall_width = random->Uniform(1, 7);
      auto wall = walls.append_child("geom");
      Set(wall, "type", "box");
      Set(wall, "name", "wall_" + std::to_string(index));
      Set(wall, "pos",
          {x, (((index % 2) != 0) ? -1 : 1) * (width - wall_width) / 2, 1.5});
      Set(wall, "size", {.08, wall_width / 2, 1.5});
      Set(wall, "rgba", {1, 1, 1, 1});
    }
    return;
  }
  Set(ground, "pos", {length / 2, 0, -10});
  Set(ground, "rgba", {0, 0, 0, 0});
  if (rodent) {
    OutdoorTexture();
  }
  auto platforms = World().append_child("body");
  Set(platforms, "name", "ground");
  auto add_platform = [&](const std::string& name, double start, double size) {
    auto geom = platforms.append_child("geom");
    Set(geom, "name", name);
    Set(geom, "type", "box");
    Set(geom, "pos", {start + size / 2, 0, -.16});
    Set(geom, "size", {size / 2, width / 2, .16});
    if (rodent) {
      Set(geom, "material", "aesthetic_material");
    } else {
      Set(geom, "rgba", {.5, .5, .5, 1});
    }
    ground_geoms.push_back(name);
  };
  add_platform("start_floor", 0, 6);
  double x = 6;
  for (int index = 0; x < length; ++index) {
    const double size =
        rodent ? random->Uniform(.4, .8) : random->Uniform(.3, 2.5);
    add_platform("floor_" + std::to_string(index), x, size);
    x += size + (rodent ? random->Uniform(.05, .2) : random->Uniform(.5, 1.25));
  }
}

void Scene::Attach(pugi::xml_node model, const std::string& prefix,
                   RootJoint joints) {
  // PyMJCF merges singleton model settings on attachment. Arena settings take
  // precedence, but walker defaults (notably rodent's shadow map) survive.
  for (const char* section : {"visual", "statistic"}) {
    if (model.child(section) != nullptr) {
      MergeSettings(Child(Root(), section), model.child(section));
    }
  }
  for (auto attribute : model.child("size").attributes()) {
    auto size = Child(Root(), "size");
    Set(size, attribute.name(),
        size.attribute(attribute.name()).as_double() + attribute.as_double());
  }
  auto defaults = Child(model, "default");
  Set(defaults, "class", "");
  Namespace(model, prefix);
  Set(defaults, "class", prefix);
  Child(Root(), "default").append_copy(defaults);
  for (const char* section :
       {"asset", "contact", "equality", "tendon", "actuator", "sensor"}) {
    for (auto node : model.child(section).children()) {
      const std::string_view tag = node.name();
      if (!node.attribute("class") &&
          (std::string_view(section) == "actuator" ||
           std::string_view(section) == "tendon" || tag == "material" ||
           tag == "mesh" || tag == "pair")) {
        Set(node, "class", prefix);
      }
      if (tag == "skin" && (node.attribute("file") != nullptr)) {
        std::ifstream input(node.attribute("file").value(), std::ios::binary);
        std::vector<unsigned char> bytes(std::istreambuf_iterator<char>{input},
                                         {});
        auto integer = [&](std::size_t offset) {
          if (offset + 4 > bytes.size()) {
            throw std::runtime_error("Truncated skin asset");
          }
          uint32_t value = 0;
          for (int i = 0; i < 4; ++i) {
            value |= static_cast<uint32_t>(bytes[offset + i]) << (8 * i);
          }
          return value;
        };
        const auto bones = integer(12);
        std::size_t offset =
            16 + 4 * (3 * integer(0) + 2 * integer(4) + 3 * integer(8));
        for (uint32_t bone = 0; bone < bones; ++bone) {
          if (offset + 72 > bytes.size()) {
            throw std::runtime_error("Truncated skin bone");
          }
          const std::string body(
              reinterpret_cast<char*>(bytes.data() + offset),
              strnlen(reinterpret_cast<char*>(bytes.data() + offset), 40));
          const auto name = prefix + body;
          if (name.size() >= 40) {
            throw std::runtime_error("Skin bone name too long");
          }
          std::fill_n(bytes.data() + offset, 40, 0);
          std::memcpy(bytes.data() + offset, name.data(), name.size());
          const auto vertices = integer(offset + 68);
          offset += 72 + vertices * 8;
        }
        auto filename = prefix + node.attribute("name").value() + ".skn";
        std::replace(filename.begin(), filename.end(), '/', '_');
        virtual_files_[filename] = std::move(bytes);
        Set(node, "file", filename);
      }
      Child(Root(), std::string_view(section) == "asset"
                        ? "envpool_attached_asset"
                        : section)
          .append_copy(node);
    }
  }
  auto frame = World().append_child("body");
  Set(frame, "name", prefix);
  Set(frame, "childclass", prefix);
  if (joints == RootJoint::kFree) {
    auto joint = frame.append_child("freejoint");
    Set(joint, "name", prefix);
  } else if (joints == RootJoint::kSlides) {
    for (int i = 0; i < 3; ++i) {
      auto joint = frame.append_child("joint");
      Set(joint, "name", prefix + "root_" + std::string(1, "xyz"[i]) + "/");
      Set(joint, "type", "slide");
      Set(joint, "class", prefix + "root");
      Set(joint, "axis",
          {static_cast<double>(i == 0), static_cast<double>(i == 1),
           static_cast<double>(i == 2)});
    }
  }
  for (auto node : model.child("worldbody").children()) {
    frame.append_copy(node);
  }
}

void Scene::AddWalker(Walker walker, const std::string& prefix, int player,
                      bool red) {
  std::string relative;
  if (walker == Walker::kCmu2019) {
    relative = "locomotion/walkers/assets/humanoid_CMU_V2019.xml";
  } else if (walker == Walker::kCmu2020) {
    relative = "locomotion/walkers/assets/humanoid_CMU_V2020.xml";
  } else if (walker == Walker::kRodent) {
    relative = "locomotion/walkers/assets/rodent.xml";
  } else if (walker == Walker::kBoxhead) {
    relative = "locomotion/soccer/assets/boxhead/boxhead.xml";
  } else {
    relative = "third_party/ant/ant.xml";
  }
  pugi::xml_document document;
  const std::string path = asset_path_ + "/" + relative;
  Load(&document, path);
  auto model = document.child("mujoco");
  for (auto entry : model.select_nodes(".//*[@file]")) {
    auto node = entry.node();
    const auto file = std::filesystem::path(path).parent_path() /
                      node.attribute("file").value();
    Set(node, "file", file.lexically_normal().string());
  }
  const bool cmu = walker == Walker::kCmu2019 || walker == Walker::kCmu2020;
  if (cmu && (walker == Walker::kCmu2020 || player >= 0)) {
    CmuVisuals(model, walker, player, red);
  }
  if (cmu) {
    auto actuators = Child(model, "actuator");
    actuators.remove_children();
    Set(Child(Child(model, "default"), "general"), "forcelimited", "true");
    const auto& parameters =
        walker == Walker::kCmu2020 ? kCmu2020Actuators : kCmu2019Actuators;
    for (const auto& parameter : parameters) {
      auto joint = Find(model, "joint", parameter.name);
      const auto range = Values(joint.attribute("range"));
      if (range.size() != 2) {
        throw std::runtime_error("Missing CMU joint range");
      }
      if (parameter.damping >= 0) {
        Set(joint, "damping", parameter.damping);
      }
      const double slope = (range[1] - range[0]) / 2;
      auto actuator = actuators.append_child("general");
      Set(actuator, "name", parameter.name);
      Set(actuator, "joint", parameter.name);
      Set(actuator, "biastype", "affine");
      Set(actuator, "gainprm", {parameter.kp * slope});
      Set(actuator, "biasprm",
          {parameter.kp * (range[0] + slope), -parameter.kp, 0});
      Set(actuator, "ctrllimited", "true");
      Set(actuator, "ctrlrange", {-1, 1});
      Set(actuator, "forcerange", {parameter.low, parameter.high});
      if (walker == Walker::kCmu2020) {
        Set(actuator, "dyntype", "filter");
        Set(actuator, "dynprm", {.030});
      }
    }
  }
  std::vector<std::string> effectors;
  std::string torso;
  if (cmu) {
    torso = "root";
    effectors = {"rradius", "lradius", "rfoot", "lfoot"};
  } else if (walker == Walker::kRodent) {
    torso = "torso";
    effectors = {"lower_arm_R", "lower_arm_L", "foot_R", "foot_L"};
  } else if (walker == Walker::kAnt) {
    torso = "torso";
    effectors = {"front_left_foot", "front_right_foot", "back_right_foot",
                 "back_left_foot"};
    for (const char* name : {"front_left_leg_geom", "front_right_leg_geom"}) {
      Set(Find(model, "geom", name), "rgba",
          red ? Numbers({.8, .1, .1, 1}) : Numbers({.1, .1, .8, 1}));
    }
  } else {
    torso = "head_body";
    effectors = {"head_body"};
    const std::array<double, 4> marker{red ? .8 : .1, .1, red ? .1 : .8, 1};
    std::vector<unsigned char> pixels;
    unsigned width = 0;
    unsigned height = 0;
    const std::string digits =
        asset_path_ + "/locomotion/soccer/assets/boxhead/digits/" +
        (player < 10 ? "0" : "") + std::to_string(player) + ".png";
    if (lodepng::decode(pixels, width, height, digits) != 0u) {
      throw std::runtime_error("Cannot decode BoxHead digit asset");
    }
    for (std::size_t i = 0; i < pixels.size(); i += 4) {
      const double alpha = pixels[i + 3] / 255.;
      const double out_alpha = alpha + (1 - alpha);
      for (int channel = 0; channel < 3; ++channel) {
        pixels[i + channel] = 255 *
                              ((pixels[i + channel] / 255.) * alpha +
                               marker[channel] * (1 - alpha)) /
                              out_alpha;
      }
      pixels[i + 3] = 255 * out_alpha;
    }
    const auto filename = "boxhead_" + std::string(red ? "red_" : "blue_") +
                          std::to_string(player) + ".png";
    if (lodepng::encode(virtual_files_[filename], pixels, width, height) !=
        0u) {
      throw std::runtime_error("Cannot encode BoxHead digit asset");
    }
    auto head_texture = Child(model, "asset").append_child("texture");
    Set(head_texture, "name", "head_texture");
    Set(head_texture, "type", "2d");
    Set(head_texture, "file", filename);
    auto head_material = Child(model, "asset").append_child("material");
    Set(head_material, "name", "head_material");
    Set(head_material, "texture", "head_texture");
    for (const char* name : {"head", "top_down_cam_box"}) {
      auto geom = Find(model, "geom", name);
      Set(geom, "material", "head_material");
      geom.remove_attribute("rgba");
    }
    auto texture = Child(model, "asset").append_child("texture");
    Set(texture, "name", "ball_body");
    Set(texture, "type", "cube");
    Set(texture, "builtin", "checker");
    Set(texture, "rgb1", Numbers(marker.data(), 3));
    Set(texture, "rgb2", {.8, .8, .8});
    Set(texture, "width", 100);
    Set(texture, "height", 100);
    auto material = Child(model, "asset").append_child("material");
    Set(material, "name", "ball_body");
    Set(material, "texture", "ball_body");
    Set(Find(model, "geom", "shell"), "material", "ball_body");
    for (const char* name : {"arm_l", "arm_r", "eye_l", "eye_r"}) {
      Set(Find(model, "geom", name), "rgba", Numbers(marker.data(), 4));
    }
    for (const char* name : {"camera_pitch", "camera_yaw"}) {
      auto joint = Find(model, "joint", name);
      joint.parent().remove_child(joint);
      for (auto actuator : model.child("actuator").children()) {
        if (std::string(actuator.attribute("name").value()) == name) {
          model.child("actuator").remove_child(actuator);
          break;
        }
      }
    }
    for (auto actuator : model.child("actuator").children()) {
      if (std::string(actuator.attribute("name").value()) == "roll") {
        Set(actuator, "gear", -60);
      }
    }
  }
  auto sensors = Child(model, "sensor");
  if (walker == Walker::kAnt) {
    for (const auto& effector : effectors) {
      AddFrameSensor(sensors, "framepos", effector, effector + "_appendage",
                     torso);
    }
    for (const auto& entry : model.select_nodes(".//worldbody//body")) {
      const std::string name = entry.node().attribute("name").value();
      AddFrameSensor(sensors, "framepos", name, name + "_ego_body_pos", torso);
    }
    for (const auto& entry : model.select_nodes(".//worldbody//body")) {
      const std::string name = entry.node().attribute("name").value();
      AddFrameSensor(sensors, "framequat", name, name + "_ego_body_quat",
                     torso);
    }
  }
  for (const auto& effector : effectors) {
    AddFrameSensor(sensors, "framepos", effector, effector + "_end_effector",
                   torso);
  }
  Attach(model, prefix,
         walker == Walker::kBoxhead ? RootJoint::kSlides : RootJoint::kFree);
}

std::string Scene::Xml() const {
  pugi::xml_document copy;
  copy.reset(document_);
  auto model = copy.child("mujoco");
  // PyMJCF gives the arena its own default class. Attached entities must not
  // inherit arena-only overrides (for example the maze's white wall color).
  auto defaults = Child(model, "default");
  auto arena_defaults = defaults.prepend_child("default");
  Set(arena_defaults, "class", "/");
  for (auto node = arena_defaults.next_sibling(); node != nullptr;) {
    auto next = node.next_sibling();
    if (std::string_view(node.name()) != "default") {
      arena_defaults.append_move(node);
    }
    node = next;
  }
  for (auto node : model.child("worldbody").children()) {
    const auto tag = std::string_view(node.name());
    if (tag == "body" && !node.attribute("childclass")) {
      Set(node, "childclass", "/");
    } else if ((tag == "geom" || tag == "site" || tag == "camera" ||
                tag == "light") &&
               !node.attribute("class")) {
      Set(node, "class", "/");
    }
  }
  for (auto node : model.child("asset").children()) {
    if (std::string_view(node.name()) == "material" &&
        !node.attribute("class")) {
      Set(node, "class", "/");
    }
  }
  for (auto node : model.child("envpool_attached_asset").children()) {
    Child(model, "asset").append_copy(node);
  }
  model.remove_child("envpool_attached_asset");
  std::ostringstream output;
  copy.save(output, "", pugi::format_raw | pugi::format_no_declaration);
  return output.str();
}

mjModel* Scene::Compile() const {
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  const auto xml = Xml();
  mj_addBufferVFS(&vfs, "locomotion.xml", xml.data(), xml.size());
  for (const auto& [name, bytes] : virtual_files_) {
    mj_addBufferVFS(&vfs, name.c_str(), bytes.data(), bytes.size());
  }
  std::array<char, 2048> error{};
  mjModel* model =
      mj_loadXML("locomotion.xml", &vfs, error.data(), error.size());
  mj_deleteVFS(&vfs);
  if (model == nullptr) {
    throw std::runtime_error(error.data());
  }
  return model;
}

}  // namespace mujoco_locomotion
