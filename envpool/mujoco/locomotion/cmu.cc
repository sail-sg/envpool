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

#include <array>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/mujoco/locomotion/scene.h"

namespace mujoco_locomotion {
namespace {

std::vector<double> Read(pugi::xml_node node, const char* attribute) {
  std::istringstream stream(node.attribute(attribute).value());
  std::vector<double> values;
  double value;
  while (stream >> value) {
    values.push_back(value);
  }
  return values;
}

void Scale(pugi::xml_node node, const char* attribute, double factor) {
  auto values = Read(node, attribute);
  if (values.empty()) {
    return;
  }
  for (double& value : values) {
    value *= factor;
  }
  Set(node, attribute, Numbers(values.data(), values.size()));
}

void Rescale(pugi::xml_node model) {
  for (auto entry : model.select_nodes(".//worldbody//*")) {
    auto node = entry.node();
    auto fromto = Read(node, "fromto");
    if (!fromto.empty()) {
      for (int i = 0; i < 3; ++i) {
        const double middle = 1.2 * .5 * (fromto[i + 3] + fromto[i]);
        const double half = 1.2 * .5 * (fromto[i + 3] - fromto[i]);
        fromto[i] = middle - half;
        fromto[i + 3] = middle + half;
      }
      Set(node, "fromto", Numbers(fromto.data(), 6));
    }
    Scale(node, "pos", 1.2);
    Scale(node, "size", 1.2);
  }
  model.child("compiler").remove_attribute("coordinate");
  std::ostringstream xml;
  model.print(xml, "", pugi::format_raw);
  const auto contents = xml.str();
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "cmu.xml", contents.data(), contents.size());
  std::array<char, 1024> error{};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> physics(
      mj_loadXML("cmu.xml", &vfs, error.data(), sizeof(error)), mj_deleteModel);
  mj_deleteVFS(&vfs);
  if (!physics) {
    throw std::runtime_error(error.data());
  }
  const int root = mj_name2id(physics.get(), mjOBJ_BODY, "root");
  const double mass_factor = 70 / physics->body_subtreemass[root];
  for (auto entry : Find(model, "body", "root").select_nodes(".//inertial")) {
    Scale(entry.node(), "mass", mass_factor);
  }
  for (auto entry : Find(model, "body", "root").select_nodes(".//geom")) {
    auto geom = entry.node();
    if (geom.attribute("mass") != nullptr) {
      Scale(geom, "mass", mass_factor);
    } else {
      Set(geom, "density",
          geom.attribute("density").as_double(1000) * mass_factor);
    }
  }
}

}  // namespace

void Scene::CmuVisuals(pugi::xml_node model, Walker walker, int player,
                       bool red) {
  auto head = Find(model, "body", "head");
  auto face = head.append_child("geom");
  Set(face, "type", "capsule");
  Set(face, "name", "face");
  Set(face, "size", {.065, .014});
  Set(face, "pos", {.000341465, .048184, .01});
  Set(face, "quat", {.717887, .696142, -.00493334, 0});
  Set(face, "mass", 0);
  Set(face, "contype", 0);
  Set(face, "conaffinity", 0);
  auto face_body = head.append_child("body");
  Set(face_body, "name", "face");
  Set(face_body, "pos", {0, .039, Read(head, "pos")[1] - .02});
  auto nose = face_body.append_child("geom");
  Set(nose, "type", "capsule");
  Set(nose, "name", "nose");
  Set(nose, "size", {Read(Find(model, "geom", "head"), "size")[0] / 4.75, .01});
  Set(nose, "pos", {0, 0, 0});
  Set(nose, "quat", {1, .7, 0, 0});
  Set(nose, "mass", 0);
  Set(nose, "contype", 0);
  Set(nose, "conaffinity", 0);
  Set(nose, "group", 1);
  if (walker == Walker::kCmu2020) {
    Rescale(model);
    return;
  }

  for (const char* hand : {"lhand", "rhand"}) {
    for (auto entry : Find(model, "body", hand).select_nodes(".//geom")) {
      auto geom = entry.node();
      const std::string name = geom.attribute("name").value();
      Set(geom, "rgba", {0, 0, 0, 0});
      auto visual = geom.parent().append_child("geom");
      Set(visual, "name", name + "_visual");
      for (const char* attr : {"type", "quat", "pos", "size"}) {
        if (geom.attribute(attr) != nullptr) {
          Set(visual, attr, geom.attribute(attr).value());
        }
      }
      Scale(visual, "size", name == hand ? 1.3 : 1.5);
      Scale(visual, "pos", 1.5);
      Set(visual, "mass", 0);
      Set(visual, "contype", 0);
      Set(visual, "conaffinity", 0);
    }
  }
  auto light = Find(model, "light", "tracking_light");
  light.parent().remove_child(light);
  const auto path = asset_path_ + "/locomotion/soccer/assets/humanoid/";
  auto texture = Child(model, "asset").append_child("texture");
  Set(texture, "name", "skin");
  Set(texture, "type", "2d");
  Set(texture, "file",
      path + (red ? "R_" : "B_") + (player + 1 < 10 ? "0" : "") +
          std::to_string(player + 1) + ".png");
  auto material = Child(model, "asset").append_child("material");
  Set(material, "name", "skin");
  Set(material, "texture", "skin");
  auto skin = Child(model, "asset").append_child("skin");
  Set(skin, "name", "skin");
  Set(skin, "file", path + "jersey.skn");
  Set(skin, "material", "skin");
  for (const char* name :
       {"lhipjoint", "rhipjoint", "lfemur", "lowerback", "upperback",
        "rclavicle", "lclavicle", "thorax", "lhumerus", "root_geom",
        "lowerneck", "rhumerus", "rfemur"}) {
    Set(Find(model, "geom", name), "rgba", {0, 0, 0, 0});
  }
  const double neck = .066 - .0452401;
  const double arm = .20 - .138421;
  const double leg = .384 - .202473;
  const std::array bodies{"lowerneck", "lhumerus", "rhumerus", "lfemur",
                          "rfemur"};
  const std::array names{"halfneck", "lelbow", "relbow", "lknee", "rknee"};
  const std::array<std::array<double, 2>, 5> sizes{{{.05, .02279225 - neck},
                                                    {.035, .1245789 - arm},
                                                    {.035, .1245789 - arm},
                                                    {.055, .1822257 - leg},
                                                    {.055, .1822257 - leg}}};
  const std::array<std::array<double, 3>, 5> positions{
      {{-.00165071, .0452401 + neck, .00534359},
       {0, -.138421 - arm, 0},
       {0, -.138421 - arm, 0},
       {-5.0684e-8, -.202473 - leg, 0},
       {-5.0684e-8, -.202473 - leg, 0}}};
  const std::array<std::array<double, 4>, 5> quats{
      {{.66437, .746906, .027253, 0},
       {.612372, -.612372, .353553, .353553},
       {.612372, -.612372, -.353553, -.353553},
       {.696364, -.696364, -.122788, -.122788},
       {.696364, -.696364, .122788, .122788}}};
  for (int i = 0; i < 5; ++i) {
    auto geom = Find(model, "body", bodies[i]).append_child("geom");
    Set(geom, "name", names[i]);
    Set(geom, "size", Numbers(sizes[i].data(), 2));
    Set(geom, "pos", Numbers(positions[i].data(), 3));
    Set(geom, "quat", Numbers(quats[i].data(), 4));
    Set(geom, "mass", 0);
    Set(geom, "contype", 0);
    Set(geom, "conaffinity", 0);
    Set(geom, "rgba", {.7, .5, .3, 1});
  }
}

}  // namespace mujoco_locomotion
