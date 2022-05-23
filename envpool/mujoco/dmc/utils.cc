// Copyright 2022 Garena Online Private Limited
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

#include "envpool/mujoco/dmc/utils.h"

#include <cmath>
#include <fstream>
#include <sstream>

#include "pugixml.hpp"

namespace mujoco_dmc {

std::string GetFileContent(const std::string& base_path,
                           const std::string& asset_name) {
  // hardcode path here :(
  std::string filename = base_path + "/mujoco/assets_dmc/" + asset_name;
  std::ifstream ifs(filename);
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

class XMLStringWriter : public pugi::xml_writer {
 public:
  std::string result;
  void write(const void* data, size_t size) override {
    result.append(static_cast<const char*>(data), size);
  }
};

std::string XMLRemoveByBodyName(const std::string& content,
                                const std::vector<std::string>& body_names) {
  pugi::xml_document doc;
  doc.load_string(content.c_str());
  for (const auto& name : body_names) {
    std::string xpath = "//body[@name='" + name + "']";
    pugi::xml_node node = doc.select_node(xpath.c_str()).node();
    auto parent = node.parent();
    parent.remove_child(node);
  }
  XMLStringWriter writer;
  doc.print(writer);
  return writer.result;
}

std::string XMLAddPoles(const std::string& content, int n_poles) {
  pugi::xml_document doc;
  doc.load_string(content.c_str());

  pugi::xml_node body = doc.select_node("//worldbody/body/body").node();
  for (int i = 2; i <= n_poles; ++i) {
    pugi::xml_node new_pole = body.append_child("body");
    new_pole.append_attribute("childclass") = "pole";
    new_pole.append_attribute("name") = ("pole_" + std::to_string(i)).c_str();
    new_pole.append_attribute("pos") = "0 0 1";
    pugi::xml_node joint = new_pole.append_child("joint");
    joint.append_attribute("name") = ("hinge_" + std::to_string(i)).c_str();
    pugi::xml_node geom = new_pole.append_child("geom");
    geom.append_attribute("name") = ("pole_" + std::to_string(i)).c_str();
    body = new_pole;
  }

  pugi::xml_node floor = doc.select_node("//worldbody/geom").node();
  floor.attribute("pos").set_value(
      ("0 0 " + std::to_string(1 - n_poles - 0.05)).c_str());
  pugi::xpath_node_set cameras = doc.select_nodes("//worldbody/camera");
  for (const pugi::xpath_node& c : cameras) {
    std::string name = c.node().attribute("name").value();
    if (name == "fixed") {
      c.node().attribute("pos").set_value(
          ("0 " + std::to_string(-1 - 2 * n_poles) + " 1").c_str());
    } else if (name == "lookatcart") {
      c.node().attribute("pos").set_value(
          ("0 " + std::to_string(-2 * n_poles) + " 2").c_str());
    }
  }
  XMLStringWriter writer;
  doc.print(writer);
  return writer.result;
}
// std::vector<std::string> splitString(std::string str, char splitter) {
//   std::vector<std::string> result;
//   std::string current = "";
//   for (int i = 0; i < str.size(); i++) {
//     if (str[i] == splitter) {
//       if (current != "") {
//         result.push_back(current);
//         current = "";
//       }
//       continue;
//     }
//     current += str[i];
//   }
//   if (current.size() != 0) result.push_back(current);
//   return result;
// }
std::string XMLMakeSwimmer(const std::string& content, int n_joints) {
  if (n_joints < 3) {
    throw std::runtime_error(
        "At least 3 bodies required for swimmer. Received " +
        std::to_string(n_joints));
  }
  pugi::xml_document doc;
  doc.load_string(content.c_str());
  pugi::xml_node mjcf = doc.document_element();
  pugi::xml_node head_body = doc.select_node("//worldbody/body").node();
  pugi::xml_node actuator = mjcf.append_child("actuator");
  pugi::xml_node sensor = mjcf.append_child("sensor");
  pugi::xml_node parent = head_body;
  for (int i = 0; i <= n_joints - 1; ++i) {
    pugi::xml_node body = parent.append_child("body");
    body.append_attribute("name") = ("segment_" + std::to_string(i)).c_str();
    body.attribute("pos").set_value(("0 .1 0").c_str());
    pugi::xml_node geom0 = body.append_child("geom");
    geom0.append_attribute("class") = "visual";
    geom0.append_attribute("name") = ("visual_" + std::to_string(i)).c_str();
    pugi::xml_node geom1 = body.append_child("geom");
    geom1.append_attribute("class") = "inertial";
    geom1.append_attribute("name") = ("inertial_" + std::to_string(i)).c_str();
    pugi::xml_node site = body.append_child("site");
    site.append_attribute("name") = ("site_" + std::to_string(i)).c_str();
    pugi::xml_node joint = body.append_child("joint");
    float joint_limit = 360.0 / n_joints;
    joint.append_attribute("name") = ("joint_" + std::to_string(i)).c_str();
    joint.append_attribute("range") =
        (std::to_string(-joint_limit)
             .substr(0, std::to_string(i).find(".") + 1 + 1) +
         " " +
         std::to_string(joint_limit)
             .substr(0, std::to_string(i).find(".") + 1 + 1))
            .c_str();
    pugi::xml_node motor = actuator.append_child("motor");
    motor.append_attribute("joint") = ("joint_" + std::to_string(i)).c_str();
    motor.append_attribute("name") = ("motor_" + std::to_string(i)).c_str();
    pugi::xml_node velocimeter = sensor.append_child("velocimeter");
    velocimeter.append_attribute("name") =
        ("velocimeter_" + std::to_string(i)).c_str();
    velocimeter.append_attribute("site") =
        ("site_" + std::to_string(i)).c_str();
    pugi::xml_node gyro = sensor.append_child("gyro");
    gyro.append_attribute("name") = ("gyro_" + std::to_string(i)).c_str();
    gyro.append_attribute("site") = ("site_" + std::to_string(i)).c_str();
    parent = body;
  }
  // Move tracking cameras further away from the swimmer according to its
  // length. pugi::xml_node floor = doc.select_node("//worldbody/geom").node();
  // floor.attribute("pos").set_value(
  //     ("0 0 " + std::to_string(1 - n_joints - 0.05)).c_str());
  pugi::xpath_node_set cameras = doc.select_nodes("//worldbody/body/camera");
  float scale = n_joints / 6.0;
  for (const pugi::xpath_node& c : cameras) {
    std::string mode = c.node().attribute("mode").value();
    if (mode == "trackcom") {
      std::string old_pos = c.node().attribute("pos").value();
      std::vector<std::string> split_old_pos;
      std::string current = "";
      for (int i = 0; i < old_pos.size(); i++) {
        if (old_pos[i] == " ") {
          if (current != "") {
            split_old_pos.push_back(current);
            current = "";
          }
          continue;
        }
        current += old_pos[i];
      }
      if (current.size() != 0) {
        split_old_pos.push_back(current);
      }
      c.node().attribute("pos").set_value(
          (std::to_string(std::stof(split_old_pos[0]) * scale)
               .substr(0, std::to_string(i).find(".") + 1 + 1) +
           " " +
           std::to_string(std::stof(split_old_pos[1]) * scale)
               .substr(0, std::to_string(i).find(".") + 1 + 1) +
           " " +
           std::to_string(std::stof(split_old_pos[2]) * scale)
               .substr(0, std::to_string(i).find(".") + 1 + 1))
              .c_str());
    }
  }
  XMLStringWriter writer;
  doc.print(writer);
  return writer.result;
}

int GetQposId(mjModel* model, const std::string& name) {
  return model->jnt_qposadr[mj_name2id(model, mjOBJ_JOINT, name.c_str())];
}

int GetQvelId(mjModel* model, const std::string& name) {
  return model->jnt_dofadr[mj_name2id(model, mjOBJ_JOINT, name.c_str())];
}

int GetSensorId(mjModel* model, const std::string& name) {
  return model->sensor_adr[mj_name2id(model, mjOBJ_SENSOR, name.c_str())];
}

// rewards
double RewardTolerance(double x, double bound_min, double bound_max,
                       double margin, double value_at_margin,
                       SigmoidType sigmoid_type) {
  if (bound_min <= x && x <= bound_max) {
    return 1.0;
  }
  if (margin <= 0.0) {
    return 0.0;
  }
  x = (x < bound_min ? bound_min - x : x - bound_max) / margin;
  if (sigmoid_type == SigmoidType::kGaussian) {
    // scale = np.sqrt(-2 * np.log(value_at_1))
    // return np.exp(-0.5 * (x*scale)**2)
    double scaled_x = std::sqrt(-2 * std::log(value_at_margin)) * x;
    return std::exp(-0.5 * scaled_x * scaled_x);
  }
  if (sigmoid_type == SigmoidType::kHyperbolic) {
    // scale = np.arccosh(1/value_at_1)
    // return 1 / np.cosh(x*scale)
    double scaled_x = std::acosh(1 / value_at_margin) * x;
    return 1 / std::cosh(scaled_x);
  }
  if (sigmoid_type == SigmoidType::kLongTail) {
    // scale = np.sqrt(1/value_at_1 - 1)
    // return 1 / ((x*scale)**2 + 1)
    double scaled_x = std::sqrt(1 / value_at_margin - 1) * x;
    return 1 / (scaled_x * scaled_x + 1);
  }
  if (sigmoid_type == SigmoidType::kReciprocal) {
    // scale = 1/value_at_1 - 1
    // return 1 / (abs(x)*scale + 1)
    double scale = 1 / value_at_margin - 1;
    return 1 / (std::abs(x) * scale + 1);
  }
  if (sigmoid_type == SigmoidType::kCosine) {
    // scale = np.arccos(2*value_at_1 - 1) / np.pi
    // scaled_x = x*scale
    // with warnings.catch_warnings():
    //   warnings.filterwarnings(
    //       action='ignore', message='invalid value encountered in cos')
    //   cos_pi_scaled_x = np.cos(np.pi*scaled_x)
    // return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x)/2, 0.0)
    const double pi = std::acos(-1);
    double scaled_x = std::acos(2 * value_at_margin - 1) / pi * x;
    return std::abs(scaled_x) < 1 ? (1 + std::cos(pi * scaled_x)) / 2 : 0.0;
  }
  if (sigmoid_type == SigmoidType::kLinear) {
    // scale = 1-value_at_1
    // scaled_x = x*scale
    // return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
    double scaled_x = (1 - value_at_margin) * x;
    return std::abs(scaled_x) < 1 ? 1 - scaled_x : 0.0;
  }
  if (sigmoid_type == SigmoidType::kQuadratic) {
    // scale = np.sqrt(1-value_at_1)
    // scaled_x = x*scale
    // return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)
    double scaled_x = std::sqrt(1 - value_at_margin) * x;
    return std::abs(scaled_x) < 1 ? 1 - scaled_x * scaled_x : 0.0;
  }
  if (sigmoid_type == SigmoidType::kTanhSquared) {
    // scale = np.arctanh(np.sqrt(1-value_at_1))
    // return 1 - np.tanh(x*scale)**2
    double scaled_x = std::atanh(std::sqrt(1 - value_at_margin)) * x;
    return 1 - std::tanh(scaled_x) * std::tanh(scaled_x);
  }
  throw std::runtime_error("Unknown sigmoid type for RewardTolerance.");
}

}  // namespace mujoco_dmc
