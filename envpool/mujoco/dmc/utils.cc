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
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

void XMLRemoveNamedElement(pugi::xml_document* doc, const std::string& tag,
                           const std::string& name) {
  std::string xpath = "//" + tag + "[@name='" + name + "']";
  pugi::xml_node node = doc->select_node(xpath.c_str()).node();
  if (!node.empty()) {
    node.parent().remove_child(node);
  }
}

std::string XMLMakeDog(const std::string& content,
                       const std::string& task_name) {
  pugi::xml_document doc;
  doc.load_string(content.c_str());

  double floor_size = 10.0;
  bool remove_ball = true;
  if (task_name == "stand" || task_name == "walk") {
    floor_size = 15.0;
  } else if (task_name == "trot") {
    floor_size = 45.0;
  } else if (task_name == "run") {
    floor_size = 135.0;
  } else if (task_name == "fetch") {
    remove_ball = false;
  } else {
    throw std::runtime_error("Unknown task_name " + task_name +
                             " for dmc dog.");
  }

  pugi::xml_node floor = doc.select_node("//geom[@name='floor']").node();
  floor.attribute("size").set_value(
      (std::to_string(floor_size) + " " + std::to_string(floor_size) + " .1")
          .c_str());

  if (remove_ball) {
    XMLRemoveNamedElement(&doc, "body", "ball");
    XMLRemoveNamedElement(&doc, "geom", "target");
    XMLRemoveNamedElement(&doc, "camera", "ball");
    XMLRemoveNamedElement(&doc, "camera", "head");
    for (const auto& wall : {"px", "nx", "py", "ny"}) {
      XMLRemoveNamedElement(&doc, "geom", std::string("wall_") + wall);
    }
  }

  XMLStringWriter writer;
  doc.print(writer);
  return writer.result;
}

std::string XMLMakeQuadruped(const std::string& content,
                             const std::string& task_name) {
  pugi::xml_document doc;
  doc.load_string(content.c_str());

  bool terrain = false;
  bool rangefinders = false;
  bool walls_and_ball = false;
  double floor_size = -1.0;
  if (task_name == "walk") {
    floor_size = 10.0;
  } else if (task_name == "run") {
    floor_size = 100.0;
  } else if (task_name == "escape") {
    floor_size = 40.0;
    terrain = true;
    rangefinders = true;
  } else if (task_name == "fetch") {
    walls_and_ball = true;
  } else {
    throw std::runtime_error("Unknown task_name " + task_name +
                             " for dmc quadruped.");
  }

  if (floor_size > 0) {
    pugi::xml_node floor = doc.select_node("//geom[@name='floor']").node();
    floor.attribute("size").set_value(
        (std::to_string(floor_size) + " " + std::to_string(floor_size) + " .5")
            .c_str());
  }
  if (!walls_and_ball) {
    for (const auto& wall : {"wall_px", "wall_py", "wall_nx", "wall_ny"}) {
      XMLRemoveNamedElement(&doc, "geom", wall);
    }
    XMLRemoveNamedElement(&doc, "body", "ball");
    XMLRemoveNamedElement(&doc, "site", "target");
  }
  if (!terrain) {
    XMLRemoveNamedElement(&doc, "geom", "terrain");
  }
  if (!rangefinders) {
    pugi::xpath_node_set sensors = doc.select_nodes("//rangefinder");
    for (const pugi::xpath_node& sensor : sensors) {
      pugi::xml_node node = sensor.node();
      node.parent().remove_child(node);
    }
  }

  XMLStringWriter writer;
  doc.print(writer);
  return writer.result;
}

std::string XMLMakeStacker(const std::string& content, int n_boxes) {
  pugi::xml_document doc;
  doc.load_string(content.c_str());
  for (int box = n_boxes; box < 4; ++box) {
    XMLRemoveNamedElement(&doc, "body", "box" + std::to_string(box));
  }
  XMLStringWriter writer;
  doc.print(writer);
  return writer.result;
}

std::string XMLMakeLqr(const std::string& content, int n_bodies,
                       int n_actuators, std::mt19937* gen) {
  if (n_bodies < 1 || n_actuators < 1 || n_actuators > n_bodies) {
    throw std::runtime_error("Invalid lqr body/actuator count.");
  }

  pugi::xml_document doc;
  doc.load_string(content.c_str());
  pugi::xml_node root = doc.select_node("/mujoco").node();
  pugi::xml_node parent = doc.select_node("/mujoco/worldbody").node();
  pugi::xml_node actuator = root.append_child("actuator");
  pugi::xml_node tendon = root.append_child("tendon");
  RandUniform stiffness_dist(15.0, 25.0);

  for (int body_id = 0; body_id < n_bodies; ++body_id) {
    std::string id = std::to_string(body_id);
    pugi::xml_node body = parent.append_child("body");
    body.append_attribute("name") = ("body_" + id).c_str();
    body.append_attribute("pos") = body_id == 0 ? ".25 0 .1" : ".25 0 0";

    pugi::xml_node joint = body.append_child("joint");
    joint.append_attribute("name") = ("joint_" + id).c_str();
    joint.append_attribute("stiffness") = stiffness_dist(*gen);
    joint.append_attribute("damping") = "0";

    pugi::xml_node geom = body.append_child("geom");
    geom.append_attribute("name") = ("geom_" + id).c_str();

    pugi::xml_node site = body.append_child("site");
    site.append_attribute("name") = ("site_" + id).c_str();

    if (body_id < n_actuators) {
      pugi::xml_node motor = actuator.append_child("motor");
      motor.append_attribute("name") = ("motor_" + id).c_str();
      motor.append_attribute("joint") = ("joint_" + id).c_str();
    }

    if (body_id < n_bodies - 1) {
      pugi::xml_node spatial = tendon.append_child("spatial");
      spatial.append_attribute("name") = ("tendon_" + id).c_str();
      pugi::xml_node current_site = spatial.append_child("site");
      current_site.append_attribute("site") = ("site_" + id).c_str();
      pugi::xml_node child_site = spatial.append_child("site");
      child_site.append_attribute("site") =
          ("site_" + std::to_string(body_id + 1)).c_str();
    }

    parent = body;
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

std::string XMLMakeSwimmer(const std::string& content, int n_bodies) {
  pugi::xml_document doc;
  doc.load_string(content.c_str());

  pugi::xml_node mjc = doc.select_node("/mujoco").node();
  pugi::xml_node actuator = mjc.append_child("actuator");
  pugi::xml_node sensor = mjc.append_child("sensor");
  pugi::xml_node body = doc.select_node("//worldbody/body").node();
  std::string joint_range = std::to_string(360.0 / n_bodies);
  joint_range = "-" + joint_range + " " + joint_range;

  for (int i = 0; i < n_bodies - 1; ++i) {
    std::string id = std::to_string(i);
    // motor
    pugi::xml_node motor = actuator.append_child("motor");
    motor.append_attribute("joint") = ("joint_" + id).c_str();
    motor.append_attribute("name") = ("motor_" + id).c_str();
    // velocimeter
    pugi::xml_node velocimeter = sensor.append_child("velocimeter");
    velocimeter.append_attribute("name") = ("velocimeter_" + id).c_str();
    velocimeter.append_attribute("site") = ("site_" + id).c_str();
    // gyro
    pugi::xml_node gyro = sensor.append_child("gyro");
    gyro.append_attribute("name") = ("gyro_" + id).c_str();
    gyro.append_attribute("site") = ("site_" + id).c_str();
    // body
    pugi::xml_node child = body.append_child("body");
    child.append_attribute("name") = ("segment_" + id).c_str();
    child.append_attribute("pos") = "0 .1 0";
    body = child;
    pugi::xml_node geom = body.append_child("geom");
    geom.append_attribute("class") = "visual";
    geom.append_attribute("name") = ("visual_" + id).c_str();
    geom = body.append_child("geom");
    geom.append_attribute("class") = "inertial";
    geom.append_attribute("name") = ("inertial_" + id).c_str();
    pugi::xml_node site = body.append_child("site");
    site.append_attribute("name") = ("site_" + id).c_str();
    pugi::xml_node joint = body.append_child("joint");
    joint.append_attribute("name") = ("joint_" + id).c_str();
    joint.append_attribute("range") = joint_range.c_str();
  }

  double scale = n_bodies / 6.0;
  pugi::xpath_node_set cameras = doc.select_nodes("//worldbody/body/camera");
  for (const pugi::xpath_node& c : cameras) {
    std::string mode = c.node().attribute("mode").value();
    if (mode != "trackcom") {
      continue;
    }
    std::istringstream in(c.node().attribute("pos").value());
    std::ostringstream out;
    for (int i = 0; i < 3; ++i) {
      double x;
      in >> x;
      if (i > 0) {
        out << " ";
      }
      out << x * scale;
    }
    c.node().attribute("pos").set_value(out.str().c_str());
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
