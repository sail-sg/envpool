// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/mujoco/mjlab/simulation.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mjlab {

float Param(const Json& object, const std::string& name, float fallback) {
  const auto* value = object.as_object().if_contains(name);
  return value == nullptr || value->is_null() ? fallback : Number(*value);
}
std::string Name(const Json& object, const std::string& name,
                 const std::string& fallback) {
  const auto* value = object.as_object().if_contains(name);
  return value == nullptr || value->is_null() ? fallback : String(*value);
}
bool Flag(const Json& object, const std::string& name, bool fallback) {
  const auto* value = object.as_object().if_contains(name);
  return value == nullptr || value->is_null() ? fallback : value->as_bool();
}
std::string Function(const Json& term) {
  const auto name = String(term.at("func").at("callable"));
  return name.substr(name.find_last_of('.') + 1);
}
Json ToJson(const std::vector<float>& values) {
  boost::json::array result;
  for (auto v : values) {
    result.push_back(v);
  }
  return result;
}

std::vector<float> Task::Observation(const std::string& fn, const Json&) {
  throw std::invalid_argument("unimplemented MJLab observation: " + fn);
}
float Task::Reward(const std::string& fn, const Json&, const Json&) {
  throw std::invalid_argument("unimplemented MJLab reward: " + fn);
}
bool Task::Terminated(const std::string& fn, const Json&) {
  throw std::invalid_argument("unimplemented MJLab termination: " + fn);
}
Json Task::State() const {
  return boost::json::object{{"command", ToJson(command)},
                             {"time_left", time_left},
                             {"counter", counter}};
}

Simulation::Simulation(const std::string& asset_path, uint32_t seed,
                       int max_steps, const std::string& motion_file,
                       int env_id, int num_envs)
    : physics(asset_path),
      cfg(physics.Metadata()),
      random(seed),
      kEnvId(env_id),
      kNumEnvs(num_envs),
      max_steps(std::min(
          max_steps, static_cast<int>(Number(cfg.at("max_episode_steps"))))),
      decimation(Number(cfg.at("decimation"))),
      physics_dt(Number(cfg.at("physics_dt"))),
      step_dt(Number(cfg.at("step_dt"))) {
  for (const auto& entry : cfg.at("entities").as_object()) {
    const auto& value = entry.value();
    const auto& index = value.at("indexing");
    Entity entity;
    entity.name = std::string(entry.key());
    entity.root = Number(index.at("root_body_id"));
    if (!index.at("mocap_id").is_null()) {
      entity.mocap = Number(index.at("mocap_id"));
    }
    entity.bodies = Indices(index.at("body_ids"));
    entity.geoms = Indices(index.at("geom_ids"));
    entity.sites = Indices(index.at("site_ids"));
    entity.joints = Indices(index.at("joint_ids"));
    entity.controls = Indices(index.at("ctrl_ids"));
    entity.qadr = Indices(index.at("joint_q_adr"));
    entity.vadr = Indices(index.at("joint_v_adr"));
    entity.freeq = Indices(index.at("free_joint_q_adr"));
    entity.freev = Indices(index.at("free_joint_v_adr"));
    for (auto item :
         {std::make_pair("default_root_state", &entity.root_default),
          {"default_joint_pos", &entity.joint_default},
          {"default_joint_vel", &entity.velocity_default},
          {"soft_joint_pos_limits", &entity.limits}}) {
      if (!value.at(item.first).is_null()) {
        *item.second = Floats(value.at(item.first));
      }
    }
    entity.bias.resize(entity.qadr.size());
    entities.emplace(entity.name, std::move(entity));
  }
  const int action_size = Number(cfg.at("action_size"));
  action.resize(action_size);
  previous_action.resize(action_size);
  previous_previous_action.resize(action_size);
  const auto& action_cfg = cfg.at("action").as_object().begin()->value();
  action_entity = String(action_cfg.at("cfg").at("entity_name"));
  action_scale = Floats(action_cfg.at("state").at("_scale"));
  action_offset = Floats(action_cfg.at("state").at("_offset"));
  if (action_scale.size() == 1) {
    action_scale.resize(action_size, action_scale[0]);
  }
  if (action_offset.size() == 1) {
    action_offset.resize(action_size, action_offset[0]);
  }

  auto* model = physics.Model();
  action_joints = Indices(action_cfg.at("state").at("_target_ids"));
  position_action = Flag(action_cfg.at("cfg"), "use_default_offset");
  const auto& controlled = entities.at(action_entity);
  for (int joint : action_joints) {
    int control = -1;
    for (int id : controlled.controls) {
      if (model->actuator_trnid[id * 2] == controlled.joints[joint]) {
        if (control != -1) {
          throw std::invalid_argument("ambiguous MJLab joint actuator");
        }
        control = id;
      }
    }
    if (control == -1) {
      throw std::invalid_argument("missing MJLab joint actuator");
    }
    action_controls.push_back(control);
  }
  for (const auto& entry : cfg.at("sensors").as_object()) {
    const auto& value = entry.value();
    const auto& sensor_cfg = value.at("cfg");
    const auto name = std::string(entry.key());
    const auto type = String(value.at("type"));
    if (type == "ContactSensor") {
      Contact contact;
      contact.history_length = Param(sensor_cfg, "history_length");
      contact.track_air = Flag(sensor_cfg, "track_air_time");
      if (Param(sensor_cfg, "num_slots", 1) != 1 ||
          (Flag(sensor_cfg, "global_frame") &&
           Name(sensor_cfg, "reduce") != "netforce")) {
        throw std::invalid_argument(
            "unexpected pinned MJLab contact configuration");
      }
      for (const auto& slot : value.at("state").at("slots").as_array()) {
        const auto sensor_name = String(slot.at("sensor_name"));
        const int id = mj_name2id(model, mjOBJ_SENSOR, sensor_name.c_str());
        if (id < 0) {
          throw std::runtime_error("missing contact sensor: " + sensor_name);
        }
        const auto field = String(slot.at("field_name"));
        contact.slots.push_back(
            {model->sensor_adr[id], model->sensor_dim[id], field});
        if (field == "found") {
          contact.found.push_back(0);
        } else if (field == "force") {
          contact.force.resize(contact.force.size() + 3);
        } else {
          throw std::invalid_argument("unexpected pinned contact field: " +
                                      field);
        }
      }
      contact.history.resize(contact.force.size() * contact.history_length);
      for (auto* values : {&contact.air, &contact.last_air, &contact.contact,
                           &contact.last_contact}) {
        values->resize(contact.found.size());
      }
      contacts.emplace(name, std::move(contact));
    } else if (type == "RayCastSensor" || type == "TerrainHeightSensor") {
      RaySensor sensor;
      const auto& state = value.at("state");
      for (const auto& frame : state.at("frames").as_array()) {
        sensor.frames.push_back(
            {String(frame.at(0)), static_cast<int>(Number(frame.at(1)))});
      }
      sensor.offsets = Floats(state.at("_local_offsets"));
      sensor.directions = Floats(state.at("_local_directions"));
      const auto& pattern = sensor_cfg.at("pattern");
      if (pattern.as_object().contains("resolution")) {
        sensor.GenerateGrid(pattern);
      }
      sensor.max_distance = Param(sensor_cfg, "max_distance");
      sensor.alignment = Name(sensor_cfg, "ray_alignment");
      sensor.terrain_height = type == "TerrainHeightSensor";
      rays.emplace(name, std::move(sensor));
    }
  }
  // The exported model is a template, not a frozen randomized episode.
  for (const auto& field : cfg.at("default_model_fields").as_object()) {
    const auto values = Floats(field.value());
    const auto name = "model." + std::string(field.key());
    if (!values.empty()) {
      physics.Set(name, values.data(), values.size() * sizeof(float));
    }
  }
  const auto& commands = cfg.at("command").as_object();
  if (commands.contains("motion")) {
    task = MakeTracking(this, motion_file);
  } else if (commands.contains("twist")) {
    task = MakeVelocity(this);
  } else if (commands.contains("lift_height")) {
    task = MakeManipulation(this);
  } else {
    task = std::make_unique<Task>(this);
  }
  Events("startup");
}

Entity& Simulation::Asset(const Json& p, const std::string& key) {
  const auto* cfg = p.as_object().if_contains(key);
  return entities.at(cfg == nullptr ? "robot" : String(cfg->at("name")));
}
const Entity& Simulation::Asset(const Json& p, const std::string& key) const {
  const auto* cfg = p.as_object().if_contains(key);
  return entities.at(cfg == nullptr ? "robot" : String(cfg->at("name")));
}

std::vector<int> Simulation::Select(const Entity& entity, const Json& params,
                                    const std::string& kind,
                                    const std::string& key) const {
  int count = 0;
  if (kind == "joint") {
    count = entity.qadr.size();
  } else if (kind == "body") {
    count = entity.bodies.size();
  } else if (kind == "site") {
    count = entity.sites.size();
  } else if (kind == "geom") {
    count = entity.geoms.size();
  } else {
    throw std::invalid_argument("unexpected MJLab selection: " + kind);
  }
  const auto* selector = params.as_object().if_contains(key);
  if (selector != nullptr && selector->at(kind + "_ids").is_array()) {
    return Indices(selector->at(kind + "_ids"));
  }
  std::vector<int> result(count);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

Vec3 Simulation::Position(int body) const {
  return Read<3>(physics.Get("data.xpos") + body * 3);
}
Quat Simulation::Orientation(int body) const {
  return Read<4>(physics.Get("data.xquat") + body * 4);
}
Vec3 Simulation::AngularVelocity(int body) const {
  return Read<3>(physics.Get("data.cvel") + body * 6);
}
Vec3 Simulation::LinearVelocity(int body, int root, const Vec3& pos) const {
  const auto cvel = Read<6>(physics.Get("data.cvel") + body * 6);
  const auto offset = Read<3>(physics.Get("data.subtree_com") + root * 3) - pos;
  const Vec3 linear{cvel[3], cvel[4], cvel[5]};
  return linear - Cross({cvel[0], cvel[1], cvel[2]}, offset);
}
Vec3 Simulation::SitePosition(int site) const {
  return Read<3>(physics.Get("data.site_xpos") + site * 3);
}
Quat Simulation::SiteOrientation(int site) const {
  return Quaternion(physics.Get("data.site_xmat") + site * 9);
}
Vec3 Simulation::SiteVelocity(int site, int root) const {
  return LinearVelocity(physics.Model()->site_bodyid[site], root,
                        SitePosition(site));
}
void Simulation::WritePose(const Entity& entity, const Vec3& pos,
                           const Quat& quat) {
  if (entity.freeq.empty()) {
    if (entity.mocap < 0) {
      throw std::invalid_argument("cannot move a fixed MJLab body");
    }
    std::copy(pos.begin(), pos.end(),
              physics.Get("data.mocap_pos") + 3 * entity.mocap);
    std::copy(quat.begin(), quat.end(),
              physics.Get("data.mocap_quat") + 4 * entity.mocap);
  } else {
    auto* qpos = physics.Get("data.qpos");
    for (int i = 0; i < 3; ++i) {
      qpos[entity.freeq[i]] = pos[i];
    }
    for (int i = 0; i < 4; ++i) {
      qpos[entity.freeq[i + 3]] = quat[i];
    }
  }
}
void Simulation::WriteVelocity(const Entity& entity, const Vec3& lin,
                               const Vec3& ang) {
  const auto* qpos = physics.Get("data.qpos");
  Quat quat;
  for (int i = 0; i < 4; ++i) {
    quat[i] = qpos[entity.freeq[i + 3]];
  }
  const auto local = RotateInverse(quat, ang);
  auto* qvel = physics.Get("data.qvel");
  for (int i = 0; i < 3; ++i) {
    qvel[entity.freev[i]] = lin[i];
    qvel[entity.freev[i + 3]] = local[i];
  }
}
std::vector<float> Simulation::JointPositions(const Entity& entity,
                                              bool biased) const {
  std::vector<float> result;
  for (std::size_t i = 0; i < entity.qadr.size(); ++i) {
    const float q = physics.Get("data.qpos")[entity.qadr[i]];
    result.push_back(biased ? q + entity.bias[i] : q);
  }
  return result;
}
std::vector<float> Simulation::JointVelocities(const Entity& entity) const {
  std::vector<float> result;
  result.reserve(entity.vadr.size());
  for (int i : entity.vadr) {
    result.push_back(physics.Get("data.qvel")[i]);
  }
  return result;
}
std::vector<float> Simulation::Sensor(const std::string& name) const {
  const auto* model = physics.Model();
  const int id = mj_name2id(model, mjOBJ_SENSOR, name.c_str());
  if (id < 0) {
    throw std::out_of_range("missing MJLab sensor: " + name);
  }
  const float* data = physics.Get("data.sensordata") + model->sensor_adr[id];
  return {data, data + model->sensor_dim[id]};
}
std::vector<float> Simulation::Heights(const std::string& name,
                                       float offset) const {
  const auto& sensor = rays.at(name);
  auto values = sensor.heights;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (sensor.hits[i]) {
      values[i] -= offset;
    }
  }
  return values;
}

float Simulation::Sample(const Json& range, bool tensor_bounds) {
  const double low = Number(range.at(0));
  const double high = Number(range.at(1));
  const float width = tensor_bounds
                          ? static_cast<float>(high) - static_cast<float>(low)
                          : static_cast<float>(high - low);
  return random.Unit() * width + static_cast<float>(low);
}
float Simulation::Uniform(const Json& range) {
  return random.Uniform(Number(range.at(0)), Number(range.at(1)));
}
std::vector<float> Simulation::SamplePose(const Json& ranges) {
  std::vector<float> result;
  for (const auto* key : {"x", "y", "z", "roll", "pitch", "yaw"}) {
    const auto* range = ranges.as_object().if_contains(key);
    result.push_back(range == nullptr ? random.Uniform(0, 0)
                                      : Sample(*range, true));
  }
  return result;
}

void Simulation::WriteActions(bool resetting) {
  auto* ctrl = physics.Get("data.ctrl");
  const auto& entity = entities.at(action_entity);
  for (std::size_t i = 0; i < action.size(); ++i) {
    float target = action[i] * action_scale[i] + action_offset[i];
    if (position_action) {
      target -= entity.bias[action_joints[i]];
    }
    ctrl[action_controls[i]] = resetting ? 0 : target;
  }
}

void Simulation::Reset() {
  Curriculum();
  physics.Run("reset");
  for (auto& entry : contacts) {
    auto& c = entry.second;
    for (auto* values : {&c.found, &c.force, &c.history, &c.air, &c.last_air,
                         &c.contact, &c.last_contact}) {
      std::fill(values->begin(), values->end(), 0);
    }
  }
  Events("reset");
  std::fill(action.begin(), action.end(), 0);
  std::fill(previous_action.begin(), previous_action.end(), 0);
  std::fill(previous_previous_action.begin(), previous_previous_action.end(),
            0);
  episode_rewards.clear();
  reward_terms.clear();
  task->Reset();
  for (const auto& entry : cfg.at("event").as_object()) {
    if (Name(entry.value(), "mode") == "interval") {
      event_timers[std::string(entry.key())] =
          Sample(entry.value().at("interval_range_s"));
    }
  }
  steps = 0;
  reward = 0;
  terminated = truncated = false;
  WriteActions(true);
  physics.Run("forward");
  task->Update(true);
  UpdateContacts(false);
  Sense();
  Observe();
  initialized = true;
}

void Simulation::Step(const float* input) {
  if (!initialized || terminated || truncated) {
    Reset();
    return;
  }
  previous_previous_action = previous_action;
  previous_action = action;
  std::copy_n(input, action.size(), action.begin());
  for (int i = 0; i < decimation; ++i) {
    WriteActions(false);
    physics.Run("step");
    UpdateContacts(true);
  }
  ++steps;
  ++total_steps;
  terminated = truncated = false;
  for (const auto& term : cfg.at("termination").as_object()) {
    const bool value = EvaluateTermination(term.value());
    if (Flag(term.value(), "time_out")) {
      truncated |= value;
    } else {
      terminated |= value;
    }
  }
  truncated |= steps >= max_steps;
  reward = 0;
  for (const auto& entry : cfg.at("reward").as_object()) {
    const auto& term = entry.value();
    float weight = Param(term, "weight");
    if (weight == 0) {
      continue;
    }
    float value = EvaluateReward(term);
    value = value * weight;
    if (Flag(cfg, "scale_rewards_by_dt")) {
      value = value * step_dt;
    }
    if (!std::isfinite(value)) {
      value = 0;
    }
    reward_terms[std::string(entry.key())] = value;
    episode_rewards[std::string(entry.key())] += value;
    reward += value;
  }
  Events("step");
  Events("interval");
  physics.Run("forward");
  task->Update(false);
  Sense();
  Observe();
}

void Simulation::Curriculum() {
  for (const auto& entry : cfg.at("curriculum").as_object()) {
    const auto& term = entry.value();
    if (Function(term) != "reward_curriculum") {
      continue;
    }
    const auto& params = term.at("params");
    for (const auto& stage : params.at("stages").as_array()) {
      if (total_steps >= Param(stage, "step")) {
        cfg.at("reward").at(String(params.at("reward_name"))).at("weight") =
            stage.at("weight");
      }
    }
  }
  task->Curriculum();
}

void Simulation::UpdateContacts(bool substep) {
  if (contacts.empty()) {
    return;
  }
  const float* data = physics.Get("data.sensordata");
  for (auto& entry : contacts) {
    auto& c = entry.second;
    std::size_t f = 0;
    std::size_t v = 0;
    for (const auto& slot : c.slots) {
      auto& values = slot.field == "found" ? c.found : c.force;
      auto& index = slot.field == "found" ? f : v;
      std::copy_n(data + slot.address, slot.size, values.begin() + index);
      index += slot.size;
    }
    if (!substep) {
      continue;
    }
    if (c.track_air) {
      for (std::size_t i = 0; i < c.found.size(); ++i) {
        if (c.found[i] > 0) {
          if (c.air[i] > 0) {
            c.last_air[i] = c.air[i] + physics_dt;
          }
          c.air[i] = 0;
          c.contact[i] += physics_dt;
        } else {
          if (c.contact[i] > 0) {
            c.last_contact[i] = c.contact[i] + physics_dt;
          }
          c.contact[i] = 0;
          c.air[i] += physics_dt;
        }
      }
    }
    for (std::size_t i = 0; i < c.force.size() / 3 && (c.history_length != 0);
         ++i) {
      auto start = c.history.begin() + i * c.history_length * 3;
      std::move_backward(start, start + (c.history_length - 1) * 3,
                         start + c.history_length * 3);
      std::copy_n(c.force.begin() + i * 3, 3, start);
    }
  }
}

float Simulation::EvaluateReward(const Json& term) {
  const auto fn = Function(term);
  const auto& p = term.at("params");
  if (fn == "self_collision_cost") {
    const auto& sensor = contacts.at(Name(p, "sensor_name"));
    if (sensor.history.empty()) {
      return std::accumulate(sensor.found.begin(), sensor.found.end(), 0.0F);
    }
    float count = 0;
    for (int h = 0; h < sensor.history_length; ++h) {
      bool hit = false;
      for (std::size_t f = 0; f < sensor.force.size() / 3; ++f) {
        hit |= Norm(Read<3>(sensor.history.data() +
                            (f * sensor.history_length + h) * 3)) >
               Param(p, "force_threshold", 10);
      }
      count += hit ? 1 : 0;
    }
    return count;
  }
  if (fn == "action_rate_l2") {
    std::vector<float> change(action.size());
    for (std::size_t i = 0; i < action.size(); ++i) {
      change[i] = action[i] - previous_action[i];
    }
    return SquaredNorm(change);
  }
  if (fn == "joint_pos_limits" || fn == "joint_velocity_hinge_penalty") {
    const auto& entity = Asset(p);
    const auto ids = Select(entity, p, "joint");
    const auto q = JointPositions(entity);
    const auto v = JointVelocities(entity);
    std::vector<float> penalties;
    for (auto i : ids) {
      if (fn == "joint_pos_limits") {
        penalties.push_back(std::max(entity.limits[i * 2] - q[i], 0.0F) +
                            std::max(q[i] - entity.limits[i * 2 + 1], 0.0F));
      } else {
        penalties.push_back(
            Square(std::max(std::abs(v[i]) - Param(p, "max_vel"), 0.0F)));
      }
    }
    return Sum(penalties);
  }
  if (fn == "cartpole_smooth_reward") {
    const auto& entity = entities.at("cartpole");
    const auto q = JointPositions(entity);
    const auto v = JointVelocities(entity);
    const float scale = std::sqrt(-2.0 * std::log(0.1));
    const float upright = (Cos(q[1]) + 1.0F) / 2.0F;
    const float centered =
        (1.0F + Exp(-0.5F * Square((q[0] / 2.0F) * scale))) / 2.0F;
    const float small_control =
        (4.0F +
         std::max(1.0F - Square(action[0] * static_cast<float>(std::sqrt(0.9))),
                  0.0F)) /
        5.0F;
    const float small_velocity =
        (1.0F + Exp(-0.5F * Square((v[1] / 5.0F) * scale))) / 2.0F;
    return ((upright * centered) * small_control) * small_velocity;
  }
  return task->Reward(fn, p, term);
}

bool Simulation::EvaluateTermination(const Json& term) {
  const auto fn = Function(term);
  const auto& p = term.at("params");
  if (fn == "time_out") {
    return steps >= max_steps;
  }
  if (fn == "bad_orientation") {
    const auto g = RotateInverse(Orientation(Asset(p).root), {0, 0, -1});
    return std::abs(Acos(std::clamp(-g[2], -1.0F, 1.0F))) >
           Param(p, "limit_angle");
  }
  if (fn == "illegal_contact") {
    const auto& contact = contacts.at(Name(p, "sensor_name"));
    if (contact.history.empty()) {
      return std::any_of(contact.found.begin(), contact.found.end(),
                         [](float v) { return v != 0; });
    }
    const auto& force = contact.history;
    for (std::size_t i = 0; i < force.size(); i += 3) {
      if (Norm(Read<3>(force.data() + i)) > Param(p, "force_threshold", 10)) {
        return true;
      }
    }
    return false;
  }
  return task->Terminated(fn, p);
}

std::vector<float> Simulation::EvaluateObservation(const Json& term) {
  const auto fn = Function(term);
  const auto& p = term.at("params");
  if (fn == "last_action") {
    return action;
  }
  if (fn == "generated_commands") {
    return task->command;
  }
  if (fn == "builtin_sensor") {
    return Sensor(Name(p, "sensor_name"));
  }
  if (fn == "joint_pos_rel" || fn == "joint_vel_rel" ||
      fn == "pole_angle_cos_sin") {
    const auto& entity = Asset(p);
    const auto values = fn == "joint_vel_rel"
                            ? JointVelocities(entity)
                            : JointPositions(entity, Flag(p, "biased"));
    std::vector<float> result;
    for (auto id : Select(entity, p, "joint")) {
      if (fn == "pole_angle_cos_sin") {
        result.push_back(Cos(values[id]));
        result.push_back(Sin(values[id]));
      } else {
        result.push_back(values[id] - (fn == "joint_vel_rel"
                                           ? entity.velocity_default[id]
                                           : entity.joint_default[id]));
      }
    }
    return result;
  }
  if (fn == "projected_gravity") {
    const auto value = RotateInverse(Orientation(Asset(p).root), {0, 0, -1});
    return {value.begin(), value.end()};
  }
  if (fn == "height_scan") {
    return Heights(Name(p, "sensor_name"), Param(p, "offset"));
  }
  if (fn == "foot_height") {
    return Heights(Name(p, "sensor_name"));
  }
  if (fn == "foot_air_time") {
    return contacts.at(Name(p, "sensor_name")).air;
  }
  if (fn == "foot_contact") {
    auto values = contacts.at(Name(p, "sensor_name")).found;
    for (auto& value : values) {
      value = value > 0 ? 1 : 0;
    }
    return values;
  }
  if (fn == "foot_contact_forces") {
    auto values = contacts.at(Name(p, "sensor_name")).force;
    for (auto& v : values) {
      const float sign = static_cast<float>(v > 0) - static_cast<float>(v < 0);
      v = sign * Log1p(std::abs(v));
    }
    return values;
  }
  return task->Observation(fn, p);
}

void Simulation::Observe() {
  for (const auto& group : cfg.at("observation").as_object()) {
    auto& result = observations[std::string(group.key())];
    result.clear();
    for (const auto& entry : group.value().as_object()) {
      const auto& term = entry.value();
      auto values = EvaluateObservation(term);
      for (std::size_t i = 0; i < values.size(); ++i) {
        float& value = values[i];
        const auto& noise = term.at("noise");
        if (!noise.is_null()) {
          const float low = Param(noise, "n_min");
          const float high = Param(noise, "n_max");
          // UniformNoiseCfg uses rand_like()*width+low, not Tensor.uniform_.
          // Keep its separate float32 operations even on platforms whose
          // uniform_ kernel contracts the scale and offset into one FMA.
          const float sample = random.Unit() * (high - low) + low;
          const auto operation = Name(noise, "operation");
          if (operation == "add") {
            value += sample;
          } else if (operation == "scale") {
            value *= sample;
          } else {
            value = sample;
          }
        }
        const auto& clip = term.at("clip");
        if (!clip.is_null()) {
          value = std::clamp(value, static_cast<float>(Number(clip.at(0))),
                             static_cast<float>(Number(clip.at(1))));
        }
        const auto& scale = term.at("scale");
        if (!scale.is_null()) {
          value *= scale.is_array() ? Number(scale.at(i)) : Number(scale);
        }
        if (!std::isfinite(value)) {
          value = 0;
        }
      }
      result.insert(result.end(), values.begin(), values.end());
    }
  }
}

Json Simulation::State() const {
  boost::json::object result{{"seed", random.Seed()},
                             {"rng_draws", random.Draws()},
                             {"steps", steps},
                             {"total_steps", total_steps},
                             {"origin", ToJson(origin)},
                             {"action", ToJson(action)},
                             {"previous_action", ToJson(previous_action)},
                             {"command", task->State()},
                             {"terminated", terminated},
                             {"truncated", truncated}};
  boost::json::object entity_state;
  boost::json::object contact_state;
  boost::json::object timers;
  boost::json::object rewards;
  boost::json::object metric_values;
  for (const auto& e : entities) {
    entity_state[e.first] =
        boost::json::object{{"encoder_bias", ToJson(e.second.bias)}};
  }
  for (const auto& e : contacts) {
    const auto& c = e.second;
    contact_state[e.first] =
        boost::json::object{{"found", ToJson(c.found)},
                            {"force", ToJson(c.force)},
                            {"force_history", ToJson(c.history)},
                            {"current_air_time", ToJson(c.air)},
                            {"last_air_time", ToJson(c.last_air)},
                            {"current_contact_time", ToJson(c.contact)},
                            {"last_contact_time", ToJson(c.last_contact)}};
  }
  for (const auto& e : event_timers) {
    timers[e.first] = e.second;
  }
  for (const auto& e : reward_terms) {
    rewards[e.first] = e.second;
  }
  for (const auto& e : metrics) {
    metric_values[e.first] = e.second;
  }
  result["entities"] = std::move(entity_state);
  result["contacts"] = std::move(contact_state);
  result["event_timers"] = std::move(timers);
  result["reward_terms"] = std::move(rewards);
  result["metrics"] = std::move(metric_values);
  return result;
}

}  // namespace mjlab
