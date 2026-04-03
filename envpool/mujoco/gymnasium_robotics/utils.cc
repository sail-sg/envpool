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

#include "envpool/mujoco/gymnasium_robotics/utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gymnasium_robotics {
namespace {

constexpr double kPi = 3.14159265358979323846;

int NamedId(const mjModel* model, mjtObj type, const std::string& name) {
  int id = mj_name2id(model, type, name.c_str());
  if (id == -1) {
    throw std::runtime_error("MuJoCo object not found: " + name);
  }
  return id;
}

bool StartsWith(const char* name, const char* prefix) {
  if (name == nullptr) {
    return false;
  }
  return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

std::array<mjtNum, 9> Euler2Mat(const std::array<mjtNum, 3>& euler) {
  double ai = -static_cast<double>(euler[2]);
  double aj = -static_cast<double>(euler[1]);
  double ak = -static_cast<double>(euler[0]);
  double si = std::sin(ai);
  double sj = std::sin(aj);
  double sk = std::sin(ak);
  double ci = std::cos(ai);
  double cj = std::cos(aj);
  double ck = std::cos(ak);
  double cc = ci * ck;
  double cs = ci * sk;
  double sc = si * ck;
  double ss = si * sk;
  return {static_cast<mjtNum>(cj * ci),      static_cast<mjtNum>(cj * si),
          static_cast<mjtNum>(-sj),          static_cast<mjtNum>(sj * cs - sc),
          static_cast<mjtNum>(sj * ss + cc), static_cast<mjtNum>(cj * sk),
          static_cast<mjtNum>(sj * cc + ss), static_cast<mjtNum>(sj * sc - cs),
          static_cast<mjtNum>(cj * ck)};
}

std::array<mjtNum, 4> CanonicalQuat(const std::array<mjtNum, 4>& quat) {
  if (quat[0] < 0.0) {
    return {-quat[0], -quat[1], -quat[2], -quat[3]};
  }
  return quat;
}

}  // namespace

int JointQposSize(int joint_type) {
  if (joint_type == mjJNT_FREE) {
    return 7;
  }
  if (joint_type == mjJNT_BALL) {
    return 4;
  }
  if (joint_type == mjJNT_HINGE || joint_type == mjJNT_SLIDE) {
    return 1;
  }
  throw std::runtime_error("Unsupported MuJoCo joint type for qpos.");
}

int JointQvelSize(int joint_type) {
  if (joint_type == mjJNT_FREE) {
    return 6;
  }
  if (joint_type == mjJNT_BALL) {
    return 3;
  }
  if (joint_type == mjJNT_HINGE || joint_type == mjJNT_SLIDE) {
    return 1;
  }
  throw std::runtime_error("Unsupported MuJoCo joint type for qvel.");
}

int JointQposAddress(const mjModel* model, const std::string& name) {
  return model->jnt_qposadr[NamedId(model, mjOBJ_JOINT, name)];
}

int JointQvelAddress(const mjModel* model, const std::string& name) {
  return model->jnt_dofadr[NamedId(model, mjOBJ_JOINT, name)];
}

int ActuatorId(const mjModel* model, const std::string& name) {
  return NamedId(model, mjOBJ_ACTUATOR, name);
}

int BodyId(const mjModel* model, const std::string& name) {
  return NamedId(model, mjOBJ_BODY, name);
}

int SensorId(const mjModel* model, const std::string& name) {
  return NamedId(model, mjOBJ_SENSOR, name);
}

int SiteId(const mjModel* model, const std::string& name) {
  return NamedId(model, mjOBJ_SITE, name);
}

void SetJointQpos(const mjModel* model, mjData* data, const std::string& name,
                  const std::vector<mjtNum>& value) {
  int joint_id = NamedId(model, mjOBJ_JOINT, name);
  int joint_type = model->jnt_type[joint_id];
  int addr = model->jnt_qposadr[joint_id];
  int dim = JointQposSize(joint_type);
  if (static_cast<int>(value.size()) != dim) {
    throw std::runtime_error("Invalid qpos dimension for joint: " + name);
  }
  for (int i = 0; i < dim; ++i) {
    data->qpos[addr + i] = value[i];
  }
}

void SetJointQpos(const mjModel* model, mjData* data, const std::string& name,
                  mjtNum value) {
  SetJointQpos(model, data, name, std::vector<mjtNum>{value});
}

void SetJointQvel(const mjModel* model, mjData* data, const std::string& name,
                  const std::vector<mjtNum>& value) {
  int joint_id = NamedId(model, mjOBJ_JOINT, name);
  int joint_type = model->jnt_type[joint_id];
  int addr = model->jnt_dofadr[joint_id];
  int dim = JointQvelSize(joint_type);
  if (static_cast<int>(value.size()) != dim) {
    throw std::runtime_error("Invalid qvel dimension for joint: " + name);
  }
  for (int i = 0; i < dim; ++i) {
    data->qvel[addr + i] = value[i];
  }
}

std::vector<mjtNum> GetJointQpos(const mjModel* model, const mjData* data,
                                 const std::string& name) {
  int joint_id = NamedId(model, mjOBJ_JOINT, name);
  int joint_type = model->jnt_type[joint_id];
  int addr = model->jnt_qposadr[joint_id];
  int dim = JointQposSize(joint_type);
  std::vector<mjtNum> value(dim);
  for (int i = 0; i < dim; ++i) {
    value[i] = data->qpos[addr + i];
  }
  return value;
}

std::vector<mjtNum> GetJointQvel(const mjModel* model, const mjData* data,
                                 const std::string& name) {
  int joint_id = NamedId(model, mjOBJ_JOINT, name);
  int joint_type = model->jnt_type[joint_id];
  int addr = model->jnt_dofadr[joint_id];
  int dim = JointQvelSize(joint_type);
  std::vector<mjtNum> value(dim);
  for (int i = 0; i < dim; ++i) {
    value[i] = data->qvel[addr + i];
  }
  return value;
}

std::array<mjtNum, 3> GetSiteXpos(const mjModel* model, const mjData* data,
                                  int site_id) {
  (void)model;
  std::array<mjtNum, 3> value{};
  for (int i = 0; i < 3; ++i) {
    value[i] = data->site_xpos[site_id * 3 + i];
  }
  return value;
}

std::array<mjtNum, 3> GetSiteXvelp(const mjModel* model, const mjData* data,
                                   int site_id) {
  std::vector<mjtNum> jacp(3 * model->nv);
  mj_jacSite(model, data, jacp.data(), nullptr, site_id);
  std::array<mjtNum, 3> value{};
  mju_mulMatVec(value.data(), jacp.data(), data->qvel, 3, model->nv);
  return value;
}

std::array<mjtNum, 3> GetSiteXvelr(const mjModel* model, const mjData* data,
                                   int site_id) {
  std::vector<mjtNum> jacr(3 * model->nv);
  mj_jacSite(model, data, nullptr, jacr.data(), site_id);
  std::array<mjtNum, 3> value{};
  mju_mulMatVec(value.data(), jacr.data(), data->qvel, 3, model->nv);
  return value;
}

std::array<mjtNum, 9> GetSiteXmat(const mjModel* model, const mjData* data,
                                  int site_id) {
  (void)model;
  std::array<mjtNum, 9> value{};
  for (int i = 0; i < 9; ++i) {
    value[i] = data->site_xmat[site_id * 9 + i];
  }
  return value;
}

std::array<mjtNum, 9> Quat2Mat(const std::array<mjtNum, 4>& quat) {
  constexpr double k_eps = 2.220446049250313e-16;
  double w = quat[0];
  double x = quat[1];
  double y = quat[2];
  double z = quat[3];
  double nq = w * w + x * x + y * y + z * z;
  if (nq <= k_eps) {
    return {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  }
  double s = 2.0 / nq;
  double x_scale = x * s;
  double y_scale = y * s;
  double z_scale = z * s;
  double w_x = w * x_scale;
  double w_y = w * y_scale;
  double w_z = w * z_scale;
  double x_x = x * x_scale;
  double x_y = x * y_scale;
  double x_z = x * z_scale;
  double y_y = y * y_scale;
  double y_z = y * z_scale;
  double z_z = z * z_scale;
  return {
      static_cast<mjtNum>(1.0 - (y_y + z_z)), static_cast<mjtNum>(x_y - w_z),
      static_cast<mjtNum>(x_z + w_y),         static_cast<mjtNum>(x_y + w_z),
      static_cast<mjtNum>(1.0 - (x_x + z_z)), static_cast<mjtNum>(y_z - w_x),
      static_cast<mjtNum>(x_z - w_y),         static_cast<mjtNum>(y_z + w_x),
      static_cast<mjtNum>(1.0 - (x_x + y_y))};
}

std::array<mjtNum, 4> Euler2Quat(const std::array<mjtNum, 3>& euler) {
  double ai = static_cast<double>(euler[2]) / 2.0;
  double aj = -static_cast<double>(euler[1]) / 2.0;
  double ak = static_cast<double>(euler[0]) / 2.0;
  double si = std::sin(ai);
  double sj = std::sin(aj);
  double sk = std::sin(ak);
  double ci = std::cos(ai);
  double cj = std::cos(aj);
  double ck = std::cos(ak);
  double cc = ci * ck;
  double cs = ci * sk;
  double sc = si * ck;
  double ss = si * sk;
  return {static_cast<mjtNum>(cj * cc + sj * ss),
          static_cast<mjtNum>(cj * cs - sj * sc),
          static_cast<mjtNum>(-(cj * ss + sj * cc)),
          static_cast<mjtNum>(cj * sc - sj * cs)};
}

std::array<mjtNum, 3> Mat2Euler(const std::array<mjtNum, 9>& mat) {
  constexpr double k_eps = 4.0 * 2.220446049250313e-16;
  double cy = std::sqrt(mat[8] * mat[8] + mat[5] * mat[5]);
  if (cy > k_eps) {
    return {static_cast<mjtNum>(-std::atan2(mat[5], mat[8])),
            static_cast<mjtNum>(-std::atan2(-mat[2], cy)),
            static_cast<mjtNum>(-std::atan2(mat[1], mat[0]))};
  }
  return {0.0, static_cast<mjtNum>(-std::atan2(-mat[2], cy)),
          static_cast<mjtNum>(-std::atan2(-mat[3], mat[4]))};
}

std::array<mjtNum, 3> Quat2Euler(const std::array<mjtNum, 4>& quat) {
  return Mat2Euler(Quat2Mat(quat));
}

std::array<mjtNum, 4> QuatConjugate(const std::array<mjtNum, 4>& quat) {
  return {quat[0], -quat[1], -quat[2], -quat[3]};
}

std::array<mjtNum, 4> QuatMul(const std::array<mjtNum, 4>& lhs,
                              const std::array<mjtNum, 4>& rhs) {
  return {
      lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] - lhs[3] * rhs[3],
      lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[3] - lhs[3] * rhs[2],
      lhs[0] * rhs[2] + lhs[2] * rhs[0] + lhs[3] * rhs[1] - lhs[1] * rhs[3],
      lhs[0] * rhs[3] + lhs[3] * rhs[0] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
  };
}

std::array<mjtNum, 4> QuatFromAngleAndAxis(mjtNum angle,
                                           const std::array<mjtNum, 3>& axis) {
  double norm =
      std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  if (norm <= 0.0) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  double sin_angle = std::sin(static_cast<double>(angle) / 2.0);
  double scale = sin_angle / norm;
  return {static_cast<mjtNum>(std::cos(static_cast<double>(angle) / 2.0)),
          static_cast<mjtNum>(axis[0] * scale),
          static_cast<mjtNum>(axis[1] * scale),
          static_cast<mjtNum>(axis[2] * scale)};
}

const std::vector<std::array<mjtNum, 4>>& ParallelQuats() {
  static const std::vector<std::array<mjtNum, 4>> k_parallel_quats = [] {
    std::vector<std::array<mjtNum, 3>> parallel_eulers;
    const std::array<mjtNum, 4> mult90 = {
        0.0,
        static_cast<mjtNum>(kPi / 2.0),
        static_cast<mjtNum>(-kPi / 2.0),
        static_cast<mjtNum>(kPi),
    };
    for (mjtNum x : mult90) {
      for (mjtNum y : mult90) {
        for (mjtNum z : mult90) {
          auto canonical = Mat2Euler(Euler2Mat({x, y, z}));
          for (int i = 0; i < 3; ++i) {
            int rounded =
                static_cast<int>(std::llround(canonical[i] / (kPi / 2.0)));
            if ((i == 0 || i == 2) && rounded == -2) {
              rounded = 2;
            }
            canonical[i] = static_cast<mjtNum>(rounded * (kPi / 2.0));
          }
          bool duplicate = false;
          for (const auto& existing : parallel_eulers) {
            if (existing == canonical) {
              duplicate = true;
              break;
            }
          }
          if (!duplicate) {
            parallel_eulers.push_back(canonical);
          }
        }
      }
    }
    std::vector<std::array<mjtNum, 4>> parallel_quats;
    parallel_quats.reserve(parallel_eulers.size());
    for (const auto& euler : parallel_eulers) {
      parallel_quats.push_back(CanonicalQuat(Euler2Quat(euler)));
    }
    if (parallel_quats.size() != 24) {
      throw std::runtime_error("Failed to derive 24 parallel rotations.");
    }
    return parallel_quats;
  }();
  return k_parallel_quats;
}

std::pair<std::vector<mjtNum>, std::vector<mjtNum>> RobotGetObs(
    const mjModel* model, const mjData* data) {
  std::vector<mjtNum> qpos;
  std::vector<mjtNum> qvel;
  for (int joint_id = 0; joint_id < model->njnt; ++joint_id) {
    const char* joint_name = mj_id2name(model, mjOBJ_JOINT, joint_id);
    if (!StartsWith(joint_name, "robot")) {
      continue;
    }
    int qpos_addr = model->jnt_qposadr[joint_id];
    int qvel_addr = model->jnt_dofadr[joint_id];
    for (int i = 0; i < JointQposSize(model->jnt_type[joint_id]); ++i) {
      qpos.push_back(data->qpos[qpos_addr + i]);
    }
    for (int i = 0; i < JointQvelSize(model->jnt_type[joint_id]); ++i) {
      qvel.push_back(data->qvel[qvel_addr + i]);
    }
  }
  return {std::move(qpos), std::move(qvel)};
}

void CtrlSetAction(const mjModel* model, mjData* data,
                   const std::vector<mjtNum>& action) {
  int action_offset = model->nmocap * 7;
  if (model->nu <= 0) {
    return;
  }
  if (static_cast<int>(action.size()) < action_offset + model->nu) {
    throw std::runtime_error("Action too short for MuJoCo ctrl update.");
  }
  for (int i = 0; i < model->nu; ++i) {
    mjtNum value = action[action_offset + i];
    if (model->actuator_biastype[i] == mjBIAS_NONE) {
      data->ctrl[i] = value;
    } else {
      int joint_id = model->actuator_trnid[2 * i];
      int qpos_addr = model->jnt_qposadr[joint_id];
      data->ctrl[i] = data->qpos[qpos_addr] + value;
    }
  }
}

void ResetMocap2BodyXpos(const mjModel* model, mjData* data) {
  for (int eq_id = 0; eq_id < model->neq; ++eq_id) {
    if (model->eq_type[eq_id] != mjEQ_WELD) {
      continue;
    }
    int obj1_id = model->eq_obj1id[eq_id];
    int obj2_id = model->eq_obj2id[eq_id];
    int mocap_id = model->body_mocapid[obj1_id];
    int body_id = obj2_id;
    if (mocap_id == -1) {
      mocap_id = model->body_mocapid[obj2_id];
      body_id = obj1_id;
    }
    if (mocap_id == -1) {
      throw std::runtime_error("Mocap weld equality does not reference mocap.");
    }
    for (int i = 0; i < 3; ++i) {
      data->mocap_pos[3 * mocap_id + i] = data->xpos[3 * body_id + i];
    }
    for (int i = 0; i < 4; ++i) {
      data->mocap_quat[4 * mocap_id + i] = data->xquat[4 * body_id + i];
    }
  }
}

void MocapSetAction(const mjModel* model, mjData* data,
                    const std::vector<mjtNum>& action) {
  if (model->nmocap <= 0) {
    return;
  }
  if (static_cast<int>(action.size()) < model->nmocap * 7) {
    throw std::runtime_error("Action too short for MuJoCo mocap update.");
  }
  ResetMocap2BodyXpos(model, data);
  for (int mocap_id = 0; mocap_id < model->nmocap; ++mocap_id) {
    int action_offset = 7 * mocap_id;
    for (int i = 0; i < 3; ++i) {
      data->mocap_pos[3 * mocap_id + i] += action[action_offset + i];
    }
    for (int i = 0; i < 4; ++i) {
      data->mocap_quat[4 * mocap_id + i] += action[action_offset + 3 + i];
    }
  }
}

void ResetMocapWelds(mjModel* model, mjData* data) {
  if (model->nmocap <= 0) {
    return;
  }
  for (int eq_id = 0; eq_id < model->neq; ++eq_id) {
    if (model->eq_type[eq_id] != mjEQ_WELD) {
      continue;
    }
    mjtNum* eq_data = model->eq_data + eq_id * mjNEQDATA;
    for (int i = 0; i < 7; ++i) {
      eq_data[i] = (i == 6) ? 1.0 : 0.0;
    }
  }
  mj_forward(model, data);
}

}  // namespace gymnasium_robotics
