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

#ifndef ENVPOOL_MUJOCO_GYMNASIUM_ROBOTICS_UTILS_H_
#define ENVPOOL_MUJOCO_GYMNASIUM_ROBOTICS_UTILS_H_

#include <mujoco.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace gymnasium_robotics {

int JointQposSize(int joint_type);
int JointQvelSize(int joint_type);
int JointQposAddress(const mjModel* model, const std::string& name);
int JointQvelAddress(const mjModel* model, const std::string& name);
int ActuatorId(const mjModel* model, const std::string& name);
int BodyId(const mjModel* model, const std::string& name);
int SensorId(const mjModel* model, const std::string& name);
int SiteId(const mjModel* model, const std::string& name);

void SetJointQpos(const mjModel* model, mjData* data, const std::string& name,
                  const std::vector<mjtNum>& value);
void SetJointQpos(const mjModel* model, mjData* data, const std::string& name,
                  mjtNum value);
void SetJointQvel(const mjModel* model, mjData* data, const std::string& name,
                  const std::vector<mjtNum>& value);
std::vector<mjtNum> GetJointQpos(const mjModel* model, const mjData* data,
                                 const std::string& name);
std::vector<mjtNum> GetJointQvel(const mjModel* model, const mjData* data,
                                 const std::string& name);

std::array<mjtNum, 3> GetSiteXpos(const mjModel* model, const mjData* data,
                                  int site_id);
std::array<mjtNum, 3> GetSiteXvelp(const mjModel* model, const mjData* data,
                                   int site_id);
std::array<mjtNum, 3> GetSiteXvelr(const mjModel* model, const mjData* data,
                                   int site_id);
std::array<mjtNum, 9> GetSiteXmat(const mjModel* model, const mjData* data,
                                  int site_id);
std::array<mjtNum, 9> Quat2Mat(const std::array<mjtNum, 4>& quat);
std::array<mjtNum, 4> Euler2Quat(const std::array<mjtNum, 3>& euler);
std::array<mjtNum, 3> Mat2Euler(const std::array<mjtNum, 9>& mat);
std::array<mjtNum, 3> Quat2Euler(const std::array<mjtNum, 4>& quat);
std::array<mjtNum, 4> QuatConjugate(const std::array<mjtNum, 4>& quat);
std::array<mjtNum, 4> QuatMul(const std::array<mjtNum, 4>& lhs,
                              const std::array<mjtNum, 4>& rhs);
std::array<mjtNum, 4> QuatFromAngleAndAxis(mjtNum angle,
                                           const std::array<mjtNum, 3>& axis);
const std::vector<std::array<mjtNum, 4>>& ParallelQuats();

std::pair<std::vector<mjtNum>, std::vector<mjtNum>> RobotGetObs(
    const mjModel* model, const mjData* data);

void CtrlSetAction(const mjModel* model, mjData* data,
                   const std::vector<mjtNum>& action);
void MocapSetAction(const mjModel* model, mjData* data,
                    const std::vector<mjtNum>& action);
void ResetMocapWelds(mjModel* model, mjData* data);
void ResetMocap2BodyXpos(const mjModel* model, mjData* data);

}  // namespace gymnasium_robotics

#endif  // ENVPOOL_MUJOCO_GYMNASIUM_ROBOTICS_UTILS_H_
