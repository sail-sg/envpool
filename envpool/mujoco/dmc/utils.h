/*
 * Copyright 2022 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MUJOCO_DMC_UTILS_H_
#define ENVPOOL_MUJOCO_DMC_UTILS_H_

#include <mjxmacro.h>
#include <mujoco.h>

#include <random>
#include <string>
#include <vector>

namespace mujoco_dmc {

using RandInt = std::uniform_int_distribution<>;
using RandUniform = std::uniform_real_distribution<>;
using RandNormal = std::normal_distribution<>;

// xml related
std::string GetFileContent(const std::string& base_path,
                           const std::string& asset_name);
std::string XMLRemoveByBodyName(const std::string& content,
                                const std::vector<std::string>& body_names);
std::string XMLAddPoles(const std::string& content, int n_poles);
std::string XMLMakeSwimmer(const std::string& content, int n_joints);

// the following id is not 1 on 1 mapping
int GetQposId(mjModel* model, const std::string& name);
int GetQvelId(mjModel* model, const std::string& name);
int GetSensorId(mjModel* model, const std::string& name);

// rewards
enum class SigmoidType {
  kGaussian,
  kHyperbolic,
  kLongTail,
  kReciprocal,
  kCosine,
  kLinear,
  kQuadratic,
  kTanhSquared,
};

double RewardTolerance(double x, double bound_min = 0.0, double bound_max = 0.0,
                       double margin = 0.0, double value_at_margin = 0.1,
                       SigmoidType sigmoid_type = SigmoidType::kGaussian);

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_UTILS_H_
