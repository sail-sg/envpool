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

#ifndef ENVPOOL_MUJOCO_MYOSUITE_MYOCHALLENGE_EXTENDED_H_
#define ENVPOOL_MUJOCO_MYOSUITE_MYOCHALLENGE_EXTENDED_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "envpool/mujoco/myosuite/myochallenge.h"

namespace myosuite_envpool {

namespace challenge_extra_detail {

inline mjtNum MeanSquareAct(const mjModel* model, const mjData* data) {
  if (model->na == 0) {
    return 0.0;
  }
  mjtNum total = 0.0;
  for (int i = 0; i < model->na; ++i) {
    total += data->act[i] * data->act[i];
  }
  return total / static_cast<mjtNum>(model->na);
}

inline mjtNum L2ActReg(const mjModel* model, const mjData* data) {
  if (model->na == 0) {
    return 0.0;
  }
  mjtNum total = 0.0;
  for (int i = 0; i < model->na; ++i) {
    total += data->act[i] * data->act[i];
  }
  return std::sqrt(total) / static_cast<mjtNum>(model->na);
}

inline mjtNum Norm2(mjtNum x, mjtNum y) { return std::sqrt(x * x + y * y); }

inline mjtNum Clip(mjtNum value, mjtNum low, mjtNum high) {
  return std::min(std::max(value, low), high);
}

inline mjtNum NormalizeSignedAngle(mjtNum angle) {
  constexpr mjtNum two_pi = static_cast<mjtNum>(2.0) * detail::kPi;
  while (angle > detail::kPi) {
    angle -= two_pi;
  }
  while (angle < -detail::kPi) {
    angle += two_pi;
  }
  return angle;
}

inline mjtNum QuatYaw(const mjtNum* quat) {
  std::array<mjtNum, 9> mat{};
  mju_quat2Mat(mat.data(), quat);
  return challenge_detail::Mat9ToEuler(mat.data())[2];
}

inline std::array<mjtNum, 4> YawToQuat(mjtNum yaw) {
  return challenge_detail::EulerXYZToQuat({0.0, 0.0, yaw});
}

inline std::array<mjtNum, 4> ScipyEulerXYZToQuat(
    const std::array<mjtNum, 3>& euler) {
  mjtNum half_x = euler[0] * static_cast<mjtNum>(0.5);
  mjtNum half_y = euler[1] * static_cast<mjtNum>(0.5);
  mjtNum half_z = euler[2] * static_cast<mjtNum>(0.5);
  mjtNum sx = std::sin(half_x);
  mjtNum sy = std::sin(half_y);
  mjtNum sz = std::sin(half_z);
  mjtNum cx = std::cos(half_x);
  mjtNum cy = std::cos(half_y);
  mjtNum cz = std::cos(half_z);
  return {
      cx * cy * cz + sx * sy * sz,
      sx * cy * cz - cx * sy * sz,
      cx * sy * cz + sx * cy * sz,
      cx * cy * sz - sx * sy * cz,
  };
}

inline std::uint16_t ReadLe16(std::istream* input) {
  unsigned char bytes[2];
  input->read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!*input) {
    throw std::runtime_error("Failed to read uint16.");
  }
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

inline std::uint32_t ReadLe32(std::istream* input) {
  unsigned char bytes[4];
  input->read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!*input) {
    throw std::runtime_error("Failed to read uint32.");
  }
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

inline std::pair<int, int> ParseNpyShape2d(const std::string& header) {
  std::size_t shape_pos = header.find("shape");
  if (shape_pos == std::string::npos) {
    throw std::runtime_error("Missing shape in NPY header.");
  }
  std::size_t open = header.find('(', shape_pos);
  std::size_t comma = header.find(',', open);
  std::size_t close = header.find(')', comma);
  if (open == std::string::npos || comma == std::string::npos ||
      close == std::string::npos) {
    throw std::runtime_error("Invalid shape in NPY header.");
  }
  int rows = std::stoi(header.substr(open + 1, comma - open - 1));
  int cols = std::stoi(header.substr(comma + 1, close - comma - 1));
  return {rows, cols};
}

inline std::vector<mjtNum> LoadNormalizedReliefPatch(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open MyoSuite relief asset: " + path);
  }
  char magic[6];
  input.read(magic, sizeof(magic));
  if (!input || std::string_view(magic, sizeof(magic)) != "\x93NUMPY") {
    throw std::runtime_error("Invalid NPY magic in MyoSuite relief asset.");
  }
  unsigned char version[2];
  input.read(reinterpret_cast<char*>(version), sizeof(version));
  if (!input) {
    throw std::runtime_error("Failed to read NPY version.");
  }
  std::size_t header_len = 0;
  if (version[0] == 1) {
    header_len = ReadLe16(&input);
  } else if (version[0] == 2) {
    header_len = ReadLe32(&input);
  } else {
    throw std::runtime_error("Unsupported NPY version for relief asset.");
  }
  std::string header(header_len, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input) {
    throw std::runtime_error("Failed to read NPY header.");
  }
  if (header.find("'fortran_order': False") == std::string::npos ||
      (header.find("'descr': '|u1'") == std::string::npos &&
       header.find("'descr': '<u1'") == std::string::npos)) {
    throw std::runtime_error("Unexpected MyoSuite relief NPY layout.");
  }
  auto [rows, cols] = ParseNpyShape2d(header);
  std::vector<unsigned char> raw(static_cast<std::size_t>(rows * cols));
  input.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(raw.size()));
  if (!input) {
    throw std::runtime_error("Failed to read MyoSuite relief payload.");
  }
  auto [min_it, max_it] = std::minmax_element(raw.begin(), raw.end());
  mjtNum low = static_cast<mjtNum>(*min_it);
  mjtNum high = static_cast<mjtNum>(*max_it);
  mjtNum denom = std::max(high - low, static_cast<mjtNum>(1e-12));
  std::vector<mjtNum> normalized(raw.size(), 0.0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      std::size_t src = static_cast<std::size_t>(row * cols + col);
      std::size_t dst = static_cast<std::size_t>((rows - 1 - row) * cols + col);
      normalized[dst] = (static_cast<mjtNum>(raw[src]) - low) / denom;
    }
  }
  return normalized;
}

inline std::vector<int> CollectJointQposAdrs(
    const mjModel* model, const std::vector<std::string>& joint_names) {
  std::vector<int> out;
  out.reserve(joint_names.size());
  for (const std::string& joint_name : joint_names) {
    int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    if (joint_id == -1) {
      throw std::runtime_error("Missing joint: " + joint_name);
    }
    out.push_back(model->jnt_qposadr[joint_id]);
  }
  return out;
}

inline std::vector<int> CollectJointDofAdrs(
    const mjModel* model, const std::vector<std::string>& joint_names) {
  std::vector<int> out;
  out.reserve(joint_names.size());
  for (const std::string& joint_name : joint_names) {
    int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    if (joint_id == -1) {
      throw std::runtime_error("Missing joint: " + joint_name);
    }
    out.push_back(model->jnt_dofadr[joint_id]);
  }
  return out;
}

inline mjtNum AverageJointLimitForce(const mjModel* model, const mjData* data,
                                     const std::vector<int>& dof_adrs) {
  if (dof_adrs.empty() || data->nefc == 0) {
    return 0.0;
  }
  std::vector<mjtNum> limit_force(data->nefc, 0.0);
  for (int efc = 0; efc < data->nefc; ++efc) {
    if (data->efc_type[efc] == mjCNSTR_LIMIT_JOINT) {
      limit_force[efc] = data->efc_force[efc];
    }
  }
  std::vector<mjtNum> joint_force(model->nv, 0.0);
  mj_mulJacTVec(model, data, joint_force.data(), limit_force.data());
  mjtNum total = 0.0;
  for (int dof_adr : dof_adrs) {
    total += std::abs(Clip(joint_force[dof_adr], -1000.0, 1000.0)) / 1000.0;
  }
  return total / static_cast<mjtNum>(dof_adrs.size());
}

inline void AssignVectorField(std::vector<mjtNum>* out, const mjtNum* src,
                              const std::vector<int>& indices,
                              mjtNum scale = 1.0) {
  out->clear();
  out->reserve(indices.size());
  for (int index : indices) {
    out->push_back(src[index] * scale);
  }
}

inline void AssignSensorField(const mjModel* model, const mjData* data,
                              const std::vector<int>& sensor_ids,
                              std::vector<mjtNum>* out) {
  out->clear();
  for (int sensor_id : sensor_ids) {
    int adr = model->sensor_adr[sensor_id];
    int dim = model->sensor_dim[sensor_id];
    out->insert(out->end(), data->sensordata + adr,
                data->sensordata + adr + dim);
  }
}

inline int RequireId(const mjModel* model, mjtObj obj,
                     const std::string& name) {
  int id = mj_name2id(model, obj, name.c_str());
  if (id == -1) {
    throw std::runtime_error("Missing MuJoCo object: " + name);
  }
  return id;
}

class ColoredNoiseProcess {
 public:
  ColoredNoiseProcess(mjtNum beta, int channels, int time_steps,
                      mjtNum scale = 1.0, mjtNum max_period = 0.0)
      : beta_(beta),
        channels_(channels),
        time_steps_(time_steps),
        scale_(scale),
        minimum_frequency_(max_period > 0.0 ? 1.0 / max_period : 0.0) {}

  void SetScale(mjtNum scale) { scale_ = scale; }

  void Reset(std::mt19937* gen) {
    buffer_ =
        PowerlawPsdGaussian(beta_, channels_, time_steps_, minimum_frequency_,
                            gen == nullptr ? fallback_gen_ : *gen);
    idx_ = 0;
  }

  void SetState(const std::vector<mjtNum>& buffer, int idx) {
    if (!buffer.empty() &&
        static_cast<int>(buffer.size()) != channels_ * time_steps_) {
      throw std::runtime_error("Colored noise buffer has unexpected size.");
    }
    buffer_ = buffer;
    idx_ = std::max(0, idx);
  }

  std::vector<mjtNum> Sample(std::mt19937* gen) {
    if (buffer_.empty() || idx_ >= time_steps_) {
      Reset(gen);
    }
    std::vector<mjtNum> out(channels_);
    for (int channel = 0; channel < channels_; ++channel) {
      out[channel] = scale_ * buffer_[channel * time_steps_ + idx_];
    }
    ++idx_;
    return out;
  }

 private:
  static std::vector<mjtNum> PowerlawPsdGaussian(mjtNum exponent, int channels,
                                                 int samples, mjtNum fmin,
                                                 std::mt19937& gen) {
    if (channels <= 0 || samples <= 0) {
      return {};
    }
    if (fmin < 0.0 || fmin > 0.5) {
      throw std::runtime_error("Colored noise fmin must be in [0, 0.5].");
    }
    constexpr mjtNum kTwoPi = static_cast<mjtNum>(2.0) * detail::kPi;
    int freq_count = samples / 2 + 1;
    std::vector<mjtNum> s_scale(freq_count, 0.0);
    for (int i = 0; i < freq_count; ++i) {
      s_scale[i] = static_cast<mjtNum>(i) / static_cast<mjtNum>(samples);
    }
    fmin =
        std::max(fmin, static_cast<mjtNum>(1.0) / static_cast<mjtNum>(samples));
    int ix = 0;
    while (ix < freq_count && s_scale[ix] < fmin) {
      ++ix;
    }
    if (ix > 0 && ix < freq_count) {
      std::fill(s_scale.begin(), s_scale.begin() + ix, s_scale[ix]);
    }
    for (mjtNum& value : s_scale) {
      value = std::pow(value, -exponent / static_cast<mjtNum>(2.0));
    }
    mjtNum sigma_sq = 0.0;
    for (int i = 1; i < freq_count; ++i) {
      mjtNum weight = s_scale[i];
      if ((samples % 2) == 0 && i == freq_count - 1) {
        weight *= static_cast<mjtNum>(0.5);
      }
      sigma_sq += weight * weight;
    }
    mjtNum sigma = static_cast<mjtNum>(2.0) * std::sqrt(sigma_sq) /
                   static_cast<mjtNum>(samples);
    if (sigma == 0.0) {
      throw std::runtime_error("Colored noise sigma unexpectedly zero.");
    }
    std::normal_distribution<mjtNum> normal(0.0, 1.0);
    std::vector<mjtNum> out(channels * samples, 0.0);
    std::vector<mjtNum> sr(freq_count, 0.0);
    std::vector<mjtNum> si(freq_count, 0.0);
    for (int channel = 0; channel < channels; ++channel) {
      for (int i = 0; i < freq_count; ++i) {
        sr[i] = normal(gen) * s_scale[i];
        si[i] = normal(gen) * s_scale[i];
      }
      si[0] = 0.0;
      sr[0] *= std::sqrt(static_cast<mjtNum>(2.0));
      if ((samples % 2) == 0) {
        si[freq_count - 1] = 0.0;
        sr[freq_count - 1] *= std::sqrt(static_cast<mjtNum>(2.0));
      }
      int mirrored_limit = (samples % 2) == 0 ? freq_count - 1 : freq_count;
      for (int t = 0; t < samples; ++t) {
        mjtNum value = sr[0];
        for (int i = 1; i < mirrored_limit; ++i) {
          mjtNum angle = kTwoPi * static_cast<mjtNum>(i) *
                         static_cast<mjtNum>(t) / static_cast<mjtNum>(samples);
          value += static_cast<mjtNum>(2.0) *
                   (sr[i] * std::cos(angle) - si[i] * std::sin(angle));
        }
        if ((samples % 2) == 0) {
          value += sr[freq_count - 1] * std::cos(detail::kPi * t);
        }
        out[channel * samples + t] =
            value / static_cast<mjtNum>(samples) / sigma;
      }
    }
    return out;
  }

  mjtNum beta_;
  int channels_;
  int time_steps_;
  mjtNum scale_;
  mjtNum minimum_frequency_;
  int idx_{0};
  std::vector<mjtNum> buffer_;
  std::mt19937 fallback_gen_{0};
};

enum class RunTrackOslState : int {
  kEarlyStance = 0,
  kLateStance = 1,
  kEarlySwing = 2,
  kLateSwing = 3,
};

struct RunTrackOslSensorData {
  mjtNum knee_angle{0.0};
  mjtNum knee_vel{0.0};
  mjtNum ankle_angle{0.0};
  mjtNum ankle_vel{0.0};
  mjtNum load{0.0};
};

class RunTrackOslController {
 public:
  RunTrackOslController() = default;

  explicit RunTrackOslController(mjtNum body_mass) { Initialize(body_mass); }

  void Reset(RunTrackOslState state) {
    state_ = state;
    running_ = false;
  }

  void Start() { running_ = true; }

  void Initialize(mjtNum body_mass) { InitializeDefaults(body_mass); }

  void SetState(RunTrackOslState state) { state_ = state; }

  int StateId() const { return static_cast<int>(state_); }

  void Update(const RunTrackOslSensorData& sensor_data) {
    sensor_data_ = sensor_data;
    if (!running_) {
      return;
    }
    const StateParams& params = states_[StateId()];
    for (int i = 0; i < params.threshold_count; ++i) {
      if (ThresholdSatisfied(params.thresholds[i], sensor_data_)) {
        state_ = NextState(state_);
        return;
      }
    }
  }

  std::array<mjtNum, 2> GetTorques() const {
    return {JointTorque(/*knee=*/true), JointTorque(/*knee=*/false)};
  }

 private:
  struct JointParams {
    mjtNum stiffness{0.0};
    mjtNum damping{0.0};
    mjtNum target_angle{0.0};
  };

  struct Threshold {
    enum class Variable : std::uint8_t {
      kLoad,
      kKneeAngle,
      kKneeVel,
      kAnkleAngle,
    };

    Variable variable{Variable::kLoad};
    mjtNum value{0.0};
    bool above{false};
  };

  struct StateParams {
    JointParams knee;
    JointParams ankle;
    std::array<Threshold, 2> thresholds{};
    int threshold_count{0};
  };

  static RunTrackOslState NextState(RunTrackOslState state) {
    switch (state) {
      case RunTrackOslState::kEarlyStance:
        return RunTrackOslState::kLateStance;
      case RunTrackOslState::kLateStance:
        return RunTrackOslState::kEarlySwing;
      case RunTrackOslState::kEarlySwing:
        return RunTrackOslState::kLateSwing;
      case RunTrackOslState::kLateSwing:
        return RunTrackOslState::kEarlyStance;
    }
    return RunTrackOslState::kEarlyStance;
  }

  static Threshold MakeThreshold(Threshold::Variable variable, mjtNum value,
                                 bool above) {
    return Threshold{variable, value, above};
  }

  static bool ThresholdSatisfied(const Threshold& threshold,
                                 const RunTrackOslSensorData& sensor_data) {
    mjtNum observed = 0.0;
    switch (threshold.variable) {
      case Threshold::Variable::kLoad:
        observed = sensor_data.load;
        break;
      case Threshold::Variable::kKneeAngle:
        observed = sensor_data.knee_angle;
        break;
      case Threshold::Variable::kKneeVel:
        observed = sensor_data.knee_vel;
        break;
      case Threshold::Variable::kAnkleAngle:
        observed = sensor_data.ankle_angle;
        break;
    }
    return threshold.above ? observed > threshold.value
                           : observed < threshold.value;
  }

  mjtNum JointTorque(bool knee) const {
    const StateParams& params = states_[StateId()];
    const JointParams& joint = knee ? params.knee : params.ankle;
    const mjtNum angle =
        knee ? sensor_data_.knee_angle : sensor_data_.ankle_angle;
    const mjtNum velocity =
        knee ? sensor_data_.knee_vel : sensor_data_.ankle_vel;
    const mjtNum peak_torque = knee ? knee_peak_torque_ : ankle_peak_torque_;
    return challenge_extra_detail::Clip(
        joint.stiffness * (joint.target_angle - angle) -
            joint.damping * velocity,
        -peak_torque, peak_torque);
  }

  void InitializeDefaults(mjtNum body_mass) {
    const mjtNum body_weight = body_mass * static_cast<mjtNum>(9.81);
    const mjtNum deg = detail::kPi / static_cast<mjtNum>(180.0);
    states_[static_cast<int>(RunTrackOslState::kEarlyStance)] = StateParams{
        JointParams{99.372, 3.180, 5.0 * deg},
        JointParams{19.874, 0.0, -2.0 * deg},
        {MakeThreshold(Threshold::Variable::kLoad,
                       static_cast<mjtNum>(0.25) * body_weight,
                       /*above=*/true),
         MakeThreshold(Threshold::Variable::kAnkleAngle, 6.0 * deg,
                       /*above=*/true)},
        2,
    };
    states_[static_cast<int>(RunTrackOslState::kLateStance)] = StateParams{
        JointParams{99.372, 1.272, 8.0 * deg},
        JointParams{79.498, 0.063, -20.0 * deg},
        {MakeThreshold(Threshold::Variable::kLoad,
                       static_cast<mjtNum>(0.15) * body_weight,
                       /*above=*/false)},
        1,
    };
    states_[static_cast<int>(RunTrackOslState::kEarlySwing)] = StateParams{
        JointParams{39.749, 0.063, 60.0 * deg},
        JointParams{7.949, 0.0, 25.0 * deg},
        {MakeThreshold(Threshold::Variable::kKneeAngle, 50.0 * deg,
                       /*above=*/true),
         MakeThreshold(Threshold::Variable::kKneeVel, 3.0 * deg,
                       /*above=*/false)},
        2,
    };
    states_[static_cast<int>(RunTrackOslState::kLateSwing)] = StateParams{
        JointParams{15.899, 3.816, 5.0 * deg},
        JointParams{7.949, 0.0, 15.0 * deg},
        {MakeThreshold(Threshold::Variable::kLoad,
                       static_cast<mjtNum>(0.4) * body_weight,
                       /*above=*/true),
         MakeThreshold(Threshold::Variable::kKneeAngle, 30.0 * deg,
                       /*above=*/false)},
        2,
    };
    Reset(RunTrackOslState::kEarlyStance);
  }

  std::array<StateParams, 4> states_{};
  RunTrackOslSensorData sensor_data_{};
  RunTrackOslState state_{RunTrackOslState::kEarlyStance};
  bool running_{false};
  mjtNum knee_peak_torque_{142.272};
  mjtNum ankle_peak_torque_{168.192};
};

}  // namespace challenge_extra_detail

class MyoChallengeRunTrackEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0), "ctrl_dim"_.Bind(0),
        "reset_type"_.Bind(std::string("random")),
        "terrain"_.Bind(std::string("flat")), "start_pos"_.Bind(14.0),
        "end_pos"_.Bind(-15.0), "real_width"_.Bind(1.0),
        "hills_difficulties"_.Bind(std::vector<double>{}),
        "rough_difficulties"_.Bind(std::vector<double>{}),
        "stairs_difficulties"_.Bind(std::vector<double>{}),
        "reward_sparse_w"_.Bind(1.0), "reward_solved_w"_.Bind(10.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_hfield_data"_.Bind(std::vector<double>{}),
        "test_terrain_geom_rgba"_.Bind(std::vector<double>{}),
        "test_terrain_geom_pos"_.Bind(std::vector<double>{}),
        "test_terrain_type"_.Bind(-1), "test_osl_state"_.Bind(-1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:act_reg"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:pain"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:sparse"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:terrain"_.Bind(Spec<mjtNum>({-1}, {0.0, 4.0})),
        "info:time"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:root_pos"_.Bind(Spec<mjtNum>({2})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using RunTrackPixelFns = PixelObservationEnvFns<MyoChallengeRunTrackEnvFns>;
using MyoChallengeRunTrackEnvSpec = EnvSpec<MyoChallengeRunTrackEnvFns>;
using MyoChallengeRunTrackPixelEnvSpec = EnvSpec<RunTrackPixelFns>;

class MyoChallengeSoccerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0), "reset_type"_.Bind(std::string("none")),
        "min_agent_spawn_distance"_.Bind(1.0), "random_vel_low"_.Bind(1.0),
        "random_vel_high"_.Bind(5.0), "rnd_pos_noise"_.Bind(1.0),
        "rnd_joint_noise"_.Bind(0.02),
        "goalkeeper_probabilities"_.Bind(std::vector<double>{0.1, 0.45, 0.45}),
        "max_time_sec"_.Bind(10.0), "reward_goal_scored_w"_.Bind(1000.0),
        "reward_time_cost_w"_.Bind(-0.01), "reward_act_reg_w"_.Bind(-100.0),
        "reward_pain_w"_.Bind(-10.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_ctrl"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_act_dot"_.Bind(std::vector<double>{}),
        "test_reset_qacc"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_goalkeeper_pose"_.Bind(std::vector<double>{}),
        "test_goalkeeper_velocity"_.Bind(std::vector<double>{}),
        "test_goalkeeper_noise_buffer"_.Bind(std::vector<double>{}),
        "test_goalkeeper_noise_idx"_.Bind(-1),
        "test_goalkeeper_block_velocity"_.Bind(-1.0),
        "test_goalkeeper_policy"_.Bind(-1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:goal_scored"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:time_cost"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:act_reg"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:pain"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:sparse"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:goalkeeper_pos"_.Bind(Spec<mjtNum>({2})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using SoccerPixelFns = PixelObservationEnvFns<MyoChallengeSoccerEnvFns>;
using MyoChallengeSoccerEnvSpec = EnvSpec<MyoChallengeSoccerEnvFns>;
using MyoChallengeSoccerPixelEnvSpec = EnvSpec<SoccerPixelFns>;

class MyoChallengeChaseTagEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0), "reset_type"_.Bind(std::string("init")),
        "win_distance"_.Bind(0.5), "min_spawn_distance"_.Bind(2.0),
        "task_choice"_.Bind(std::string("CHASE")),
        "terrain"_.Bind(std::string("FLAT")), "repeller_opponent"_.Bind(false),
        "hills_range"_.Bind(std::vector<double>{0.0, 0.0}),
        "rough_range"_.Bind(std::vector<double>{0.0, 0.0}),
        "relief_range"_.Bind(std::vector<double>{0.0, 0.0}),
        "chase_vel_low"_.Bind(1.0), "chase_vel_high"_.Bind(1.0),
        "random_vel_low"_.Bind(-2.0), "random_vel_high"_.Bind(2.0),
        "repeller_vel_low"_.Bind(0.3), "repeller_vel_high"_.Bind(1.0),
        "opponent_probabilities"_.Bind(std::vector<double>{0.1, 0.45, 0.45}),
        "reward_distance_w"_.Bind(-0.1), "reward_lose_w"_.Bind(-1000.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_ctrl"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_act_dot"_.Bind(std::vector<double>{}),
        "test_reset_qacc"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_hfield_data"_.Bind(std::vector<double>{}),
        "test_terrain_geom_rgba"_.Bind(std::vector<double>{}),
        "test_terrain_geom_pos"_.Bind(std::vector<double>{}),
        "test_task"_.Bind(-1),
        "test_opponent_pose"_.Bind(std::vector<double>{}),
        "test_opponent_velocity"_.Bind(std::vector<double>{}),
        "test_opponent_noise_buffer"_.Bind(std::vector<double>{}),
        "test_opponent_noise_idx"_.Bind(-1), "test_chase_velocity"_.Bind(-1.0),
        "test_opponent_policy"_.Bind(-1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:distance"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:lose"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:sparse"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:task"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:opponent_pose"_.Bind(Spec<mjtNum>({3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using ChaseTagPixelFns = PixelObservationEnvFns<MyoChallengeChaseTagEnvFns>;
using MyoChallengeChaseTagEnvSpec = EnvSpec<MyoChallengeChaseTagEnvFns>;
using MyoChallengeChaseTagPixelEnvSpec = EnvSpec<ChaseTagPixelFns>;

class MyoChallengeTableTennisEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0), "ball_xyz_low"_.Bind(std::vector<double>{}),
        "ball_xyz_high"_.Bind(std::vector<double>{}), "ball_qvel"_.Bind(false),
        "ball_friction_low"_.Bind(std::vector<double>{}),
        "ball_friction_high"_.Bind(std::vector<double>{}),
        "paddle_mass_low"_.Bind(0.0), "paddle_mass_high"_.Bind(0.0),
        "qpos_noise_low"_.Bind(std::numeric_limits<double>::quiet_NaN()),
        "qpos_noise_high"_.Bind(std::numeric_limits<double>::quiet_NaN()),
        "rally_count"_.Bind(1), "reward_reach_dist_w"_.Bind(1.0),
        "reward_palm_dist_w"_.Bind(1.0), "reward_paddle_quat_w"_.Bind(2.0),
        "reward_act_reg_w"_.Bind(0.5), "reward_torso_up_w"_.Bind(2.0),
        "reward_sparse_w"_.Bind(100.0), "reward_solved_w"_.Bind(1000.0),
        "reward_done_w"_.Bind(-10.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_act_dot"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_ball_body_pos"_.Bind(std::vector<double>{}),
        "test_ball_geom_friction"_.Bind(std::vector<double>{}),
        "test_paddle_body_mass"_.Bind(std::vector<double>{}),
        "test_init_qpos"_.Bind(std::vector<double>{}),
        "test_init_qvel"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:reach_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:palm_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:paddle_quat"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:torso_up"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:act_reg"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:sparse"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:touching_info"_.Bind(Spec<mjtNum>({6}, {0.0, inf})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using TablePixelFns = PixelObservationEnvFns<MyoChallengeTableTennisEnvFns>;
using MyoChallengeTableTennisEnvSpec = EnvSpec<MyoChallengeTableTennisEnvFns>;
using MyoChallengeTableTennisPixelEnvSpec = EnvSpec<TablePixelFns>;

template <typename EnvSpecT, bool kFromPixels>
class MyoChallengeRunTrackEnvBase : public Env<EnvSpecT>,
                                    public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum act_reg{0.0};
    mjtNum pain{0.0};
    mjtNum sparse{0.0};
    bool solved{false};
    bool done{false};
  };

  bool normalize_act_;
  int ctrl_dim_;
  std::string reset_type_;
  std::string terrain_;
  mjtNum start_pos_;
  mjtNum end_pos_;
  mjtNum real_width_;
  mjtNum reward_sparse_w_;
  mjtNum reward_solved_w_;
  std::vector<mjtNum> hills_difficulties_;
  std::vector<mjtNum> rough_difficulties_;
  std::vector<mjtNum> stairs_difficulties_;
  int terrain_geom_id_{-1};
  int terrain_hfield_id_{-1};
  int pelvis_body_id_{-1};
  int head_site_id_{-1};
  int foot_l_body_id_{-1};
  int foot_r_body_id_{-1};
  std::array<int, 4> foot_height_site_ids_{-1, -1, -1, -1};
  std::array<int, 2> grf_sensor_ids_{-1, -1};
  int osl_load_sensor_id_{-1};
  int osl_knee_torque_actuator_id_{-1};
  int osl_ankle_torque_actuator_id_{-1};
  int osl_knee_qposadr_{-1};
  int osl_ankle_qposadr_{-1};
  int osl_knee_dofadr_{-1};
  int osl_ankle_dofadr_{-1};
  std::vector<int> biological_qposadrs_;
  std::vector<int> biological_dofadrs_;
  std::vector<int> biological_actuator_ids_;
  std::vector<int> pain_dofadrs_;
  std::vector<mjtNum> initial_hfield_data_;
  std::vector<mjtNum> initial_terrain_rgba_;
  std::vector<mjtNum> initial_terrain_pos_;
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;
  challenge_extra_detail::RunTrackOslController osl_controller_;
  int terrain_type_{0};
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_act_dot_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_hfield_data_;
  std::vector<mjtNum> test_terrain_geom_rgba_;
  std::vector<mjtNum> test_terrain_geom_pos_;
  int test_terrain_type_{-1};
  int test_osl_state_{-1};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoChallengeRunTrackEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        ctrl_dim_(spec.config["ctrl_dim"_]),
        reset_type_(spec.config["reset_type"_]),
        terrain_(spec.config["terrain"_]),
        start_pos_(spec.config["start_pos"_]),
        end_pos_(spec.config["end_pos"_]),
        real_width_(spec.config["real_width"_]),
        reward_sparse_w_(spec.config["reward_sparse_w"_]),
        reward_solved_w_(spec.config["reward_solved_w"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_hfield_data_(
            detail::ToMjtVector(spec.config["test_hfield_data"_])),
        test_terrain_geom_rgba_(
            detail::ToMjtVector(spec.config["test_terrain_geom_rgba"_])),
        test_terrain_geom_pos_(
            detail::ToMjtVector(spec.config["test_terrain_geom_pos"_])),
        test_terrain_type_(spec.config["test_terrain_type"_]),
        test_osl_state_(spec.config["test_osl_state"_]) {
    hills_difficulties_ =
        detail::ToMjtVector(spec.config["hills_difficulties"_]);
    rough_difficulties_ =
        detail::ToMjtVector(spec.config["rough_difficulties"_]);
    stairs_difficulties_ =
        detail::ToMjtVector(spec.config["stairs_difficulties"_]);
    std::reverse(hills_difficulties_.begin(), hills_difficulties_.end());
    std::reverse(rough_difficulties_.begin(), rough_difficulties_.end());
    std::reverse(stairs_difficulties_.begin(), stairs_difficulties_.end());
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    mjtNum total_body_mass = 0.0;
    for (int body = 0; body < model_->nbody; ++body) {
      total_body_mass += model_->body_mass[body];
    }
    osl_controller_.Initialize(total_body_mass);
    InitializeRobotEnv();
  }

  envpool::mujoco::CameraPolicy RenderCameraPolicy() const override {
    return detail::MyoSuiteRenderCameraPolicy();
  }

  void ConfigureRenderOption(mjvOption* option) const override {
    detail::ConfigureMyoSuiteRenderOptions(option, true);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    RestoreTrackTerrainState();
    ApplyTrackTerrainReset();
    ApplyResetState();
    InvalidateRenderCache();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    std::vector<float> ctrl = BuildAction(raw);
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                ctrl.data());
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
    mj_forward(model_, data_);
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->na != spec_.config["act_dim"_] || ctrl_dim_ != model_->nu ||
        spec_.config["action_dim"_] != model_->na) {
      throw std::runtime_error("RunTrack dims do not match model.");
    }
  }

  void CacheIds() {
    pelvis_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "pelvis");
    head_site_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "head");
    foot_l_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "talus_l");
    foot_r_body_id_ = challenge_extra_detail::RequireId(model_, mjOBJ_BODY,
                                                        "osl_foot_assembly");
    grf_sensor_ids_[0] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "l_foot");
    grf_sensor_ids_[1] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "l_toes");
    terrain_geom_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_GEOM, "terrain");
    terrain_hfield_id_ = model_->geom_dataid[terrain_geom_id_];
    osl_load_sensor_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "r_osl_load");
    osl_knee_torque_actuator_id_ = challenge_extra_detail::RequireId(
        model_, mjOBJ_ACTUATOR, "osl_knee_torque_actuator");
    osl_ankle_torque_actuator_id_ = challenge_extra_detail::RequireId(
        model_, mjOBJ_ACTUATOR, "osl_ankle_torque_actuator");
    int osl_knee_joint_id = challenge_extra_detail::RequireId(
        model_, mjOBJ_JOINT, "osl_knee_angle_r");
    int osl_ankle_joint_id = challenge_extra_detail::RequireId(
        model_, mjOBJ_JOINT, "osl_ankle_angle_r");
    osl_knee_qposadr_ = model_->jnt_qposadr[osl_knee_joint_id];
    osl_ankle_qposadr_ = model_->jnt_qposadr[osl_ankle_joint_id];
    osl_knee_dofadr_ = model_->jnt_dofadr[osl_knee_joint_id];
    osl_ankle_dofadr_ = model_->jnt_dofadr[osl_ankle_joint_id];
    foot_height_site_ids_[0] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "r_heel_btm");
    foot_height_site_ids_[1] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "r_toe_btm");
    foot_height_site_ids_[2] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "l_heel_btm");
    foot_height_site_ids_[3] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "l_toe_btm");
    static const std::vector<std::string> biological_joints = {
        "hip_adduction_l",
        "hip_flexion_l",
        "hip_rotation_l",
        "hip_adduction_r",
        "hip_flexion_r",
        "hip_rotation_r",
        "knee_angle_l",
        "knee_angle_l_beta_rotation1",
        "knee_angle_l_beta_translation1",
        "knee_angle_l_beta_translation2",
        "knee_angle_l_rotation2",
        "knee_angle_l_rotation3",
        "knee_angle_l_translation1",
        "knee_angle_l_translation2",
        "mtp_angle_l",
        "ankle_angle_l",
        "subtalar_angle_l",
    };
    static const std::vector<std::string> pain_joints = {
        "hip_adduction_l", "hip_adduction_r",        "hip_flexion_l",
        "hip_flexion_r",   "hip_rotation_l",         "hip_rotation_r",
        "knee_angle_l",    "knee_angle_l_rotation2", "knee_angle_l_rotation3",
        "mtp_angle_l",     "ankle_angle_l",          "subtalar_angle_l",
    };
    static const std::vector<std::string> biological_actuators = {
        "addbrev_l",    "addbrev_r",    "addlong_l",   "addlong_r",
        "addmagDist_l", "addmagIsch_l", "addmagMid_l", "addmagProx_l",
        "bflh_l",       "bfsh_l",       "edl_l",       "ehl_l",
        "fdl_l",        "fhl_l",        "gaslat_l",    "gasmed_l",
        "glmax1_l",     "glmax1_r",     "glmax2_l",    "glmax2_r",
        "glmax3_l",     "glmax3_r",     "glmed1_l",    "glmed1_r",
        "glmed2_l",     "glmed2_r",     "glmed3_l",    "glmed3_r",
        "glmin1_l",     "glmin1_r",     "glmin2_l",    "glmin2_r",
        "glmin3_l",     "glmin3_r",     "grac_l",      "iliacus_l",
        "iliacus_r",    "perbrev_l",    "perlong_l",   "piri_l",
        "piri_r",       "psoas_l",      "psoas_r",     "recfem_l",
        "sart_l",       "semimem_l",    "semiten_l",   "soleus_l",
        "tfl_l",        "tibant_l",     "tibpost_l",   "vasint_l",
        "vaslat_l",     "vasmed_l",
    };
    biological_qposadrs_ =
        challenge_extra_detail::CollectJointQposAdrs(model_, biological_joints);
    biological_dofadrs_ =
        challenge_extra_detail::CollectJointDofAdrs(model_, biological_joints);
    for (const std::string& actuator_name : biological_actuators) {
      biological_actuator_ids_.push_back(challenge_extra_detail::RequireId(
          model_, mjOBJ_ACTUATOR, actuator_name));
    }
    pain_dofadrs_ =
        challenge_extra_detail::CollectJointDofAdrs(model_, pain_joints);
    detail::CopyModelGeomRgba(model_, terrain_geom_id_, &initial_terrain_rgba_);
    detail::CopyModelGeomPos(model_, terrain_geom_id_, &initial_terrain_pos_);
    if (terrain_hfield_id_ >= 0) {
      detail::CopyModelHfieldData(model_, terrain_hfield_id_,
                                  &initial_hfield_data_);
    }
    int expected_obs = static_cast<int>(biological_qposadrs_.size()) +
                       static_cast<int>(biological_dofadrs_.size()) + 2 + 4 +
                       model_->na + model_->na + model_->na + model_->na + 2 +
                       2;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("RunTrack obs_dim does not match model.");
    }
  }

  void RestoreTrackTerrainState() {
    detail::RestoreModelGeomRgba(model_, terrain_geom_id_,
                                 initial_terrain_rgba_);
    detail::RestoreModelGeomPos(model_, terrain_geom_id_, initial_terrain_pos_);
    if (terrain_hfield_id_ >= 0) {
      detail::RestoreModelHfieldData(model_, terrain_hfield_id_,
                                     initial_hfield_data_);
    }
  }

  void SetTrackTerrainVisible() {
    std::vector<mjtNum> rgba = initial_terrain_rgba_;
    std::vector<mjtNum> pos = initial_terrain_pos_;
    rgba[3] = 1.0;
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = 0.005;
    detail::RestoreModelGeomRgba(model_, terrain_geom_id_, rgba);
    detail::RestoreModelGeomPos(model_, terrain_geom_id_, pos);
  }

  void ClearTrackTerrain() {
    if (terrain_hfield_id_ < 0) {
      return;
    }
    int rows = model_->hfield_nrow[terrain_hfield_id_];
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    detail::RestoreModelHfieldData(
        model_, terrain_hfield_id_,
        std::vector<mjtNum>(rows * cols, static_cast<mjtNum>(0.0)));
  }

  void FillStairsTrackPatch(int patch_start, int patch_end, int index) {
    int rows = patch_end - patch_start;
    if (rows <= 0) {
      return;
    }
    int stair_height_count = 3;
    int stair_count = stair_height_count * 2;
    int stair_flat = rows / stair_count;
    mjtNum stair_height = stairs_difficulties_.at(index);
    int filled_rows = stair_flat * stair_count;
    int adr = model_->hfield_adr[terrain_hfield_id_];
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    mjtNum height = 0.0;
    for (int stair = 0; stair < stair_count; ++stair) {
      for (int row = 0; row < stair_flat; ++row) {
        int global_row = patch_start + stair * stair_flat + row;
        if (global_row >= patch_start + filled_rows) {
          break;
        }
        for (int col = 0; col < cols; ++col) {
          model_->hfield_data[adr + global_row * cols + col] =
              static_cast<float>(height);
        }
      }
      if (stair < stair_count / 2) {
        height += stair_height;
      } else {
        height -= stair_height;
      }
    }
  }

  void FillHillyTrackPatch(int patch_start, int patch_end, int index) {
    int rows = patch_end - patch_start;
    if (rows <= 0) {
      return;
    }
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    int total = rows * cols;
    std::vector<mjtNum> data(total);
    for (int i = 0; i < total; ++i) {
      data[i] = static_cast<mjtNum>(std::sin(
          static_cast<double>(i) / std::max(total - 1, 1) * detail::kPi));
    }
    mjtNum low = *std::min_element(data.begin(), data.end());
    mjtNum high = *std::max_element(data.begin(), data.end());
    mjtNum denom = std::max(high - low, static_cast<mjtNum>(1e-12));
    mjtNum scalar = hills_difficulties_.at(index);
    int adr = model_->hfield_adr[terrain_hfield_id_];
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        int src = row * cols + col;
        int dst_row = rows - 1 - row;
        int dst_col = cols - 1 - col;
        mjtNum value = (data[src] - low) / denom * scalar;
        model_->hfield_data[adr + (patch_start + dst_row) * cols + dst_col] =
            static_cast<float>(value);
      }
    }
  }

  void FillRoughTrackPatch(int patch_start, int patch_end, int index) {
    int rows = patch_end - patch_start;
    if (rows <= 0) {
      return;
    }
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    std::vector<mjtNum> fill_data(rows * cols);
    for (mjtNum& value : fill_data) {
      value = std::uniform_real_distribution<double>(-1.0, 1.0)(gen_);
    }
    mjtNum scalar = std::uniform_real_distribution<double>(
        0.0, static_cast<double>(rough_difficulties_.at(index)))(gen_);
    mjtNum low = *std::min_element(fill_data.begin(), fill_data.end());
    mjtNum high = *std::max_element(fill_data.begin(), fill_data.end());
    mjtNum denom = std::max(high - low, static_cast<mjtNum>(1e-12));
    int adr = model_->hfield_adr[terrain_hfield_id_];
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        mjtNum normalized = (fill_data[row * cols + col] - low) / denom;
        model_->hfield_data[adr + (patch_start + row) * cols + col] =
            static_cast<float>(normalized * scalar);
      }
    }
  }

  void ApplyTrackTerrainReset() {
    SetTrackTerrainVisible();
    ClearTrackTerrain();
    if (test_terrain_type_ >= 0) {
      terrain_type_ = test_terrain_type_;
    }
    if (!test_terrain_geom_rgba_.empty()) {
      detail::RestoreModelGeomRgba(model_, terrain_geom_id_,
                                   test_terrain_geom_rgba_);
    }
    if (!test_terrain_geom_pos_.empty()) {
      detail::RestoreModelGeomPos(model_, terrain_geom_id_,
                                  test_terrain_geom_pos_);
    }
    if (!test_hfield_data_.empty()) {
      detail::RestoreModelHfieldData(model_, terrain_hfield_id_,
                                     test_hfield_data_);
      return;
    }
    terrain_type_ = 0;
    if (terrain_ == "flat") {
      return;
    }
    if (terrain_ != "random") {
      throw std::runtime_error("Unsupported RunTrack terrain reset type.");
    }
    using TerrainFn = void (MyoChallengeRunTrackEnvBase::*)(int, int, int);
    std::vector<std::pair<TerrainFn, int>> terrain_fns = {
        {&MyoChallengeRunTrackEnvBase::FillStairsTrackPatch, 3},
        {&MyoChallengeRunTrackEnvBase::FillHillyTrackPatch, 1},
        {&MyoChallengeRunTrackEnvBase::FillRoughTrackPatch, 2},
    };
    int terrain_choice = std::uniform_int_distribution<int>(
        0, static_cast<int>(terrain_fns.size()) - 1)(gen_);
    terrain_type_ = terrain_fns[terrain_choice].second;
    const std::vector<mjtNum>* difficulties = nullptr;
    switch (terrain_type_) {
      case 1:
        difficulties = &hills_difficulties_;
        break;
      case 2:
        difficulties = &rough_difficulties_;
        break;
      case 3:
        difficulties = &stairs_difficulties_;
        break;
      default:
        return;
    }
    if (difficulties->size() < 2) {
      return;
    }
    int rows = model_->hfield_nrow[terrain_hfield_id_];
    int patch_rows = rows / static_cast<int>(difficulties->size());
    if (patch_rows <= 0) {
      return;
    }
    int patch_start = 0;
    for (std::size_t i = 0; i + 1 < difficulties->size(); ++i) {
      int patch_end = patch_start + patch_rows;
      (this->*terrain_fns[terrain_choice].first)(
          patch_start, std::min(patch_end, rows), static_cast<int>(i));
      patch_start = patch_end;
    }
  }

  std::pair<std::vector<mjtNum>, std::vector<mjtNum>> RandomizedResetState() {
    int key = std::uniform_int_distribution<int>(0, 2)(gen_);
    std::vector<mjtNum> qpos(model_->key_qpos + key * model_->nq,
                             model_->key_qpos + (key + 1) * model_->nq);
    std::vector<mjtNum> qvel(model_->key_qvel + key * model_->nv,
                             model_->key_qvel + (key + 1) * model_->nv);
    osl_controller_.Reset(
        key == 1 ? challenge_extra_detail::RunTrackOslState::kEarlySwing
                 : challenge_extra_detail::RunTrackOslState::kEarlyStance);
    std::uniform_real_distribution<double> x_dist(
        -static_cast<double>(real_width_) * 0.8,
        static_cast<double>(real_width_) * 0.8);
    qpos[0] = static_cast<mjtNum>(x_dist(gen_));
    qpos[1] = start_pos_ + 1.0;
    return {qpos, qvel};
  }

  void AdjustModelHeight() {
    mjtNum min_height = std::numeric_limits<mjtNum>::infinity();
    for (int site_id : foot_height_site_ids_) {
      min_height = std::min(min_height, data_->site_xpos[site_id * 3 + 2]);
    }
    data_->qpos[2] += static_cast<mjtNum>(0.005) - min_height;
  }

  challenge_extra_detail::RunTrackOslSensorData OslSensorData() const {
    int sensor_adr = model_->sensor_adr[osl_load_sensor_id_];
    return {
        data_->qpos[osl_knee_qposadr_],     data_->qvel[osl_knee_dofadr_],
        data_->qpos[osl_ankle_qposadr_],    data_->qvel[osl_ankle_dofadr_],
        -data_->sensordata[sensor_adr + 1],
    };
  }

  std::vector<float> BuildAction(const float* raw) {
    std::vector<float> ctrl(model_->nu, 0.0f);
    std::copy(raw, raw + model_->na, ctrl.begin());
    osl_controller_.Update(OslSensorData());
    auto torques = osl_controller_.GetTorques();
    const std::array<std::pair<int, mjtNum>, 2> actuator_data = {{
        {osl_knee_torque_actuator_id_, torques[0]},
        {osl_ankle_torque_actuator_id_, torques[1]},
    }};
    for (const auto& [actuator_id, torque] : actuator_data) {
      mjtNum ctrl_value = torque / model_->actuator_gear[actuator_id * 6];
      mjtNum min_ctrl = model_->actuator_ctrlrange[actuator_id * 2];
      mjtNum max_ctrl = model_->actuator_ctrlrange[actuator_id * 2 + 1];
      ctrl_value = challenge_extra_detail::Clip(ctrl_value, min_ctrl, max_ctrl);
      if (normalize_act_) {
        mjtNum ctrl_mean = (min_ctrl + max_ctrl) * static_cast<mjtNum>(0.5);
        mjtNum ctrl_rng = (max_ctrl - min_ctrl) * static_cast<mjtNum>(0.5);
        ctrl_value = (ctrl_value - ctrl_mean) / ctrl_rng;
      }
      ctrl[actuator_id] = static_cast<float>(ctrl_value);
    }
    return ctrl;
  }

  void ResetSimulationState(const std::vector<mjtNum>& qpos,
                            const std::vector<mjtNum>& qvel) {
    mj_resetData(model_, data_);
    data_->time = initial_time_;
    detail::RestoreVector(qpos, data_->qpos);
    detail::RestoreVector(qvel, data_->qvel);
    if (model_->nu != 0) {
      mju_zero(data_->ctrl, model_->nu);
    }
    if (model_->na != 0) {
      mju_zero(data_->act, model_->na);
    }
    mj_forward(model_, data_);
  }

  void ApplyResetState() {
    osl_controller_.Reset(
        challenge_extra_detail::RunTrackOslState::kEarlyStance);
    std::vector<mjtNum> qpos;
    std::vector<mjtNum> qvel;
    if (!test_reset_qpos_.empty()) {
      qpos = test_reset_qpos_;
      qvel = test_reset_qvel_;
    } else if (reset_type_ == "random") {
      std::tie(qpos, qvel) = RandomizedResetState();
    } else {
      osl_controller_.Reset(
          challenge_extra_detail::RunTrackOslState::kEarlyStance);
      qpos.assign(model_->key_qpos, model_->key_qpos + model_->nq);
      qvel.assign(model_->key_qvel, model_->key_qvel + model_->nv);
    }

    ResetSimulationState(qpos, qvel);
    if (test_reset_qpos_.empty() && reset_type_ != "init") {
      AdjustModelHeight();
      qpos.assign(data_->qpos, data_->qpos + model_->nq);
      qvel.assign(data_->qvel, data_->qvel + model_->nv);
      ResetSimulationState(qpos, qvel);
    }
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
      mj_forward(model_, data_);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
#ifdef ENVPOOL_TEST
      mjtNum max_diff = 0.0;
      for (int i = 0; i < model_->nv; ++i) {
        max_diff = std::max(max_diff, std::abs(data_->qacc_warmstart[i] -
                                               test_reset_qacc_warmstart_[i]));
      }
      std::fprintf(stderr, "RunTrack qacc_warmstart restore max diff: %.17g\n",
                   static_cast<double>(max_diff));
#endif
    }
    if (test_osl_state_ >= 0) {
      osl_controller_.SetState(
          static_cast<challenge_extra_detail::RunTrackOslState>(
              test_osl_state_));
    }
    osl_controller_.Start();
  }

  bool Fallen() const {
    mjtNum head_z = data_->site_xpos[head_site_id_ * 3 + 2];
    mjtNum left_z = data_->xpos[foot_l_body_id_ * 3 + 2];
    mjtNum right_z = data_->xpos[foot_r_body_id_ * 3 + 2];
    mjtNum mean_feet_z = (left_z + right_z) * 0.5;
    return head_z - mean_feet_z < 0.2 || head_z < 0.2;
  }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    reward.act_reg = challenge_extra_detail::MeanSquareAct(model_, data_);
    reward.pain = challenge_extra_detail::AverageJointLimitForce(model_, data_,
                                                                 pain_dofadrs_);
    mjtNum x_pos = data_->qpos[0];
    mjtNum y_pos = data_->qpos[1];
    mjtNum y_vel = data_->qvel[1];
    reward.sparse = -y_vel;
    reward.solved = y_pos < end_pos_;
    reward.done = x_pos > real_width_ || x_pos < -real_width_ ||
                  y_pos > start_pos_ + 2.0 || Fallen() || reward.solved;
    reward.dense_reward = reward_sparse_w_ * reward.sparse +
                          reward_solved_w_ * static_cast<mjtNum>(reward.solved);
    return reward;
  }

  std::vector<mjtNum> Observation() const {
    std::vector<mjtNum> obs;
    obs.reserve(spec_.config["obs_dim"_]);
    for (int adr : biological_qposadrs_) {
      obs.push_back(data_->qpos[adr]);
    }
    for (int adr : biological_dofadrs_) {
      obs.push_back(data_->qvel[adr] * Dt());
    }
    for (int sensor_id : grf_sensor_ids_) {
      int adr = model_->sensor_adr[sensor_id];
      obs.push_back(data_->sensordata[adr]);
    }
    obs.insert(obs.end(), data_->xquat + pelvis_body_id_ * 4,
               data_->xquat + pelvis_body_id_ * 4 + 4);
    obs.push_back(data_->qpos[0]);
    obs.push_back(data_->qpos[1]);
    obs.push_back(data_->qvel[0]);
    obs.push_back(data_->qvel[1]);
    for (int actuator_id : biological_actuator_ids_) {
      obs.push_back(data_->actuator_length[actuator_id]);
    }
    for (int actuator_id : biological_actuator_ids_) {
      obs.push_back(challenge_extra_detail::Clip(
          data_->actuator_velocity[actuator_id], -100.0, 100.0));
    }
    for (int actuator_id : biological_actuator_ids_) {
      obs.push_back(challenge_extra_detail::Clip(
          data_->actuator_force[actuator_id] / 1000.0, -100.0, 100.0));
    }
    obs.insert(obs.end(), data_->act, data_->act + model_->na);
    return obs;
  }

  void WriteState(const RewardInfo& reward, bool reset, float reward_value) {
    auto obs = Observation();
    auto state = Allocate();
    if constexpr (!kFromPixels) {
      AssignObservation("obs", &state["obs"_], obs.data(), obs.size(), reset);
    }
    state["reward"_] = reward_value;
    state["discount"_] = 1.0f;
    state["done"_] = done_;
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    state["info:act_reg"_] = reward.act_reg;
    state["info:pain"_] = reward.pain;
    state["info:sparse"_] = reward.sparse;
    state["info:solved"_] = static_cast<mjtNum>(reward.solved);
    state["info:terrain"_] = static_cast<mjtNum>(terrain_type_);
    state["info:time"_] = data_->time;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    std::array<mjtNum, 2> root_pos = {data_->qpos[0], data_->qpos[1]};
    state["info:root_pos"_].Assign(root_pos.data(), 2);
    if constexpr (kFromPixels) {
      AssignPixelObservation("obs:pixels", &state["obs:pixels"_], reset);
    }
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoChallengeSoccerEnvBase : public Env<EnvSpecT>,
                                  public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum goal_scored{0.0};
    mjtNum time_cost{0.0};
    mjtNum act_reg{0.0};
    mjtNum pain{0.0};
    mjtNum sparse{0.0};
    bool solved{false};
    bool done{false};
  };

  static constexpr mjtNum kGoalX = 50.0;
  static constexpr mjtNum kGoalYMin = -3.3;
  static constexpr mjtNum kGoalYMax = 3.3;
  static constexpr mjtNum kGoalZMin = 0.0;
  static constexpr mjtNum kGoalZMax = 2.2;
  static constexpr int kGoalkeeperBlockBall = 0;
  static constexpr int kGoalkeeperRandom = 1;
  static constexpr int kGoalkeeperStationary = 2;

  bool normalize_act_;
  std::string reset_type_;
  mjtNum min_agent_spawn_distance_;
  mjtNum random_vel_low_;
  mjtNum random_vel_high_;
  mjtNum rnd_pos_noise_;
  mjtNum rnd_joint_noise_;
  mjtNum max_time_sec_;
  mjtNum reward_goal_scored_w_;
  mjtNum reward_time_cost_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_pain_w_;
  std::array<mjtNum, 3> goalkeeper_probabilities_{0.1, 0.45, 0.45};
  int soccer_ball_body_id_{-1};
  int goalkeeper_body_id_{-1};
  int goalkeeper_mocap_id_{-1};
  int pelvis_body_id_{-1};
  int torso_body_id_{-1};
  int root_joint_id_{-1};
  std::array<int, 4> grf_sensor_ids_{-1, -1, -1, -1};
  std::vector<int> internal_qposadrs_;
  std::vector<int> internal_dofadrs_;
  std::vector<int> pain_dofadrs_;
  std::vector<int> muscle_actuator_ids_;
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;
  challenge_extra_detail::ColoredNoiseProcess goalkeeper_noise_process_{2.0, 2,
                                                                        10};
  int goalkeeper_policy_{kGoalkeeperStationary};
  std::array<mjtNum, 2> goalkeeper_velocity_{0.0, 0.0};
  mjtNum goalkeeper_block_velocity_{1.0};
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_ctrl_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_act_dot_;
  std::vector<mjtNum> test_reset_qacc_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_goalkeeper_pose_;
  std::vector<mjtNum> test_goalkeeper_velocity_;
  std::vector<mjtNum> test_goalkeeper_noise_buffer_;
  int test_goalkeeper_noise_idx_{-1};
  mjtNum test_goalkeeper_block_velocity_{-1.0};
  int test_goalkeeper_policy_{-1};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoChallengeSoccerEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        reset_type_(spec.config["reset_type"_]),
        min_agent_spawn_distance_(spec.config["min_agent_spawn_distance"_]),
        random_vel_low_(spec.config["random_vel_low"_]),
        random_vel_high_(spec.config["random_vel_high"_]),
        rnd_pos_noise_(spec.config["rnd_pos_noise"_]),
        rnd_joint_noise_(spec.config["rnd_joint_noise"_]),
        max_time_sec_(spec.config["max_time_sec"_]),
        reward_goal_scored_w_(spec.config["reward_goal_scored_w"_]),
        reward_time_cost_w_(spec.config["reward_time_cost_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        reward_pain_w_(spec.config["reward_pain_w"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_ctrl_(detail::ToMjtVector(spec.config["test_reset_ctrl"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_act_dot_(
            detail::ToMjtVector(spec.config["test_reset_act_dot"_])),
        test_reset_qacc_(detail::ToMjtVector(spec.config["test_reset_qacc"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_goalkeeper_pose_(
            detail::ToMjtVector(spec.config["test_goalkeeper_pose"_])),
        test_goalkeeper_velocity_(
            detail::ToMjtVector(spec.config["test_goalkeeper_velocity"_])),
        test_goalkeeper_noise_buffer_(
            detail::ToMjtVector(spec.config["test_goalkeeper_noise_buffer"_])),
        test_goalkeeper_noise_idx_(spec.config["test_goalkeeper_noise_idx"_]),
        test_goalkeeper_block_velocity_(
            spec.config["test_goalkeeper_block_velocity"_]),
        test_goalkeeper_policy_(spec.config["test_goalkeeper_policy"_]) {
    auto probs = spec.config["goalkeeper_probabilities"_];
    if (probs.size() != 3) {
      throw std::runtime_error("Expected three goalkeeper probabilities.");
    }
    for (int i = 0; i < 3; ++i) {
      goalkeeper_probabilities_[i] = static_cast<mjtNum>(probs[i]);
    }
    goalkeeper_block_velocity_ = challenge_detail::UniformScalar(
        &gen_, random_vel_low_, random_vel_high_);
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    for (int actuator = 0; actuator < model_->nu; ++actuator) {
      if (muscle_actuator_[actuator]) {
        muscle_actuator_ids_.push_back(actuator);
      }
    }
    InitializeRobotEnv();
  }

  envpool::mujoco::CameraPolicy RenderCameraPolicy() const override {
    return detail::MyoSuiteRenderCameraPolicy();
  }

  void ConfigureRenderOption(mjvOption* option) const override {
    detail::ConfigureMyoSuiteRenderOptions(option, true);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    ApplyResetState();
    ResetGoalkeeper();
    ApplyGoalkeeperTestState();
    mj_forward(model_, data_);
    InvalidateRenderCache();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    UpdateGoalkeeperState();
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
    detail::RefreshObservedMyoSuiteState(model_, data_);
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->na != spec_.config["act_dim"_] ||
        model_->nu != spec_.config["action_dim"_]) {
      throw std::runtime_error("Soccer dims do not match model.");
    }
  }

  void CacheIds() {
    soccer_ball_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "soccer_ball");
    goalkeeper_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "goalkeeper");
    goalkeeper_mocap_id_ = model_->body_mocapid[goalkeeper_body_id_];
    pelvis_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "pelvis");
    torso_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "torso");
    root_joint_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_JOINT, "root");
    grf_sensor_ids_[0] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "r_foot");
    grf_sensor_ids_[1] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "r_toes");
    grf_sensor_ids_[2] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "l_foot");
    grf_sensor_ids_[3] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "l_toes");
    static const std::vector<std::string> kMyoJoints = {
        "Abs_r3",
        "Abs_t1",
        "Abs_t2",
        "L1_L2_AR",
        "L1_L2_FE",
        "L1_L2_LB",
        "L2_L3_AR",
        "L2_L3_FE",
        "L2_L3_LB",
        "L3_L4_AR",
        "L3_L4_FE",
        "L3_L4_LB",
        "L4_L5_AR",
        "L4_L5_FE",
        "L4_L5_LB",
        "ankle_angle_l",
        "ankle_angle_r",
        "axial_rotation",
        "flex_extension",
        "hip_adduction_l",
        "hip_adduction_r",
        "hip_flexion_l",
        "hip_flexion_r",
        "hip_rotation_l",
        "hip_rotation_r",
        "knee_angle_l",
        "knee_angle_l_beta_rotation1",
        "knee_angle_l_beta_translation1",
        "knee_angle_l_beta_translation2",
        "knee_angle_l_rotation2",
        "knee_angle_l_rotation3",
        "knee_angle_l_translation1",
        "knee_angle_l_translation2",
        "knee_angle_r",
        "knee_angle_r_beta_rotation1",
        "knee_angle_r_beta_translation1",
        "knee_angle_r_beta_translation2",
        "knee_angle_r_rotation2",
        "knee_angle_r_rotation3",
        "knee_angle_r_translation1",
        "knee_angle_r_translation2",
        "lat_bending",
        "mtp_angle_l",
        "mtp_angle_r",
        "subtalar_angle_l",
        "subtalar_angle_r",
    };
    internal_qposadrs_ =
        challenge_extra_detail::CollectJointQposAdrs(model_, kMyoJoints);
    internal_dofadrs_ =
        challenge_extra_detail::CollectJointDofAdrs(model_, kMyoJoints);
    static const std::vector<std::string> pain_joints = {
        "hip_adduction_l", "hip_adduction_r",        "hip_flexion_l",
        "hip_flexion_r",   "hip_rotation_l",         "hip_rotation_r",
        "knee_angle_l",    "knee_angle_l_rotation2", "knee_angle_l_rotation3",
        "mtp_angle_l",     "ankle_angle_l",          "subtalar_angle_l",
    };
    pain_dofadrs_ =
        challenge_extra_detail::CollectJointDofAdrs(model_, pain_joints);
    int expected_obs = static_cast<int>(internal_qposadrs_.size()) +
                       static_cast<int>(internal_dofadrs_.size()) + 4 + 4 + 3 +
                       7 + 6 + model_->na + model_->na + model_->na +
                       model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("Soccer obs_dim does not match model.");
    }
  }

  void SetGoalkeeperPose(mjtNum x, mjtNum y, mjtNum yaw) {
    y = challenge_extra_detail::Clip(y, kGoalYMin, kGoalYMax);
    data_->mocap_pos[goalkeeper_mocap_id_ * 3 + 0] = x;
    data_->mocap_pos[goalkeeper_mocap_id_ * 3 + 1] = y;
    data_->mocap_pos[goalkeeper_mocap_id_ * 3 + 2] = 0.0;
    auto quat = challenge_extra_detail::YawToQuat(yaw);
    std::memcpy(data_->mocap_quat + goalkeeper_mocap_id_ * 4, quat.data(),
                sizeof(mjtNum) * 4);
  }

  std::array<mjtNum, 3> GoalkeeperPose() const {
    std::array<mjtNum, 3> pose = {
        data_->mocap_pos[goalkeeper_mocap_id_ * 3 + 0],
        data_->mocap_pos[goalkeeper_mocap_id_ * 3 + 1],
        challenge_extra_detail::QuatYaw(data_->mocap_quat +
                                        goalkeeper_mocap_id_ * 4),
    };
    return pose;
  }

  void SampleGoalkeeperPolicy() {
    mjtNum u = challenge_detail::UniformScalar(&gen_, 0.0, 1.0);
    if (u < goalkeeper_probabilities_[0]) {
      goalkeeper_policy_ = kGoalkeeperBlockBall;
    } else if (u <
               goalkeeper_probabilities_[0] + goalkeeper_probabilities_[1]) {
      goalkeeper_policy_ = kGoalkeeperRandom;
    } else {
      goalkeeper_policy_ = kGoalkeeperStationary;
    }
  }

  void ResetGoalkeeper() {
    goalkeeper_noise_process_.SetScale(goalkeeper_block_velocity_);
    goalkeeper_noise_process_.Reset(&gen_);
    goalkeeper_velocity_ = {0.0, 0.0};
    SampleGoalkeeperPolicy();
    mjtNum y = challenge_detail::UniformScalar(&gen_, kGoalYMin, kGoalYMax);
    SetGoalkeeperPose(kGoalX, y, 0.0);
    goalkeeper_velocity_ = {0.0, 0.0};
    goalkeeper_block_velocity_ = challenge_detail::UniformScalar(
        &gen_, random_vel_low_, random_vel_high_);
  }

  void UpdateGoalkeeperState() {
    std::array<mjtNum, 2> command = {0.0, 0.0};
    if (goalkeeper_policy_ == kGoalkeeperBlockBall) {
      mjtNum ball_y = data_->xpos[soccer_ball_body_id_ * 3 + 1];
      command[0] = challenge_extra_detail::Clip(ball_y - GoalkeeperPose()[1],
                                                -goalkeeper_block_velocity_,
                                                goalkeeper_block_velocity_);
    } else if (goalkeeper_policy_ == kGoalkeeperRandom) {
      auto sample = goalkeeper_noise_process_.Sample(&gen_);
      command[0] = challenge_extra_detail::Clip(
          sample[0], -goalkeeper_block_velocity_, goalkeeper_block_velocity_);
      command[1] = challenge_extra_detail::Clip(
          sample[1], -goalkeeper_block_velocity_, goalkeeper_block_velocity_);
    }
    auto pose = GoalkeeperPose();
    pose[1] += Dt() * command[0];
    SetGoalkeeperPose(pose[0], pose[1], pose[2]);
    goalkeeper_velocity_ = command;
  }

  void ApplyGoalkeeperTestState() {
    if (test_goalkeeper_pose_.size() == 3) {
      SetGoalkeeperPose(test_goalkeeper_pose_[0], test_goalkeeper_pose_[1],
                        test_goalkeeper_pose_[2]);
    }
    if (test_goalkeeper_velocity_.size() == 2) {
      goalkeeper_velocity_[0] = test_goalkeeper_velocity_[0];
      goalkeeper_velocity_[1] = test_goalkeeper_velocity_[1];
    }
    if (!test_goalkeeper_noise_buffer_.empty()) {
      goalkeeper_noise_process_.SetState(test_goalkeeper_noise_buffer_,
                                         test_goalkeeper_noise_idx_);
    }
    if (test_goalkeeper_block_velocity_ >= 0.0) {
      goalkeeper_block_velocity_ = test_goalkeeper_block_velocity_;
    }
    if (test_goalkeeper_policy_ >= 0) {
      goalkeeper_policy_ = test_goalkeeper_policy_;
    }
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    } else {
      detail::RestoreVector(
          std::vector<mjtNum>(model_->key_qpos, model_->key_qpos + model_->nq),
          data_->qpos);
      detail::RestoreVector(
          std::vector<mjtNum>(model_->key_qvel, model_->key_qvel + model_->nv),
          data_->qvel);
      if (reset_type_ == "random") {
        for (int adr : internal_qposadrs_) {
          data_->qpos[adr] += challenge_detail::UniformScalar(
              &gen_, -std::abs(rnd_joint_noise_), std::abs(rnd_joint_noise_));
        }
        data_->qpos[model_->jnt_qposadr[root_joint_id_] + 0] +=
            challenge_detail::UniformScalar(&gen_, -rnd_pos_noise_, 0.0);
        data_->qpos[model_->jnt_qposadr[root_joint_id_] + 1] +=
            challenge_detail::UniformScalar(&gen_, -rnd_pos_noise_,
                                            rnd_pos_noise_);
      }
    }
    mj_forward(model_, data_);
    bool rerun_forward = false;
    if (!test_reset_ctrl_.empty()) {
      detail::RestoreVector(test_reset_ctrl_, data_->ctrl);
      rerun_forward = true;
    }
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
      rerun_forward = true;
    }
    if (!test_reset_act_dot_.empty()) {
      detail::RestoreVector(test_reset_act_dot_, data_->act_dot);
    }
    if (!test_reset_qacc_.empty()) {
      detail::RestoreVector(test_reset_qacc_, data_->qacc);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
      rerun_forward = true;
    }
    if (rerun_forward) {
      mj_forward(model_, data_);
    }
  }

  bool GoalScored() const {
    const mjtNum* ball_pos = data_->xpos + soccer_ball_body_id_ * 3;
    return ball_pos[0] >= kGoalX && ball_pos[1] >= kGoalYMin &&
           ball_pos[1] <= kGoalYMax && ball_pos[2] >= kGoalZMin &&
           ball_pos[2] <= kGoalZMax;
  }

  bool Fallen() const { return data_->xpos[pelvis_body_id_ * 3 + 2] < 0.2; }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    reward.goal_scored = GoalScored() ? 1.0 : 0.0;
    reward.time_cost = data_->time;
    reward.act_reg = challenge_extra_detail::MeanSquareAct(model_, data_);
    reward.pain = challenge_extra_detail::AverageJointLimitForce(model_, data_,
                                                                 pain_dofadrs_);
    reward.solved = reward.goal_scored > 0.0;
    reward.done = reward.solved || data_->time >= max_time_sec_ || Fallen();
    reward.sparse = reward.done ? 1.0 : 0.0;
    reward.dense_reward = reward_goal_scored_w_ * reward.goal_scored +
                          reward_time_cost_w_ * reward.time_cost +
                          reward_act_reg_w_ * reward.act_reg +
                          reward_pain_w_ * reward.pain;
    return reward;
  }

  std::vector<mjtNum> Observation() const {
    std::vector<mjtNum> obs;
    obs.reserve(spec_.config["obs_dim"_]);
    for (int adr : internal_qposadrs_) {
      obs.push_back(data_->qpos[adr]);
    }
    for (int adr : internal_dofadrs_) {
      obs.push_back(data_->qvel[adr] * Dt());
    }
    for (int sensor_id : grf_sensor_ids_) {
      obs.push_back(data_->sensordata[model_->sensor_adr[sensor_id]]);
    }
    obs.insert(obs.end(), data_->xquat + torso_body_id_ * 4,
               data_->xquat + torso_body_id_ * 4 + 4);
    obs.insert(obs.end(), data_->xpos + soccer_ball_body_id_ * 3,
               data_->xpos + soccer_ball_body_id_ * 3 + 3);
    int root_qposadr = model_->jnt_qposadr[root_joint_id_];
    int root_dofadr = model_->jnt_dofadr[root_joint_id_];
    obs.insert(obs.end(), data_->qpos + root_qposadr,
               data_->qpos + root_qposadr + 7);
    obs.insert(obs.end(), data_->qvel + root_dofadr,
               data_->qvel + root_dofadr + 6);
    for (int actuator : muscle_actuator_ids_) {
      obs.push_back(data_->actuator_length[actuator]);
    }
    for (int actuator : muscle_actuator_ids_) {
      obs.push_back(challenge_extra_detail::Clip(
          data_->actuator_velocity[actuator], -100.0, 100.0));
    }
    for (int actuator : muscle_actuator_ids_) {
      obs.push_back(challenge_extra_detail::Clip(
          data_->actuator_force[actuator] / 1000.0, -100.0, 100.0));
    }
    obs.insert(obs.end(), data_->act, data_->act + model_->na);
    return obs;
  }

  void WriteState(const RewardInfo& reward, bool reset, float reward_value) {
    auto obs = Observation();
    auto state = Allocate();
    if constexpr (!kFromPixels) {
      AssignObservation("obs", &state["obs"_], obs.data(), obs.size(), reset);
    }
    state["reward"_] = reward_value;
    state["discount"_] = 1.0f;
    state["done"_] = done_;
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    state["info:goal_scored"_] = reward.goal_scored;
    state["info:time_cost"_] = reward.time_cost;
    state["info:act_reg"_] = reward.act_reg;
    state["info:pain"_] = reward.pain;
    state["info:sparse"_] = reward.sparse;
    state["info:solved"_] = static_cast<mjtNum>(reward.solved);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    std::array<mjtNum, 2> keeper = {GoalkeeperPose()[0], GoalkeeperPose()[1]};
    state["info:goalkeeper_pos"_].Assign(keeper.data(), 2);
    if constexpr (kFromPixels) {
      AssignPixelObservation("obs:pixels", &state["obs:pixels"_], reset);
    }
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoChallengeChaseTagEnvBase : public Env<EnvSpecT>,
                                    public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum distance{0.0};
    mjtNum lose{0.0};
    mjtNum sparse{0.0};
    bool solved{false};
    bool done{false};
  };

  enum TaskType { kChase = 0, kEvade = 1 };
  enum OpponentPolicy {
    kStaticStationary = 0,
    kStationary = 1,
    kRandom = 2,
    kChasePlayer = 3,
    kRepeller = 4,
  };

  bool normalize_act_;
  std::string reset_type_;
  mjtNum win_distance_;
  mjtNum min_spawn_distance_;
  std::string task_choice_;
  std::string terrain_;
  bool repeller_opponent_;
  std::vector<mjtNum> hills_range_;
  std::vector<mjtNum> rough_range_;
  std::vector<mjtNum> relief_range_;
  mjtNum chase_vel_low_;
  mjtNum chase_vel_high_;
  mjtNum random_vel_low_;
  mjtNum random_vel_high_;
  mjtNum repeller_vel_low_;
  mjtNum repeller_vel_high_;
  mjtNum reward_distance_w_;
  mjtNum reward_lose_w_;
  std::vector<mjtNum> opponent_probabilities_;
  int pelvis_body_id_{-1};
  int head_site_id_{-1};
  int foot_l_body_id_{-1};
  int foot_r_body_id_{-1};
  int opponent_body_id_{-1};
  int opponent_mocap_id_{-1};
  int success_indicator_site_id_{-1};
  int terrain_geom_id_{-1};
  int terrain_hfield_id_{-1};
  std::array<int, 4> grf_sensor_ids_{-1, -1, -1, -1};
  std::vector<int> muscle_actuator_ids_;
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;
  challenge_extra_detail::ColoredNoiseProcess opponent_noise_process_{
      2.0, 2, 2000, 10.0};
  std::vector<mjtNum> initial_hfield_data_;
  std::vector<mjtNum> initial_terrain_rgba_;
  std::vector<mjtNum> initial_terrain_pos_;
  std::vector<mjtNum> relief_patch_template_;
  int current_task_{kChase};
  int opponent_policy_{kStationary};
  std::array<mjtNum, 2> opponent_velocity_{0.0, 0.0};
  mjtNum chase_velocity_{1.0};
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_ctrl_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_act_dot_;
  std::vector<mjtNum> test_reset_qacc_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_hfield_data_;
  std::vector<mjtNum> test_terrain_geom_rgba_;
  std::vector<mjtNum> test_terrain_geom_pos_;
  int test_task_{-1};
  std::vector<mjtNum> test_opponent_pose_;
  std::vector<mjtNum> test_opponent_velocity_;
  std::vector<mjtNum> test_opponent_noise_buffer_;
  int test_opponent_noise_idx_{-1};
  mjtNum test_chase_velocity_{-1.0};
  int test_opponent_policy_{-1};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoChallengeChaseTagEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        reset_type_(spec.config["reset_type"_]),
        win_distance_(spec.config["win_distance"_]),
        min_spawn_distance_(spec.config["min_spawn_distance"_]),
        task_choice_(spec.config["task_choice"_]),
        terrain_(spec.config["terrain"_]),
        repeller_opponent_(spec.config["repeller_opponent"_]),
        hills_range_(detail::ToMjtVector(spec.config["hills_range"_])),
        rough_range_(detail::ToMjtVector(spec.config["rough_range"_])),
        relief_range_(detail::ToMjtVector(spec.config["relief_range"_])),
        chase_vel_low_(spec.config["chase_vel_low"_]),
        chase_vel_high_(spec.config["chase_vel_high"_]),
        random_vel_low_(spec.config["random_vel_low"_]),
        random_vel_high_(spec.config["random_vel_high"_]),
        repeller_vel_low_(spec.config["repeller_vel_low"_]),
        repeller_vel_high_(spec.config["repeller_vel_high"_]),
        reward_distance_w_(spec.config["reward_distance_w"_]),
        reward_lose_w_(spec.config["reward_lose_w"_]),
        opponent_probabilities_(
            detail::ToMjtVector(spec.config["opponent_probabilities"_])),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_ctrl_(detail::ToMjtVector(spec.config["test_reset_ctrl"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_act_dot_(
            detail::ToMjtVector(spec.config["test_reset_act_dot"_])),
        test_reset_qacc_(detail::ToMjtVector(spec.config["test_reset_qacc"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_hfield_data_(
            detail::ToMjtVector(spec.config["test_hfield_data"_])),
        test_terrain_geom_rgba_(
            detail::ToMjtVector(spec.config["test_terrain_geom_rgba"_])),
        test_terrain_geom_pos_(
            detail::ToMjtVector(spec.config["test_terrain_geom_pos"_])),
        test_task_(spec.config["test_task"_]),
        test_opponent_pose_(
            detail::ToMjtVector(spec.config["test_opponent_pose"_])),
        test_opponent_velocity_(
            detail::ToMjtVector(spec.config["test_opponent_velocity"_])),
        test_opponent_noise_buffer_(
            detail::ToMjtVector(spec.config["test_opponent_noise_buffer"_])),
        test_opponent_noise_idx_(spec.config["test_opponent_noise_idx"_]),
        test_chase_velocity_(spec.config["test_chase_velocity"_]),
        test_opponent_policy_(spec.config["test_opponent_policy"_]) {
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    for (int actuator = 0; actuator < model_->nu; ++actuator) {
      if (muscle_actuator_[actuator]) {
        muscle_actuator_ids_.push_back(actuator);
      }
    }
    InitializeRobotEnv();
  }

  envpool::mujoco::CameraPolicy RenderCameraPolicy() const override {
    return detail::MyoSuiteRenderCameraPolicy();
  }

  void ConfigureRenderOption(mjvOption* option) const override {
    detail::ConfigureMyoSuiteRenderOptions(option, true);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    ApplyTerrainReset();
    SampleTask();
    ApplyResetState();
    FlattenAgentTerrainPatch();
    ResetOpponent();
    ApplyOpponentTestState();
    mj_forward(model_, data_);
    CaptureResetState();
    InvalidateRenderCache();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    UpdateOpponentState();
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
    detail::RefreshObservedMyoSuiteState(model_, data_);
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->na != spec_.config["act_dim"_] ||
        model_->nu != spec_.config["action_dim"_]) {
      throw std::runtime_error("ChaseTag dims do not match model.");
    }
  }

  void CacheIds() {
    pelvis_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "pelvis");
    head_site_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "head");
    foot_l_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "talus_l");
    foot_r_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "talus_r");
    opponent_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "opponent");
    opponent_mocap_id_ = model_->body_mocapid[opponent_body_id_];
    success_indicator_site_id_ = challenge_extra_detail::RequireId(
        model_, mjOBJ_SITE, "opponent_indicator");
    grf_sensor_ids_[0] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "r_foot");
    grf_sensor_ids_[1] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "r_toes");
    grf_sensor_ids_[2] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "l_foot");
    grf_sensor_ids_[3] =
        challenge_extra_detail::RequireId(model_, mjOBJ_SENSOR, "l_toes");
    terrain_geom_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_GEOM, "terrain");
    terrain_hfield_id_ = model_->geom_dataid[terrain_geom_id_];
    detail::CopyModelGeomRgba(model_, terrain_geom_id_, &initial_terrain_rgba_);
    detail::CopyModelGeomPos(model_, terrain_geom_id_, &initial_terrain_pos_);
    if (terrain_hfield_id_ >= 0) {
      detail::CopyModelHfieldData(model_, terrain_hfield_id_,
                                  &initial_hfield_data_);
      if (terrain_ != "FLAT") {
        relief_patch_template_ =
            challenge_extra_detail::LoadNormalizedReliefPatch(
                myosuite::MyoSuiteAssetRoot(spec_.config["base_path"_]) +
                "/envs/myo/assets/myo_relief.npy");
        int patch_rows = model_->hfield_nrow[terrain_hfield_id_] / 3;
        int patch_cols = model_->hfield_ncol[terrain_hfield_id_] / 3;
        if (static_cast<int>(relief_patch_template_.size()) !=
            patch_rows * patch_cols) {
          throw std::runtime_error(
              "MyoSuite relief patch size does not match ChaseTag hfield.");
        }
      }
    }
    int expected_obs = 28 + 28 + 4 + 4 + 3 + 2 + 2 + 2 + model_->na * 4;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("ChaseTag obs_dim does not match model.");
    }
  }

  static void RotateSquarePatch90(std::vector<mjtNum>* patch, int side) {
    std::vector<mjtNum> rotated(patch->size(), 0.0);
    for (int row = 0; row < side; ++row) {
      for (int col = 0; col < side; ++col) {
        rotated[(side - 1 - col) * side + row] = (*patch)[row * side + col];
      }
    }
    *patch = std::move(rotated);
  }

  mjtNum RangeLow(const std::vector<mjtNum>& range) const {
    return range.empty() ? static_cast<mjtNum>(0.0) : range[0];
  }

  mjtNum RangeHigh(const std::vector<mjtNum>& range) const {
    if (range.size() < 2) {
      return RangeLow(range);
    }
    return range[1];
  }

  void RestoreTerrainModelState() {
    detail::RestoreModelGeomRgba(model_, terrain_geom_id_,
                                 initial_terrain_rgba_);
    detail::RestoreModelGeomPos(model_, terrain_geom_id_, initial_terrain_pos_);
    if (terrain_hfield_id_ >= 0) {
      detail::RestoreModelHfieldData(model_, terrain_hfield_id_,
                                     initial_hfield_data_);
    }
  }

  void SetTerrainVisible() {
    std::vector<mjtNum> rgba = initial_terrain_rgba_;
    std::vector<mjtNum> pos = initial_terrain_pos_;
    if (rgba.size() >= 4) {
      rgba[3] = 1.0;
    }
    if (pos.size() >= 3) {
      pos[0] = 0.0;
      pos[1] = 0.0;
      pos[2] = 0.0;
    }
    detail::RestoreModelGeomRgba(model_, terrain_geom_id_, rgba);
    detail::RestoreModelGeomPos(model_, terrain_geom_id_, pos);
  }

  void HideTerrain() {
    std::vector<mjtNum> rgba = initial_terrain_rgba_;
    std::vector<mjtNum> pos = initial_terrain_pos_;
    if (rgba.size() >= 4) {
      rgba[3] = 0.0;
    }
    if (pos.size() >= 3) {
      pos[0] = 0.0;
      pos[1] = 0.0;
      pos[2] = -10.0;
    }
    detail::RestoreModelGeomRgba(model_, terrain_geom_id_, rgba);
    detail::RestoreModelGeomPos(model_, terrain_geom_id_, pos);
  }

  void FillTerrainPatch(int patch_i, int patch_j,
                        const std::vector<mjtNum>& patch) {
    if (terrain_hfield_id_ < 0) {
      return;
    }
    int rows = model_->hfield_nrow[terrain_hfield_id_];
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    int patch_rows = rows / 3;
    int patch_cols = cols / 3;
    if (patch_rows <= 0 || patch_cols <= 0 ||
        static_cast<int>(patch.size()) != patch_rows * patch_cols) {
      throw std::runtime_error("Invalid ChaseTag terrain patch.");
    }
    int adr = model_->hfield_adr[terrain_hfield_id_];
    for (int row = 0; row < patch_rows; ++row) {
      for (int col = 0; col < patch_cols; ++col) {
        int dst_row = patch_i * patch_rows + row;
        int dst_col = patch_j * patch_cols + col;
        if (dst_row >= rows || dst_col >= cols) {
          continue;
        }
        model_->hfield_data[adr + dst_row * cols + dst_col] =
            static_cast<float>(patch[row * patch_cols + col]);
      }
    }
  }

  std::vector<mjtNum> FlatTerrainPatch(int patch_rows, int patch_cols) const {
    return std::vector<mjtNum>(patch_rows * patch_cols,
                               static_cast<mjtNum>(0.0));
  }

  std::vector<mjtNum> RoughTerrainPatch(int patch_rows, int patch_cols) {
    std::vector<mjtNum> patch(patch_rows * patch_cols, 0.0);
    for (mjtNum& value : patch) {
      value = challenge_detail::UniformScalar(&gen_, -1.0, 1.0);
    }
    auto [low_it, high_it] = std::minmax_element(patch.begin(), patch.end());
    mjtNum denom = std::max(*high_it - *low_it, static_cast<mjtNum>(1e-12));
    mjtNum scalar = challenge_detail::UniformScalar(
        &gen_, RangeLow(rough_range_), RangeHigh(rough_range_));
    for (mjtNum& value : patch) {
      value = (value - *low_it) / denom * scalar - static_cast<mjtNum>(0.02);
    }
    return patch;
  }

  std::vector<mjtNum> HillyTerrainPatch(int patch_rows, int patch_cols) {
    int total = patch_rows * patch_cols;
    std::vector<mjtNum> data(total, 0.0);
    for (int i = 0; i < total; ++i) {
      mjtNum angle =
          static_cast<mjtNum>(10.0 * detail::kPi * i / std::max(total - 1, 1));
      data[i] = std::sin(angle + static_cast<mjtNum>(0.5) * detail::kPi) -
                static_cast<mjtNum>(1.0);
    }
    auto [low_it, high_it] = std::minmax_element(data.begin(), data.end());
    mjtNum denom = std::max(*high_it - *low_it, static_cast<mjtNum>(1e-12));
    mjtNum scalar = challenge_detail::UniformScalar(
        &gen_, RangeLow(hills_range_), RangeHigh(hills_range_));
    std::vector<mjtNum> patch(total, 0.0);
    for (int row = 0; row < patch_rows; ++row) {
      for (int col = 0; col < patch_cols; ++col) {
        int src = row * patch_cols + col;
        int dst_row = patch_rows - 1 - row;
        int dst_col = patch_cols - 1 - col;
        patch[dst_row * patch_cols + dst_col] =
            (data[src] - *low_it) / denom * scalar;
      }
    }
    if (patch_rows == patch_cols &&
        challenge_detail::UniformScalar(&gen_, 0.0, 1.0) < 0.5) {
      RotateSquarePatch90(&patch, patch_rows);
    }
    return patch;
  }

  std::vector<mjtNum> ReliefTerrainPatch(int patch_rows, int patch_cols) {
    if (static_cast<int>(relief_patch_template_.size()) !=
        patch_rows * patch_cols) {
      throw std::runtime_error("Unexpected MyoSuite relief patch dimensions.");
    }
    mjtNum scalar = challenge_detail::UniformScalar(
        &gen_, RangeLow(relief_range_), RangeHigh(relief_range_));
    std::vector<mjtNum> patch = relief_patch_template_;
    for (mjtNum& value : patch) {
      value *= scalar;
    }
    return patch;
  }

  void ApplyTerrainReset() {
    RestoreTerrainModelState();
    if (!test_terrain_geom_rgba_.empty()) {
      detail::RestoreModelGeomRgba(model_, terrain_geom_id_,
                                   test_terrain_geom_rgba_);
    }
    if (!test_terrain_geom_pos_.empty()) {
      detail::RestoreModelGeomPos(model_, terrain_geom_id_,
                                  test_terrain_geom_pos_);
    }
    if (terrain_hfield_id_ < 0) {
      return;
    }
    if (!test_hfield_data_.empty()) {
      detail::RestoreModelHfieldData(model_, terrain_hfield_id_,
                                     test_hfield_data_);
      return;
    }
    if (terrain_ == "FLAT") {
      HideTerrain();
      return;
    }
    if (terrain_ != "random") {
      throw std::runtime_error("Unsupported ChaseTag terrain type.");
    }
    SetTerrainVisible();
    int rows = model_->hfield_nrow[terrain_hfield_id_];
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    int patch_rows = rows / 3;
    int patch_cols = cols / 3;
    detail::RestoreModelHfieldData(
        model_, terrain_hfield_id_,
        std::vector<mjtNum>(rows * cols, static_cast<mjtNum>(0.0)));
    std::array<int, 3> generated = {0, 0, 0};
    for (int patch_i = 0; patch_i < 3; ++patch_i) {
      for (int patch_j = 0; patch_j < 3; ++patch_j) {
        int terrain_choice = std::uniform_int_distribution<int>(0, 2)(gen_);
        while (terrain_choice == 1 && generated[terrain_choice] >= 2) {
          terrain_choice = std::uniform_int_distribution<int>(0, 2)(gen_);
        }
        ++generated[terrain_choice];
        if (terrain_choice == 0) {
          FillTerrainPatch(patch_i, patch_j,
                           FlatTerrainPatch(patch_rows, patch_cols));
        } else if (terrain_choice == 1) {
          FillTerrainPatch(patch_i, patch_j,
                           HillyTerrainPatch(patch_rows, patch_cols));
        } else {
          FillTerrainPatch(patch_i, patch_j,
                           RoughTerrainPatch(patch_rows, patch_cols));
        }
      }
    }
    if (challenge_detail::UniformScalar(&gen_, 0.0, 1.0) < 0.2) {
      int patch_i = std::uniform_int_distribution<int>(0, 2)(gen_);
      int patch_j = std::uniform_int_distribution<int>(0, 2)(gen_);
      FillTerrainPatch(patch_i, patch_j,
                       ReliefTerrainPatch(patch_rows, patch_cols));
    }
  }

  void FlattenAgentTerrainPatch() {
    if (terrain_hfield_id_ < 0 || !test_hfield_data_.empty() ||
        terrain_ == "FLAT") {
      return;
    }
    int rows = model_->hfield_nrow[terrain_hfield_id_];
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    int patch_rows = rows / 3;
    int patch_cols = cols / 3;
    if (patch_rows <= 0 || patch_cols <= 0) {
      return;
    }
    constexpr mjtNum kRealLength = 12.0;
    constexpr mjtNum kRealWidth = 12.0;
    mjtNum map_i =
        data_->qpos[0] / (kRealLength / rows) + static_cast<mjtNum>(rows) * 0.5;
    mjtNum map_j =
        data_->qpos[1] / (kRealWidth / cols) + static_cast<mjtNum>(cols) * 0.5;
    int clipped_i = static_cast<int>(
        std::clamp(map_i, static_cast<mjtNum>(0.0),
                   static_cast<mjtNum>(std::max(rows - 2, 0))));
    int clipped_j = static_cast<int>(
        std::clamp(map_j, static_cast<mjtNum>(0.0),
                   static_cast<mjtNum>(std::max(cols - 2, 0))));
    FillTerrainPatch(clipped_i / patch_rows, clipped_j / patch_cols,
                     FlatTerrainPatch(patch_rows, patch_cols));
  }

  void SampleTask() {
    if (test_task_ >= 0) {
      current_task_ = test_task_;
    } else if (task_choice_ == "random") {
      current_task_ = std::uniform_int_distribution<int>(0, 1)(gen_);
    } else {
      current_task_ = task_choice_ == "EVADE" ? kEvade : kChase;
    }
  }

  std::array<mjtNum, 3> OpponentPose() const {
    return {
        data_->mocap_pos[opponent_mocap_id_ * 3 + 0],
        data_->mocap_pos[opponent_mocap_id_ * 3 + 1],
        challenge_extra_detail::QuatYaw(data_->mocap_quat +
                                        opponent_mocap_id_ * 4),
    };
  }

  void SetOpponentPose(mjtNum x, mjtNum y, mjtNum yaw) {
    data_->mocap_pos[opponent_mocap_id_ * 3 + 0] =
        challenge_extra_detail::Clip(x, -5.5, 5.5);
    data_->mocap_pos[opponent_mocap_id_ * 3 + 1] =
        challenge_extra_detail::Clip(y, -5.5, 5.5);
    data_->mocap_pos[opponent_mocap_id_ * 3 + 2] = 0.0;
    auto quat = challenge_extra_detail::YawToQuat(yaw);
    std::memcpy(data_->mocap_quat + opponent_mocap_id_ * 4, quat.data(),
                sizeof(mjtNum) * 4);
  }

  void MoveOpponent(const std::array<mjtNum, 2>& velocity) {
    auto pose = OpponentPose();
    std::array<mjtNum, 2> reported = velocity;
    reported[0] = std::abs(reported[0]);
    std::array<mjtNum, 2> clipped = reported;
    clipped[0] = challenge_extra_detail::Clip(clipped[0], -2.0, 2.0);
    clipped[1] = challenge_extra_detail::Clip(clipped[1], -2.0, 2.0);
    mjtNum x_vel =
        clipped[0] * std::cos(pose[2] + static_cast<mjtNum>(0.5) * detail::kPi);
    mjtNum y_vel =
        clipped[0] * std::sin(pose[2] + static_cast<mjtNum>(0.5) * detail::kPi);
    pose[0] -= Dt() * x_vel;
    pose[1] -= Dt() * y_vel;
    pose[2] += Dt() * clipped[1];
    SetOpponentPose(pose[0], pose[1], pose[2]);
    opponent_velocity_ = reported;
  }

  std::array<mjtNum, 2> RandomMovement() {
    auto sample = opponent_noise_process_.Sample(&gen_);
    return {
        challenge_extra_detail::Clip(sample[0], random_vel_low_,
                                     random_vel_high_),
        challenge_extra_detail::Clip(sample[1], random_vel_low_,
                                     random_vel_high_),
    };
  }

  std::array<mjtNum, 2> ChasePlayer() const {
    auto pose = OpponentPose();
    std::array<mjtNum, 2> vec = {pose[0], pose[1]};
    std::array<mjtNum, 2> pelvis = {data_->xpos[pelvis_body_id_ * 3 + 0],
                                    data_->xpos[pelvis_body_id_ * 3 + 1]};
    std::array<mjtNum, 2> facing = {std::cos(pose[2]), std::sin(pose[2])};
    std::array<mjtNum, 2> to_pelvis = {pelvis[0] - vec[0], pelvis[1] - vec[1]};
    mjtNum angular = facing[0] * to_pelvis[0] + facing[1] * to_pelvis[1];
    return {chase_velocity_, angular};
  }

  std::vector<std::array<mjtNum, 2>> WallPositions() const {
    std::vector<std::array<mjtNum, 2>> out;
    out.reserve(100);
    for (int i = 0; i < 25; ++i) {
      mjtNum value = -8.7 + static_cast<mjtNum>(17.4) * static_cast<mjtNum>(i) /
                                static_cast<mjtNum>(24);
      out.push_back({8.7, value});
      out.push_back({-8.7, value});
      out.push_back({value, 8.7});
      out.push_back({value, -8.7});
    }
    return out;
  }

  mjtNum CalcRepellerAngularVel(mjtNum current_yaw, mjtNum desired_yaw) const {
    auto normalize = [](mjtNum angle) {
      constexpr mjtNum two_pi = static_cast<mjtNum>(2.0) * detail::kPi;
      while (angle > two_pi) {
        angle -= two_pi;
      }
      while (angle < 0.0) {
        angle += two_pi;
      }
      return angle;
    };
    current_yaw = normalize(current_yaw);
    desired_yaw = normalize(desired_yaw);
    mjtNum clockwise = std::abs(current_yaw) +
                       (static_cast<mjtNum>(2.0) * detail::kPi - desired_yaw);
    mjtNum anticlockwise =
        (static_cast<mjtNum>(2.0) * detail::kPi - current_yaw) + desired_yaw;
    return clockwise < anticlockwise ? 1.0 : -1.0;
  }

  std::array<mjtNum, 2> RepellerPolicy() {
    constexpr mjtNum kDistInfluence = 3.5;
    constexpr mjtNum kEta = 20.0;
    auto pose = OpponentPose();
    std::vector<std::array<mjtNum, 2>> repellers = WallPositions();
    repellers.insert(repellers.begin(), {data_->xpos[pelvis_body_id_ * 3 + 0],
                                         data_->xpos[pelvis_body_id_ * 3 + 1]});
    std::vector<int> active_indices;
    active_indices.reserve(repellers.size());
    for (std::size_t i = 0; i < repellers.size(); ++i) {
      mjtNum dist = challenge_extra_detail::Norm2(repellers[i][0] - pose[0],
                                                  repellers[i][1] - pose[1]);
      if (dist < kDistInfluence) {
        active_indices.push_back(static_cast<int>(i));
      }
    }
    if (active_indices.empty()) {
      auto noise = opponent_noise_process_.Sample(&gen_);
      mjtNum linear = challenge_extra_detail::Clip(noise[0], repeller_vel_low_,
                                                   repeller_vel_high_);
      mjtNum angular = CalcRepellerAngularVel(pose[2], noise[1]);
      return {linear, angular};
    }
    std::array<mjtNum, 2> repel_center{0.0, 0.0};
    mjtNum repel_force = 0.0;
    for (int index : active_indices) {
      repel_center[0] += repellers[index][0];
      repel_center[1] += repellers[index][1];
      mjtNum dist =
          std::max(challenge_extra_detail::Norm2(repellers[index][0] - pose[0],
                                                 repellers[index][1] - pose[1]),
                   static_cast<mjtNum>(1e-5));
      mjtNum term = static_cast<mjtNum>(1.0) / dist -
                    static_cast<mjtNum>(1.0) / kDistInfluence;
      repel_force += static_cast<mjtNum>(0.5) * kEta * term * term;
    }
    repel_center[0] /= static_cast<mjtNum>(active_indices.size());
    repel_center[1] /= static_cast<mjtNum>(active_indices.size());
    mjtNum linear = challenge_extra_detail::Clip(
        repel_force / static_cast<mjtNum>(active_indices.size()),
        repeller_vel_low_, repeller_vel_high_);
    std::array<mjtNum, 2> escape = {pose[0] - repel_center[0],
                                    pose[1] - repel_center[1]};
    for (mjtNum& component : escape) {
      if (std::abs(component) <= 0.1) {
        mjtNum sign = component > 0.0 ? 1.0 : (component < 0.0 ? -1.0 : 0.0);
        component = -sign * challenge_detail::UniformScalar(&gen_, 0.3, 0.9);
      }
    }
    mjtNum direction =
        std::atan2(escape[1], escape[0]) + static_cast<mjtNum>(1.57);
    mjtNum angular = CalcRepellerAngularVel(pose[2], direction);
    return {linear, angular};
  }

  void SampleOpponentPolicy() {
    if (current_task_ == kEvade) {
      opponent_policy_ = kChasePlayer;
      return;
    }
    mjtNum u = challenge_detail::UniformScalar(&gen_, 0.0, 1.0);
    if (u < opponent_probabilities_[0]) {
      opponent_policy_ = kStaticStationary;
    } else if (u < opponent_probabilities_[0] + opponent_probabilities_[1]) {
      opponent_policy_ = kStationary;
    } else if (u < opponent_probabilities_[0] + opponent_probabilities_[1] +
                       opponent_probabilities_[2]) {
      opponent_policy_ = kRandom;
    } else {
      opponent_policy_ = kRepeller;
    }
  }

  void ResetOpponent() {
    opponent_noise_process_.Reset(&gen_);
    opponent_velocity_ = {0.0, 0.0};
    SampleOpponentPolicy();
    chase_velocity_ =
        challenge_detail::UniformScalar(&gen_, chase_vel_low_, chase_vel_high_);
    std::array<mjtNum, 2> pelvis = {data_->xpos[pelvis_body_id_ * 3 + 0],
                                    data_->xpos[pelvis_body_id_ * 3 + 1]};
    if (opponent_policy_ == kStaticStationary) {
      SetOpponentPose(0.0, -5.0, 0.0);
      opponent_velocity_ = {0.0, 0.0};
      return;
    }
    while (true) {
      mjtNum x = challenge_detail::UniformScalar(&gen_, -5.0, 5.0);
      mjtNum y = challenge_detail::UniformScalar(&gen_, -5.0, 5.0);
      if (challenge_extra_detail::Norm2(x - pelvis[0], y - pelvis[1]) <
          min_spawn_distance_) {
        continue;
      }
      mjtNum yaw = challenge_detail::UniformScalar(
          &gen_, -static_cast<mjtNum>(2.0) * detail::kPi,
          static_cast<mjtNum>(2.0) * detail::kPi);
      SetOpponentPose(x, y, yaw);
      opponent_velocity_ = {0.0, 0.0};
      return;
    }
  }

  void UpdateOpponentState() {
    std::array<mjtNum, 2> velocity = {0.0, 0.0};
    if (opponent_policy_ == kRandom) {
      velocity = RandomMovement();
    } else if (opponent_policy_ == kChasePlayer) {
      velocity = ChasePlayer();
    } else if (opponent_policy_ == kRepeller) {
      velocity = RepellerPolicy();
    }
    MoveOpponent(velocity);
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    } else if (reset_type_ == "random") {
      int key = std::uniform_int_distribution<int>(0, 1)(gen_) == 0 ? 2 : 3;
      detail::RestoreVector(
          std::vector<mjtNum>(model_->key_qpos + key * model_->nq,
                              model_->key_qpos + (key + 1) * model_->nq),
          data_->qpos);
      detail::RestoreVector(
          std::vector<mjtNum>(model_->key_qvel + key * model_->nv,
                              model_->key_qvel + (key + 1) * model_->nv),
          data_->qvel);
      std::normal_distribution<double> noise(0.0, 0.02);
      std::array<mjtNum, 4> quat = {data_->qpos[3], data_->qpos[4],
                                    data_->qpos[5], data_->qpos[6]};
      mjtNum height = data_->qpos[2];
      for (int i = 0; i < model_->nq; ++i) {
        data_->qpos[i] += static_cast<mjtNum>(noise(gen_));
      }
      data_->qpos[2] = height;
      std::copy(quat.begin(), quat.end(), data_->qpos + 3);
      data_->qpos[0] = challenge_detail::UniformScalar(&gen_, -5.0, 5.0);
      data_->qpos[1] = challenge_detail::UniformScalar(&gen_, -5.0, 5.0);
      mjtNum orientation = challenge_detail::UniformScalar(
          &gen_, 0.0, static_cast<mjtNum>(2.0) * detail::kPi);
      std::array<mjtNum, 9> mat{};
      mju_quat2Mat(mat.data(), data_->qpos + 3);
      auto euler = challenge_detail::Mat9ToEuler(mat.data());
      auto quat_with_orientation =
          challenge_detail::EulerXYZToQuat({euler[0], euler[1], orientation});
      std::copy(quat_with_orientation.begin(), quat_with_orientation.end(),
                data_->qpos + 3);
      mjtNum planar_speed =
          challenge_extra_detail::Norm2(data_->qvel[0], data_->qvel[1]);
      data_->qvel[0] = std::cos(orientation) * planar_speed;
      data_->qvel[1] = std::sin(orientation) * planar_speed;
    } else if (reset_type_ == "init") {
      detail::RestoreVector(
          std::vector<mjtNum>(model_->key_qpos + 2 * model_->nq,
                              model_->key_qpos + 3 * model_->nq),
          data_->qpos);
      detail::RestoreVector(
          std::vector<mjtNum>(model_->key_qvel + 2 * model_->nv,
                              model_->key_qvel + 3 * model_->nv),
          data_->qvel);
    } else {
      detail::RestoreVector(
          std::vector<mjtNum>(model_->key_qpos, model_->key_qpos + model_->nq),
          data_->qpos);
      detail::RestoreVector(
          std::vector<mjtNum>(model_->key_qvel, model_->key_qvel + model_->nv),
          data_->qvel);
    }
    mj_forward(model_, data_);
    bool rerun_forward = false;
    if (!test_reset_ctrl_.empty()) {
      detail::RestoreVector(test_reset_ctrl_, data_->ctrl);
      rerun_forward = true;
    }
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
      rerun_forward = true;
    }
    if (!test_reset_act_dot_.empty()) {
      detail::RestoreVector(test_reset_act_dot_, data_->act_dot);
    }
    if (!test_reset_qacc_.empty()) {
      detail::RestoreVector(test_reset_qacc_, data_->qacc);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
      rerun_forward = true;
    }
    if (rerun_forward) {
      mj_forward(model_, data_);
    }
  }

  void ApplyOpponentTestState() {
    if (test_opponent_pose_.size() == 3) {
      SetOpponentPose(test_opponent_pose_[0], test_opponent_pose_[1],
                      test_opponent_pose_[2]);
    }
    if (test_opponent_velocity_.size() == 2) {
      opponent_velocity_[0] = test_opponent_velocity_[0];
      opponent_velocity_[1] = test_opponent_velocity_[1];
    }
    if (!test_opponent_noise_buffer_.empty()) {
      opponent_noise_process_.SetState(test_opponent_noise_buffer_,
                                       test_opponent_noise_idx_);
    }
    if (test_chase_velocity_ >= 0.0) {
      chase_velocity_ = test_chase_velocity_;
    }
    if (test_opponent_policy_ >= 0) {
      opponent_policy_ = test_opponent_policy_;
    }
  }

  bool Fallen() const {
    if (terrain_ == "FLAT") {
      return data_->xpos[pelvis_body_id_ * 3 + 2] < 0.5;
    }
    mjtNum head_z = data_->site_xpos[head_site_id_ * 3 + 2];
    mjtNum left_z = data_->xpos[foot_l_body_id_ * 3 + 2];
    mjtNum right_z = data_->xpos[foot_r_body_id_ * 3 + 2];
    return head_z - (left_z + right_z) * 0.5 < 0.2;
  }

  bool ChaseWin() const {
    auto pose = OpponentPose();
    return challenge_extra_detail::Norm2(data_->qpos[0] - pose[0],
                                         data_->qpos[1] - pose[1]) <=
           win_distance_;
  }

  bool ChaseLose() const {
    return data_->time >= 20.0 || std::abs(data_->qpos[0]) > 6.5 ||
           std::abs(data_->qpos[1]) > 6.5 || Fallen();
  }

  bool EvadeLose() const {
    auto pose = OpponentPose();
    return challenge_extra_detail::Norm2(data_->qpos[0] - pose[0],
                                         data_->qpos[1] - pose[1]) <=
               win_distance_ ||
           std::abs(data_->qpos[0]) > 6.5 || std::abs(data_->qpos[1]) > 6.5;
  }

  RewardInfo ComputeRewardInfo() {
    RewardInfo reward;
    auto pose = OpponentPose();
    reward.distance = challenge_extra_detail::Norm2(data_->qpos[0] - pose[0],
                                                    data_->qpos[1] - pose[1]);
    bool win = false;
    bool lose = false;
    if (current_task_ == kChase) {
      win = ChaseWin();
      lose = ChaseLose();
      reward.sparse = win ? static_cast<mjtNum>(1.0 - data_->time / 20.0) : 0.0;
    } else {
      win = data_->time >= 20.0;
      lose = EvadeLose();
      reward.sparse =
          (win || lose) ? static_cast<mjtNum>(data_->time / 20.0) : 0.0;
    }
    reward.lose = lose ? 1.0 : 0.0;
    reward.solved = win;
    reward.done = win || lose;
    reward.dense_reward =
        reward_distance_w_ * reward.distance + reward_lose_w_ * reward.lose;
    if (success_indicator_site_id_ >= 0) {
      model_->site_rgba[success_indicator_site_id_ * 4 + 0] = win ? 0.0 : 2.0;
      model_->site_rgba[success_indicator_site_id_ * 4 + 1] = win ? 2.0 : 0.0;
      model_->site_rgba[success_indicator_site_id_ * 4 + 2] = 0.0;
      model_->site_rgba[success_indicator_site_id_ * 4 + 3] = win ? 0.2 : 0.0;
    }
    return reward;
  }

  std::vector<mjtNum> Observation() const {
    std::vector<mjtNum> obs;
    obs.reserve(spec_.config["obs_dim"_]);
    obs.insert(obs.end(), data_->qpos + 7, data_->qpos + 35);
    for (int i = 6; i < 34; ++i) {
      obs.push_back(data_->qvel[i] * Dt());
    }
    for (int sensor_id : grf_sensor_ids_) {
      obs.push_back(data_->sensordata[model_->sensor_adr[sensor_id]]);
    }
    obs.insert(obs.end(), data_->xquat + pelvis_body_id_ * 4,
               data_->xquat + pelvis_body_id_ * 4 + 4);
    auto pose = OpponentPose();
    obs.insert(obs.end(), pose.begin(), pose.end());
    obs.insert(obs.end(), opponent_velocity_.begin(), opponent_velocity_.end());
    obs.push_back(data_->qpos[0]);
    obs.push_back(data_->qpos[1]);
    obs.push_back(data_->qvel[0]);
    obs.push_back(data_->qvel[1]);
    for (int actuator : muscle_actuator_ids_) {
      obs.push_back(data_->actuator_length[actuator]);
    }
    for (int actuator : muscle_actuator_ids_) {
      obs.push_back(challenge_extra_detail::Clip(
          data_->actuator_velocity[actuator], -100.0, 100.0));
    }
    for (int actuator : muscle_actuator_ids_) {
      obs.push_back(challenge_extra_detail::Clip(
          data_->actuator_force[actuator] / 1000.0, -100.0, 100.0));
    }
    obs.insert(obs.end(), data_->act, data_->act + model_->na);
    return obs;
  }

  void WriteState(const RewardInfo& reward, bool reset, float reward_value) {
    auto obs = Observation();
    auto state = Allocate();
    if constexpr (!kFromPixels) {
      AssignObservation("obs", &state["obs"_], obs.data(), obs.size(), reset);
    }
    state["reward"_] = reward_value;
    state["discount"_] = 1.0f;
    state["done"_] = done_;
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    state["info:distance"_] = reward.distance;
    state["info:lose"_] = reward.lose;
    state["info:sparse"_] = reward.sparse;
    state["info:solved"_] = static_cast<mjtNum>(reward.solved);
    state["info:task"_] = static_cast<mjtNum>(current_task_);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    auto pose = OpponentPose();
    state["info:opponent_pose"_].Assign(pose.data(), 3);
    if constexpr (kFromPixels) {
      AssignPixelObservation("obs:pixels", &state["obs:pixels"_], reset);
    }
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoChallengeTableTennisEnvBase
    : public Env<EnvSpecT>,
      public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum reach_dist{0.0};
    mjtNum palm_dist{0.0};
    mjtNum paddle_quat{0.0};
    mjtNum torso_up{0.0};
    mjtNum act_reg{0.0};
    mjtNum sparse{0.0};
    bool solved{false};
    bool done{false};
    std::array<mjtNum, 6> touching_info{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  };

  enum PingpongLabel {
    kPaddle = 0,
    kOwn = 1,
    kOpponent = 2,
    kGround = 3,
    kNet = 4,
    kEnv = 5,
  };
  enum TrajIssue {
    kOwnHalf = 0,
    kMiss = 1,
    kNoPaddle = 2,
    kDoubleTouch = 3,
    kSuccess = 4,
  };

  static constexpr mjtNum kMaxTime = 3.0;

  bool normalize_act_;
  bool ball_qvel_;
  bool has_ball_xyz_range_{false};
  bool has_ball_friction_range_{false};
  bool has_paddle_mass_range_{false};
  bool has_qpos_noise_range_{false};
  std::array<mjtNum, 3> ball_xyz_low_{0.0, 0.0, 0.0};
  std::array<mjtNum, 3> ball_xyz_high_{0.0, 0.0, 0.0};
  std::array<mjtNum, 3> ball_friction_low_{0.0, 0.0, 0.0};
  std::array<mjtNum, 3> ball_friction_high_{0.0, 0.0, 0.0};
  mjtNum paddle_mass_low_{0.0};
  mjtNum paddle_mass_high_{0.0};
  mjtNum qpos_noise_low_{0.0};
  mjtNum qpos_noise_high_{0.0};
  int rally_count_{1};
  mjtNum reward_reach_dist_w_;
  mjtNum reward_palm_dist_w_;
  mjtNum reward_paddle_quat_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_torso_up_w_;
  mjtNum reward_sparse_w_;
  mjtNum reward_solved_w_;
  mjtNum reward_done_w_;
  int pelvis_site_id_{-1};
  int palm_site_id_{-1};
  int paddle_body_id_{-1};
  int paddle_site_id_{-1};
  int paddle_geom_id_{-1};
  int ball_body_id_{-1};
  int ball_site_id_{-1};
  int ball_geom_id_{-1};
  int own_half_geom_id_{-1};
  int opponent_half_geom_id_{-1};
  int ground_geom_id_{-1};
  int net_geom_id_{-1};
  int ball_vel_sensor_id_{-1};
  int paddle_vel_sensor_id_{-1};
  int flex_extension_joint_id_{-1};
  int ball_joint_id_{-1};
  int ball_qposadr_{-1};
  int ball_dofadr_{-1};
  std::vector<int> body_qposadrs_;
  std::vector<int> body_dofadrs_;
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;
  std::vector<mjtNum> default_init_qpos_;
  std::vector<mjtNum> default_init_qvel_;
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> init_qvel_;
  std::array<mjtNum, 4> init_paddle_quat_{};
  std::vector<mjtNum> default_ball_body_pos_;
  std::vector<mjtNum> default_ball_geom_friction_;
  mjtNum default_paddle_body_mass_{0.0};
  int cur_rally_{0};
  std::vector<std::array<bool, 6>> contact_trajectory_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_act_dot_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_ball_body_pos_;
  std::vector<mjtNum> test_ball_geom_friction_;
  std::vector<mjtNum> test_paddle_body_mass_;
  std::vector<mjtNum> test_init_qpos_;
  std::vector<mjtNum> test_init_qvel_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoChallengeTableTennisEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        ball_qvel_(spec.config["ball_qvel"_]),
        rally_count_(spec.config["rally_count"_]),
        reward_reach_dist_w_(spec.config["reward_reach_dist_w"_]),
        reward_palm_dist_w_(spec.config["reward_palm_dist_w"_]),
        reward_paddle_quat_w_(spec.config["reward_paddle_quat_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        reward_torso_up_w_(spec.config["reward_torso_up_w"_]),
        reward_sparse_w_(spec.config["reward_sparse_w"_]),
        reward_solved_w_(spec.config["reward_solved_w"_]),
        reward_done_w_(spec.config["reward_done_w"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_act_dot_(
            detail::ToMjtVector(spec.config["test_reset_act_dot"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_ball_body_pos_(
            detail::ToMjtVector(spec.config["test_ball_body_pos"_])),
        test_ball_geom_friction_(
            detail::ToMjtVector(spec.config["test_ball_geom_friction"_])),
        test_paddle_body_mass_(
            detail::ToMjtVector(spec.config["test_paddle_body_mass"_])),
        test_init_qpos_(detail::ToMjtVector(spec.config["test_init_qpos"_])),
        test_init_qvel_(detail::ToMjtVector(spec.config["test_init_qvel"_])) {
    ParseRanges(spec);
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    default_init_qpos_.assign(model_->key_qpos, model_->key_qpos + model_->nq);
    default_init_qvel_.assign(model_->key_qvel, model_->key_qvel + model_->nv);
    init_qpos_ = default_init_qpos_;
    init_qvel_ = default_init_qvel_;
    init_qvel_[ball_dofadr_ + 0] = 5.6;
    init_qvel_[ball_dofadr_ + 1] = 1.6;
    init_qvel_[ball_dofadr_ + 2] = 0.1;
    default_init_qvel_ = init_qvel_;
    // Upstream TableTennis seeds this reference via
    // scipy.spatial.transform.Rotation.from_euler('xyz', ...).
    // as_quat()[[3,0,1,2]].
    init_paddle_quat_ =
        challenge_extra_detail::ScipyEulerXYZToQuat({-0.3, 1.57, 0.0});
    detail::CopyModelBodyPos(model_, ball_body_id_, &default_ball_body_pos_);
    detail::CopyModelGeomFriction(model_, ball_geom_id_,
                                  &default_ball_geom_friction_);
    detail::CopyModelBodyMass(model_, paddle_body_id_,
                              &default_paddle_body_mass_);
    InitializeRobotEnv();
  }

  envpool::mujoco::CameraPolicy RenderCameraPolicy() const override {
    return detail::MyoSuiteRenderCameraPolicy();
  }

  void ConfigureRenderOption(mjvOption* option) const override {
    detail::ConfigureMyoSuiteRenderOptions(option);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    cur_rally_ = 0;
    contact_trajectory_.clear();
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    init_qpos_ = default_init_qpos_;
    init_qvel_ = default_init_qvel_;
    RestoreModelState();
    ApplyModelRandomization();
    ApplyOracleModelOverrides();
    ApplyResetState();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    std::vector<float> processed_controls(model_->nu);
    for (int i = 0; i < model_->nu; ++i) {
      processed_controls[i] = raw[i];
      if (normalize_act_ && model_->actuator_dyntype[i] != mjDYN_MUSCLE) {
        mjtNum low = model_->actuator_ctrlrange[i * 2];
        mjtNum high = model_->actuator_ctrlrange[i * 2 + 1];
        processed_controls[i] =
            static_cast<float>((low + high) * static_cast<mjtNum>(0.5) +
                               static_cast<mjtNum>(raw[i]) * (high - low) *
                                   static_cast<mjtNum>(0.5));
      }
    }
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                processed_controls.data());
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    if (reward.solved && cur_rally_ + 1 < rally_count_) {
      ++cur_rally_;
      data_->time = 0.0;
      contact_trajectory_.clear();
      RelaunchBall();
      reward.solved = false;
      reward.done = false;
    }
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ParseRanges(const Spec& spec) {
    auto ball_xyz_low = spec.config["ball_xyz_low"_];
    auto ball_xyz_high = spec.config["ball_xyz_high"_];
    if (!ball_xyz_low.empty() && !ball_xyz_high.empty()) {
      has_ball_xyz_range_ = true;
      for (int i = 0; i < 3; ++i) {
        ball_xyz_low_[i] = static_cast<mjtNum>(ball_xyz_low[i]);
        ball_xyz_high_[i] = static_cast<mjtNum>(ball_xyz_high[i]);
      }
    }
    auto ball_friction_low = spec.config["ball_friction_low"_];
    auto ball_friction_high = spec.config["ball_friction_high"_];
    if (!ball_friction_low.empty() && !ball_friction_high.empty()) {
      has_ball_friction_range_ = true;
      for (int i = 0; i < 3; ++i) {
        ball_friction_low_[i] = static_cast<mjtNum>(ball_friction_low[i]);
        ball_friction_high_[i] = static_cast<mjtNum>(ball_friction_high[i]);
      }
    }
    has_paddle_mass_range_ =
        spec.config["paddle_mass_high"_] > spec.config["paddle_mass_low"_];
    paddle_mass_low_ = spec.config["paddle_mass_low"_];
    paddle_mass_high_ = spec.config["paddle_mass_high"_];
    if (!std::isnan(spec.config["qpos_noise_low"_]) &&
        !std::isnan(spec.config["qpos_noise_high"_])) {
      has_qpos_noise_range_ = true;
      qpos_noise_low_ = spec.config["qpos_noise_low"_];
      qpos_noise_high_ = spec.config["qpos_noise_high"_];
    }
  }

  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->na != spec_.config["act_dim"_] ||
        model_->nu != spec_.config["action_dim"_]) {
      throw std::runtime_error("TableTennis dims do not match model.");
    }
  }

  void CacheIds() {
    pelvis_site_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "pelvis");
    palm_site_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "S_grasp");
    paddle_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "paddle");
    paddle_site_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "paddle");
    paddle_geom_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_GEOM, "pad");
    ball_body_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_BODY, "pingpong");
    ball_site_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_SITE, "pingpong");
    ball_geom_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_GEOM, "pingpong");
    own_half_geom_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_GEOM, "coll_own_half");
    opponent_half_geom_id_ = challenge_extra_detail::RequireId(
        model_, mjOBJ_GEOM, "coll_opponent_half");
    ground_geom_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_GEOM, "ground");
    net_geom_id_ =
        challenge_extra_detail::RequireId(model_, mjOBJ_GEOM, "coll_net");
    ball_vel_sensor_id_ = challenge_extra_detail::RequireId(
        model_, mjOBJ_SENSOR, "pingpong_vel_sensor");
    paddle_vel_sensor_id_ = challenge_extra_detail::RequireId(
        model_, mjOBJ_SENSOR, "paddle_vel_sensor");
    flex_extension_joint_id_ = challenge_extra_detail::RequireId(
        model_, mjOBJ_JOINT, "flex_extension");
    ball_joint_id_ = challenge_extra_detail::RequireId(model_, mjOBJ_JOINT,
                                                       "pingpong_freejoint");
    ball_qposadr_ = model_->jnt_qposadr[ball_joint_id_];
    ball_dofadr_ = model_->jnt_dofadr[ball_joint_id_];
    for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
      const char* raw_name = mj_id2name(model_, mjOBJ_JOINT, joint_id);
      if (raw_name == nullptr) {
        continue;
      }
      std::string_view name(raw_name);
      if (name.rfind("ping", 0) == 0 || name == "paddle_freejoint") {
        continue;
      }
      body_qposadrs_.push_back(model_->jnt_qposadr[joint_id]);
      body_dofadrs_.push_back(model_->jnt_dofadr[joint_id]);
    }
    int expected_obs = 3 + static_cast<int>(body_qposadrs_.size()) +
                       static_cast<int>(body_dofadrs_.size()) + 3 + 3 + 3 + 3 +
                       4 + 3 + 6 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("TableTennis obs_dim does not match model.");
    }
  }

  void RestoreModelState() {
    detail::RestoreModelBodyPos(model_, ball_body_id_, default_ball_body_pos_);
    detail::RestoreModelGeomFriction(model_, ball_geom_id_,
                                     default_ball_geom_friction_);
    detail::RestoreModelBodyMass(model_, paddle_body_id_,
                                 default_paddle_body_mass_);
  }

  void ApplyModelRandomization() {
    if (has_paddle_mass_range_) {
      detail::RestoreModelBodyMass(
          model_, paddle_body_id_,
          challenge_detail::UniformScalar(&gen_, paddle_mass_low_,
                                          paddle_mass_high_));
    }
    if (has_ball_friction_range_) {
      auto friction = challenge_detail::UniformVec3(&gen_, ball_friction_low_,
                                                    ball_friction_high_);
      detail::RestoreModelGeomFriction(
          model_, ball_geom_id_,
          std::vector<mjtNum>(friction.begin(), friction.end()));
    }
    if (has_ball_xyz_range_) {
      auto ball_pos =
          challenge_detail::UniformVec3(&gen_, ball_xyz_low_, ball_xyz_high_);
      detail::RestoreModelBodyPos(model_, ball_body_id_,
                                  {ball_pos[0], ball_pos[1], ball_pos[2]});
      init_qpos_[ball_qposadr_ + 0] = ball_pos[0];
      init_qpos_[ball_qposadr_ + 1] = ball_pos[1];
      init_qpos_[ball_qposadr_ + 2] = ball_pos[2];
    }
    if (ball_qvel_) {
      auto vel_bounds = CalcBallQvelBounds({init_qpos_[ball_qposadr_ + 0],
                                            init_qpos_[ball_qposadr_ + 1],
                                            init_qpos_[ball_qposadr_ + 2]});
      for (int axis = 0; axis < 3; ++axis) {
        std::uniform_real_distribution<double> dist(vel_bounds[1][axis],
                                                    vel_bounds[0][axis]);
        init_qvel_[ball_dofadr_ + axis] = static_cast<mjtNum>(dist(gen_));
      }
    }
  }

  void ApplyResetState() {
    std::vector<mjtNum> qpos = init_qpos_;
    std::vector<mjtNum> qvel = init_qvel_;
    if (!test_reset_qpos_.empty()) {
      qpos = test_reset_qpos_;
      if (!test_reset_qvel_.empty()) {
        qvel = test_reset_qvel_;
      }
    } else if (has_qpos_noise_range_) {
      for (int joint_id = 0; joint_id < model_->njnt - 2; ++joint_id) {
        mjtNum joint_span = model_->jnt_range[joint_id * 2 + 1] -
                            model_->jnt_range[joint_id * 2 + 0];
        std::uniform_real_distribution<double> dist(qpos_noise_low_,
                                                    qpos_noise_high_);
        int adr = model_->jnt_qposadr[joint_id];
        qpos[adr] += static_cast<mjtNum>(dist(gen_)) * joint_span;
        qpos[adr] = challenge_extra_detail::Clip(
            qpos[adr], model_->jnt_range[joint_id * 2 + 0],
            model_->jnt_range[joint_id * 2 + 1]);
      }
    }
    detail::RestoreVector(qpos, data_->qpos);
    detail::RestoreVector(qvel, data_->qvel);
    mj_forward(model_, data_);
    bool rerun_forward = false;
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
      rerun_forward = true;
    }
    if (!test_reset_act_dot_.empty()) {
      detail::RestoreVector(test_reset_act_dot_, data_->act_dot);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
      rerun_forward = true;
    }
    if (rerun_forward) {
      mj_forward(model_, data_);
    }
  }

  void ApplyOracleModelOverrides() {
    if (test_ball_body_pos_.size() == 3) {
      detail::RestoreModelBodyPos(model_, ball_body_id_, test_ball_body_pos_);
    }
    if (!test_ball_geom_friction_.empty()) {
      detail::RestoreModelGeomFriction(model_, ball_geom_id_,
                                       test_ball_geom_friction_);
    }
    if (!test_paddle_body_mass_.empty()) {
      detail::RestoreModelBodyMass(model_, paddle_body_id_,
                                   test_paddle_body_mass_[0]);
    }
    if (test_init_qpos_.size() == static_cast<std::size_t>(model_->nq)) {
      init_qpos_ = test_init_qpos_;
    } else if (test_ball_body_pos_.size() == 3) {
      init_qpos_[ball_qposadr_ + 0] = test_ball_body_pos_[0];
      init_qpos_[ball_qposadr_ + 1] = test_ball_body_pos_[1];
      init_qpos_[ball_qposadr_ + 2] = test_ball_body_pos_[2];
    }
    if (test_init_qvel_.size() == static_cast<std::size_t>(model_->nv)) {
      init_qvel_ = test_init_qvel_;
    }
  }

  std::array<std::array<mjtNum, 3>, 2> CalcBallQvelBounds(
      const std::array<mjtNum, 3>& ball_pos) {
    std::array<mjtNum, 3> table_upper = {1.35, 0.70, 0.785};
    std::array<mjtNum, 3> table_lower = {0.5, -0.60, 0.785};
    constexpr mjtNum gravity = 9.81;
    mjtNum vz = challenge_detail::UniformScalar(&gen_, -0.1, 0.1);
    mjtNum a = -0.5 * gravity;
    mjtNum b = vz;
    mjtNum c = ball_pos[2] - table_upper[2];
    mjtNum disc = b * b - 4.0 * a * c;
    mjtNum t =
        (-b - std::sqrt(std::max(disc, static_cast<mjtNum>(0.0)))) / (2.0 * a);
    return {{
        {(table_upper[0] - ball_pos[0]) / t, (table_upper[1] - ball_pos[1]) / t,
         vz},
        {(table_lower[0] - ball_pos[0]) / t, (table_lower[1] - ball_pos[1]) / t,
         vz},
    }};
  }

  void RelaunchBall() {
    std::array<mjtNum, 3> ball_pos = {init_qpos_[ball_qposadr_ + 0],
                                      init_qpos_[ball_qposadr_ + 1],
                                      init_qpos_[ball_qposadr_ + 2]};
    std::array<mjtNum, 6> ball_vel = {
        init_qvel_[ball_dofadr_ + 0], init_qvel_[ball_dofadr_ + 1],
        init_qvel_[ball_dofadr_ + 2], init_qvel_[ball_dofadr_ + 3],
        init_qvel_[ball_dofadr_ + 4], init_qvel_[ball_dofadr_ + 5]};
    if (has_ball_xyz_range_) {
      auto sampled =
          challenge_detail::UniformVec3(&gen_, ball_xyz_low_, ball_xyz_high_);
      ball_pos = {sampled[0], sampled[1], sampled[2]};
    }
    if (ball_qvel_) {
      auto vel_bounds = CalcBallQvelBounds(ball_pos);
      for (int axis = 0; axis < 3; ++axis) {
        std::uniform_real_distribution<double> dist(vel_bounds[1][axis],
                                                    vel_bounds[0][axis]);
        ball_vel[axis] = static_cast<mjtNum>(dist(gen_));
      }
    }
    data_->qpos[ball_qposadr_ + 0] = ball_pos[0];
    data_->qpos[ball_qposadr_ + 1] = ball_pos[1];
    data_->qpos[ball_qposadr_ + 2] = ball_pos[2];
    for (int axis = 0; axis < 6; ++axis) {
      data_->qvel[ball_dofadr_ + axis] = ball_vel[axis];
    }
    mj_forward(model_, data_);
  }

  std::array<bool, 6> BallContactLabels() const {
    std::array<bool, 6> labels = {false, false, false, false, false, false};
    for (int i = 0; i < data_->ncon; ++i) {
      const mjContact& contact = data_->contact[i];
      if (model_->geom_bodyid[contact.geom1] == ball_body_id_) {
        labels[GeomIdToLabel(contact.geom2)] = true;
      } else if (model_->geom_bodyid[contact.geom2] == ball_body_id_) {
        labels[GeomIdToLabel(contact.geom1)] = true;
      }
    }
    return labels;
  }

  int GeomIdToLabel(int geom_id) const {
    if (geom_id == paddle_geom_id_) {
      return kPaddle;
    }
    if (geom_id == own_half_geom_id_) {
      return kOwn;
    }
    if (geom_id == opponent_half_geom_id_) {
      return kOpponent;
    }
    if (geom_id == net_geom_id_) {
      return kNet;
    }
    if (geom_id == ground_geom_id_) {
      return kGround;
    }
    return kEnv;
  }

  int EvaluateTrajectory() const {
    bool has_hit_paddle = false;
    bool has_bounced_from_paddle = false;
    bool has_bounced_from_table = false;
    int own_contact_count = 0;
    bool own_contact_phase_done = false;
    for (const auto& labels : contact_trajectory_) {
      if (!labels[kPaddle] && has_hit_paddle) {
        has_bounced_from_paddle = true;
      }
      if (labels[kPaddle] && has_bounced_from_paddle) {
        return kDoubleTouch;
      }
      if (labels[kPaddle]) {
        has_hit_paddle = true;
      }
      if (labels[kOwn]) {
        if (!has_bounced_from_table) {
          has_bounced_from_table = true;
          own_contact_count = 1;
        } else if (!own_contact_phase_done) {
          ++own_contact_count;
          if (own_contact_count > 2) {
            own_contact_phase_done = true;
            return kOwnHalf;
          }
        } else {
          return kOwnHalf;
        }
      } else if (has_bounced_from_table) {
        own_contact_phase_done = true;
      }
      if (labels[kOpponent]) {
        return has_hit_paddle ? kSuccess : kNoPaddle;
      }
    }
    return kMiss;
  }

  RewardInfo ComputeRewardInfo() {
    RewardInfo reward;
    auto labels = BallContactLabels();
    contact_trajectory_.push_back(labels);
    for (int i = 0; i < 6; ++i) {
      reward.touching_info[i] = labels[i] ? 1.0 : 0.0;
    }
    std::array<mjtNum, 3> ball_pos{};
    detail::CopySitePos(model_, data_, ball_site_id_, ball_pos.data());
    std::array<mjtNum, 3> paddle_pos{};
    detail::CopySitePos(model_, data_, paddle_site_id_, paddle_pos.data());
    std::array<mjtNum, 3> palm_pos{};
    detail::CopySitePos(model_, data_, palm_site_id_, palm_pos.data());
    auto reach_err = challenge_detail::Sub3(paddle_pos, ball_pos);
    auto palm_err = challenge_detail::Sub3(palm_pos, paddle_pos);
    reward.reach_dist = std::exp(-challenge_detail::Norm3(reach_err));
    reward.palm_dist = std::exp(-5.0 * challenge_detail::Norm3(palm_err));
    std::array<mjtNum, 4> paddle_quat = {data_->xquat[paddle_body_id_ * 4 + 0],
                                         data_->xquat[paddle_body_id_ * 4 + 1],
                                         data_->xquat[paddle_body_id_ * 4 + 2],
                                         data_->xquat[paddle_body_id_ * 4 + 3]};
    mjtNum quat_err = 0.0;
    for (int i = 0; i < 4; ++i) {
      mjtNum diff = paddle_quat[i] - init_paddle_quat_[i];
      quat_err += diff * diff;
    }
    reward.paddle_quat = std::exp(-5.0 * std::sqrt(quat_err));
    reward.torso_up = std::exp(
        -5.0 *
        std::abs(data_->qpos[model_->jnt_qposadr[flex_extension_joint_id_]]));
    reward.act_reg = -challenge_extra_detail::L2ActReg(model_, data_);
    reward.sparse = labels[kPaddle] ? 1.0 : 0.0;
    int traj = EvaluateTrajectory();
    reward.solved = traj == kSuccess;
    reward.done = data_->time > kMaxTime || ball_pos[2] < 0.3 ||
                  reward.solved || traj == kOwnHalf || traj == kNoPaddle ||
                  traj == kDoubleTouch;
    reward.dense_reward =
        reward_reach_dist_w_ * reward.reach_dist +
        reward_palm_dist_w_ * reward.palm_dist +
        reward_paddle_quat_w_ * reward.paddle_quat +
        reward_act_reg_w_ * reward.act_reg +
        reward_torso_up_w_ * reward.torso_up +
        reward_sparse_w_ * reward.sparse +
        reward_solved_w_ * static_cast<mjtNum>(reward.solved) +
        reward_done_w_ * static_cast<mjtNum>(reward.done);
    return reward;
  }

  std::vector<mjtNum> Observation(const RewardInfo& reward) const {
    std::vector<mjtNum> obs;
    obs.reserve(spec_.config["obs_dim"_]);
    obs.insert(obs.end(), data_->site_xpos + pelvis_site_id_ * 3,
               data_->site_xpos + pelvis_site_id_ * 3 + 3);
    for (int adr : body_qposadrs_) {
      obs.push_back(data_->qpos[adr]);
    }
    for (int adr : body_dofadrs_) {
      obs.push_back(data_->qvel[adr]);
    }
    obs.insert(obs.end(), data_->site_xpos + ball_site_id_ * 3,
               data_->site_xpos + ball_site_id_ * 3 + 3);
    int ball_vel_adr = model_->sensor_adr[ball_vel_sensor_id_];
    obs.insert(obs.end(), data_->sensordata + ball_vel_adr,
               data_->sensordata + ball_vel_adr + 3);
    obs.insert(obs.end(), data_->site_xpos + paddle_site_id_ * 3,
               data_->site_xpos + paddle_site_id_ * 3 + 3);
    int paddle_vel_adr = model_->sensor_adr[paddle_vel_sensor_id_];
    obs.insert(obs.end(), data_->sensordata + paddle_vel_adr,
               data_->sensordata + paddle_vel_adr + 3);
    obs.insert(obs.end(), data_->xquat + paddle_body_id_ * 4,
               data_->xquat + paddle_body_id_ * 4 + 4);
    std::array<mjtNum, 3> ball_pos{};
    std::array<mjtNum, 3> paddle_pos{};
    detail::CopySitePos(model_, data_, ball_site_id_, ball_pos.data());
    detail::CopySitePos(model_, data_, paddle_site_id_, paddle_pos.data());
    auto reach_err = challenge_detail::Sub3(paddle_pos, ball_pos);
    obs.insert(obs.end(), reach_err.begin(), reach_err.end());
    obs.insert(obs.end(), reward.touching_info.begin(),
               reward.touching_info.end());
    obs.insert(obs.end(), data_->act, data_->act + model_->na);
    return obs;
  }

  void WriteState(const RewardInfo& reward, bool reset, float reward_value) {
    auto obs = Observation(reward);
    auto state = Allocate();
    if constexpr (!kFromPixels) {
      AssignObservation("obs", &state["obs"_], obs.data(), obs.size(), reset);
    }
    state["reward"_] = reward_value;
    state["discount"_] = 1.0f;
    state["done"_] = done_;
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    state["info:reach_dist"_] = reward.reach_dist;
    state["info:palm_dist"_] = reward.palm_dist;
    state["info:paddle_quat"_] = reward.paddle_quat;
    state["info:torso_up"_] = reward.torso_up;
    state["info:act_reg"_] = reward.act_reg;
    state["info:sparse"_] = reward.sparse;
    state["info:solved"_] = static_cast<mjtNum>(reward.solved);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:touching_info"_].Assign(reward.touching_info.data(), 6);
    if constexpr (kFromPixels) {
      AssignPixelObservation("obs:pixels", &state["obs:pixels"_], reset);
    }
  }
};

template <typename Spec>
using RunTrackEnvAlias = MyoChallengeRunTrackEnvBase<Spec, false>;

template <typename Spec>
using RunTrackPixelEnvAlias = MyoChallengeRunTrackEnvBase<Spec, true>;

template <typename Spec>
using SoccerEnvAlias = MyoChallengeSoccerEnvBase<Spec, false>;

template <typename Spec>
using SoccerPixelEnvAlias = MyoChallengeSoccerEnvBase<Spec, true>;

template <typename Spec>
using ChaseTagEnvAlias = MyoChallengeChaseTagEnvBase<Spec, false>;

template <typename Spec>
using ChaseTagPixelEnvAlias = MyoChallengeChaseTagEnvBase<Spec, true>;

template <typename Spec>
using TableTennisEnvAlias = MyoChallengeTableTennisEnvBase<Spec, false>;

template <typename Spec>
using TableTennisPixelEnvAlias = MyoChallengeTableTennisEnvBase<Spec, true>;

using RunTrackSpec = MyoChallengeRunTrackEnvSpec;
using RunTrackPixelSpec = MyoChallengeRunTrackPixelEnvSpec;
using SoccerSpec = MyoChallengeSoccerEnvSpec;
using SoccerPixelSpec = MyoChallengeSoccerPixelEnvSpec;
using ChaseTagSpec = MyoChallengeChaseTagEnvSpec;
using ChaseTagPixelSpec = MyoChallengeChaseTagPixelEnvSpec;
using TableTennisSpec = MyoChallengeTableTennisEnvSpec;
using TableTennisPixelSpec = MyoChallengeTableTennisPixelEnvSpec;

using RunTrackEnv = RunTrackEnvAlias<RunTrackSpec>;
using RunTrackPixelEnv = RunTrackPixelEnvAlias<RunTrackPixelSpec>;
using MyoChallengeRunTrackEnv = RunTrackEnv;
using MyoChallengeRunTrackPixelEnv = RunTrackPixelEnv;
using MyoChallengeRunTrackEnvPool = AsyncEnvPool<RunTrackEnv>;
using MyoChallengeRunTrackPixelEnvPool = AsyncEnvPool<RunTrackPixelEnv>;

using SoccerEnv = SoccerEnvAlias<SoccerSpec>;
using SoccerPixelEnv = SoccerPixelEnvAlias<SoccerPixelSpec>;
using MyoChallengeSoccerEnv = SoccerEnv;
using MyoChallengeSoccerPixelEnv = SoccerPixelEnv;
using MyoChallengeSoccerEnvPool = AsyncEnvPool<SoccerEnv>;
using MyoChallengeSoccerPixelEnvPool = AsyncEnvPool<SoccerPixelEnv>;

using ChaseTagEnv = ChaseTagEnvAlias<ChaseTagSpec>;
using ChaseTagPixelEnv = ChaseTagPixelEnvAlias<ChaseTagPixelSpec>;
using MyoChallengeChaseTagEnv = ChaseTagEnv;
using MyoChallengeChaseTagPixelEnv = ChaseTagPixelEnv;
using MyoChallengeChaseTagEnvPool = AsyncEnvPool<ChaseTagEnv>;
using MyoChallengeChaseTagPixelEnvPool = AsyncEnvPool<ChaseTagPixelEnv>;

using TableTennisEnv = TableTennisEnvAlias<TableTennisSpec>;
using TableTennisPixelEnv = TableTennisPixelEnvAlias<TableTennisPixelSpec>;
using MyoChallengeTableTennisEnv = TableTennisEnv;
using MyoChallengeTableTennisPixelEnv = TableTennisPixelEnv;
using MyoChallengeTableTennisEnvPool = AsyncEnvPool<TableTennisEnv>;
using MyoChallengeTableTennisPixelEnvPool = AsyncEnvPool<TableTennisPixelEnv>;

}  // namespace myosuite_envpool

#endif  // ENVPOOL_MUJOCO_MYOSUITE_MYOCHALLENGE_EXTENDED_H_
