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

#ifndef ENVPOOL_MUJOCO_MYOSUITE_MYODM_H_
#define ENVPOOL_MUJOCO_MYOSUITE_MYODM_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "envpool/mujoco/myosuite/myobase.h"

namespace myosuite_envpool {

namespace detail {

inline void ReplaceAll(std::string* text, std::string_view from,
                       std::string_view to) {
  std::size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.size(), to);
    pos += to.size();
  }
}

inline int ProcessId() {
#ifdef _WIN32
  return _getpid();
#else
  return getpid();
#endif
}

inline std::string TemporaryDirectory() {
  for (const char* env_key : {"TMPDIR", "TEMP", "TMP"}) {
    const char* env_value = std::getenv(env_key);
    if (env_value != nullptr && env_value[0] != '\0') {
      return env_value;
    }
  }
#ifdef _WIN32
  return "C:\\Windows\\Temp";
#else
  return "/tmp";
#endif
}

inline std::string MakeTempPath(const std::string& stem,
                                std::string_view suffix) {
  static std::atomic<int> counter{0};
  std::string path = TemporaryDirectory();
  if (!path.empty() && path.back() != '/' && path.back() != '\\') {
#ifdef _WIN32
    path.push_back('\\');
#else
    path.push_back('/');
#endif
  }
  path += stem + "_" + std::to_string(ProcessId()) + "_" +
          std::to_string(counter.fetch_add(1)) + std::string(suffix);
  return path;
}

inline std::string ReadTextFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

inline void WriteTextFile(const std::string& path, const std::string& text) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to create file: " + path);
  }
  output << text;
}

inline std::string MyoSuiteAbsolutePath(const std::string& base_path,
                                        std::string_view relative_path) {
  return myosuite::MyoSuiteModelPath(base_path, relative_path);
}

inline std::string BuildTrackModelPath(const std::string& base_path,
                                       std::string_view relative_model_path,
                                       std::string_view object_name) {
  std::string asset_root = MyoSuiteAbsolutePath(base_path, "");
  if (!asset_root.empty() && asset_root.back() == '/') {
    asset_root.pop_back();
  }
  std::string source_model =
      asset_root + "/" + std::string(relative_model_path);
  std::string object_xml = ReadTextFile(source_model);
  std::string tabletop_xml =
      ReadTextFile(asset_root + "/envs/myo/assets/hand/myohand_tabletop.xml");
  std::string hand_assets_xml = ReadTextFile(
      asset_root + "/simhive/myo_sim/hand/assets/myohand_assets.xml");

  std::string stem = "envpool_myodm_track";
  std::string prefix = MakeTempPath(stem, "");
  std::string hand_assets_tmp = prefix + "_assets.xml";
  std::string tabletop_tmp = prefix + "_tabletop.xml";
  std::string object_tmp = prefix + "_object.xml";

  std::string myo_sim_root = asset_root + "/simhive/myo_sim";
  ReplaceAll(
      &hand_assets_xml, R"(meshdir=".." texturedir="..")",
      "meshdir=\"" + myo_sim_root + "\" texturedir=\"" + myo_sim_root + "\"");
  WriteTextFile(hand_assets_tmp, hand_assets_xml);

  ReplaceAll(&tabletop_xml,
             "../../../../simhive/myo_sim/hand/assets/myohand_assets.xml",
             hand_assets_tmp);
  ReplaceAll(
      &tabletop_xml,
      "../../../../simhive/furniture_sim/simpleTable/simpleTable_asset.xml",
      asset_root + "/simhive/furniture_sim/simpleTable/simpleTable_asset.xml");
  ReplaceAll(&tabletop_xml,
             "../../../../simhive/myo_sim/hand/assets/myohand_body.xml",
             asset_root + "/simhive/myo_sim/hand/assets/myohand_body.xml");
  ReplaceAll(
      &tabletop_xml,
      "../../../../simhive/furniture_sim/simpleTable/"
      "simpleGraniteTable_body.xml",
      asset_root +
          "/simhive/furniture_sim/simpleTable/simpleGraniteTable_body.xml");
  ReplaceAll(
      &tabletop_xml,
      "meshdir=\"../../../../simhive/myo_sim/\" "
      "texturedir=\"../../../../simhive/myo_sim/\"",
      "meshdir=\"" + myo_sim_root + "\" texturedir=\"" + myo_sim_root + "\"");
  WriteTextFile(tabletop_tmp, tabletop_xml);

  ReplaceAll(&object_xml, "OBJECT_NAME", object_name);
  ReplaceAll(&object_xml, "myohand_tabletop.xml", tabletop_tmp);
  ReplaceAll(&object_xml, "../../../../simhive/object_sim/common.xml",
             asset_root + "/simhive/object_sim/common.xml");
  ReplaceAll(&object_xml,
             "../../../../simhive/object_sim/" + std::string(object_name) +
                 "/assets.xml",
             asset_root + "/simhive/object_sim/" + std::string(object_name) +
                 "/assets.xml");
  ReplaceAll(&object_xml,
             "../../../../simhive/object_sim/" + std::string(object_name) +
                 "/body.xml",
             asset_root + "/simhive/object_sim/" + std::string(object_name) +
                 "/body.xml");
  WriteTextFile(object_tmp, object_xml);
  return object_tmp;
}

inline void RemoveTrackTemporaryFiles(const std::string& model_path) {
  std::remove(model_path.c_str());
  std::size_t object_pos = model_path.rfind("_object.xml");
  if (object_pos == std::string::npos) {
    return;
  }
  std::string prefix = model_path.substr(0, object_pos);
  std::remove((prefix + "_tabletop.xml").c_str());
  std::remove((prefix + "_assets.xml").c_str());
}

inline std::uint16_t ReadLe16(std::istream* input) {
  std::array<unsigned char, 2> data{};
  input->read(reinterpret_cast<char*>(data.data()), data.size());
  return static_cast<std::uint16_t>(data[0]) |
         (static_cast<std::uint16_t>(data[1]) << 8);
}

inline std::uint32_t ReadLe32(std::istream* input) {
  std::array<unsigned char, 4> data{};
  input->read(reinterpret_cast<char*>(data.data()), data.size());
  return static_cast<std::uint32_t>(data[0]) |
         (static_cast<std::uint32_t>(data[1]) << 8) |
         (static_cast<std::uint32_t>(data[2]) << 16) |
         (static_cast<std::uint32_t>(data[3]) << 24);
}

struct NpyArray {
  std::vector<double> values;
  std::vector<int> shape;
};

inline NpyArray ParseNpyPayload(const std::vector<char>& payload) {
  if (payload.size() < 10 || std::memcmp(payload.data(), "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("Unsupported NPY payload.");
  }
  auto major = static_cast<unsigned char>(payload[6]);
  auto minor = static_cast<unsigned char>(payload[7]);
  (void)minor;
  std::size_t offset = 8;
  std::size_t header_len = 0;
  if (major == 1) {
    header_len = static_cast<unsigned char>(payload[offset]) |
                 (static_cast<std::size_t>(
                      static_cast<unsigned char>(payload[offset + 1]))
                  << 8);
    offset += 2;
  } else if (major == 2) {
    header_len = static_cast<unsigned char>(payload[offset]) |
                 (static_cast<std::size_t>(
                      static_cast<unsigned char>(payload[offset + 1]))
                  << 8) |
                 (static_cast<std::size_t>(
                      static_cast<unsigned char>(payload[offset + 2]))
                  << 16) |
                 (static_cast<std::size_t>(
                      static_cast<unsigned char>(payload[offset + 3]))
                  << 24);
    offset += 4;
  } else {
    throw std::runtime_error("Unsupported NPY version.");
  }
  if (offset + header_len > payload.size()) {
    throw std::runtime_error("Malformed NPY header.");
  }
  std::string header(payload.data() + offset, header_len);
  if (header.find("'descr': '<f8'") == std::string::npos ||
      header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error("Unsupported NPY dtype/order.");
  }
  std::size_t shape_pos = header.find("'shape':");
  if (shape_pos == std::string::npos) {
    throw std::runtime_error("Missing NPY shape.");
  }
  std::size_t open = header.find('(', shape_pos);
  std::size_t close = header.find(')', open);
  if (open == std::string::npos || close == std::string::npos) {
    throw std::runtime_error("Malformed NPY shape.");
  }
  std::vector<int> shape;
  std::string shape_text = header.substr(open + 1, close - open - 1);
  std::size_t start = 0;
  while (start < shape_text.size()) {
    std::size_t end = shape_text.find(',', start);
    if (end == std::string::npos) {
      end = shape_text.size();
    }
    std::string token = shape_text.substr(start, end - start);
    token.erase(
        std::remove_if(token.begin(), token.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; }),
        token.end());
    if (!token.empty()) {
      shape.push_back(std::stoi(token));
    }
    start = end + 1;
  }
  if (shape.empty()) {
    throw std::runtime_error("Empty NPY shape.");
  }
  std::size_t count = 1;
  for (int dim : shape) {
    count *= static_cast<std::size_t>(dim);
  }
  offset += header_len;
  std::size_t data_bytes = count * sizeof(double);
  if (offset + data_bytes > payload.size()) {
    throw std::runtime_error("Malformed NPY payload size.");
  }
  NpyArray array;
  array.shape = std::move(shape);
  array.values.resize(count);
  std::memcpy(array.values.data(), payload.data() + offset, data_bytes);
  return array;
}

inline std::vector<char> ReadZipStoredEntry(std::istream* input,
                                            std::uint32_t size) {
  std::vector<char> payload(size);
  input->read(payload.data(), static_cast<std::streamsize>(payload.size()));
  if (!*input) {
    throw std::runtime_error("Failed to read NPZ payload.");
  }
  return payload;
}

inline std::vector<std::pair<std::string, NpyArray>> LoadStoredNpzArrays(
    const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open NPZ file: " + path);
  }
  std::vector<std::pair<std::string, NpyArray>> arrays;
  while (true) {
    std::uint32_t signature = ReadLe32(&input);
    if (!input) {
      break;
    }
    if (signature == 0x06054b50 || signature == 0x02014b50) {
      break;
    }
    if (signature != 0x04034b50) {
      throw std::runtime_error("Unsupported NPZ zip record.");
    }
    ReadLe16(&input);  // version
    std::uint16_t flags = ReadLe16(&input);
    std::uint16_t compression = ReadLe16(&input);
    ReadLe16(&input);  // mod time
    ReadLe16(&input);  // mod date
    ReadLe32(&input);  // crc32
    std::uint32_t compressed_size = ReadLe32(&input);
    std::uint32_t uncompressed_size = ReadLe32(&input);
    std::uint16_t name_len = ReadLe16(&input);
    std::uint16_t extra_len = ReadLe16(&input);
    if (flags != 0 || compression != 0 ||
        compressed_size != uncompressed_size) {
      throw std::runtime_error("Only stored NPZ entries are supported.");
    }
    std::string name(name_len, '\0');
    input.read(name.data(), static_cast<std::streamsize>(name.size()));
    input.seekg(extra_len, std::ios::cur);
    auto payload = ReadZipStoredEntry(&input, compressed_size);
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".npy") {
      name.erase(name.size() - 4);
    }
    arrays.emplace_back(std::move(name), ParseNpyPayload(payload));
  }
  return arrays;
}

inline std::array<mjtNum, 4> Mat9ToQuat(const mjtNum* mat) {
  mjtNum trace = mat[0] + mat[4] + mat[8];
  std::array<mjtNum, 4> quat{};
  if (trace > 0.0) {
    mjtNum s = std::sqrt(trace + 1.0) * 2.0;
    quat[0] = 0.25 * s;
    quat[1] = (mat[7] - mat[5]) / s;
    quat[2] = (mat[2] - mat[6]) / s;
    quat[3] = (mat[3] - mat[1]) / s;
  } else if (mat[0] > mat[4] && mat[0] > mat[8]) {
    mjtNum s = std::sqrt(1.0 + mat[0] - mat[4] - mat[8]) * 2.0;
    quat[0] = (mat[7] - mat[5]) / s;
    quat[1] = 0.25 * s;
    quat[2] = (mat[1] + mat[3]) / s;
    quat[3] = (mat[2] + mat[6]) / s;
  } else if (mat[4] > mat[8]) {
    mjtNum s = std::sqrt(1.0 + mat[4] - mat[0] - mat[8]) * 2.0;
    quat[0] = (mat[2] - mat[6]) / s;
    quat[1] = (mat[1] + mat[3]) / s;
    quat[2] = 0.25 * s;
    quat[3] = (mat[5] + mat[7]) / s;
  } else {
    mjtNum s = std::sqrt(1.0 + mat[8] - mat[0] - mat[4]) * 2.0;
    quat[0] = (mat[3] - mat[1]) / s;
    quat[1] = (mat[2] + mat[6]) / s;
    quat[2] = (mat[5] + mat[7]) / s;
    quat[3] = 0.25 * s;
  }
  if (quat[0] < 0.0) {
    for (mjtNum& value : quat) {
      value = -value;
    }
  }
  return quat;
}

inline std::array<mjtNum, 9> QuatToMat(const std::array<mjtNum, 4>& quat) {
  mjtNum w = quat[0];
  mjtNum x = quat[1];
  mjtNum y = quat[2];
  mjtNum z = quat[3];
  mjtNum nq = w * w + x * x + y * y + z * z;
  mjtNum s = nq > 0.0 ? 2.0 / nq : 0.0;
  mjtNum x_scaled = x * s;
  mjtNum y_scaled = y * s;
  mjtNum z_scaled = z * s;
  mjtNum w_x = w * x_scaled;
  mjtNum w_y = w * y_scaled;
  mjtNum w_z = w * z_scaled;
  mjtNum x_x = x * x_scaled;
  mjtNum x_y = x * y_scaled;
  mjtNum x_z = x * z_scaled;
  mjtNum y_y = y * y_scaled;
  mjtNum y_z = y * z_scaled;
  mjtNum z_z = z * z_scaled;
  return {
      1.0 - (y_y + z_z), x_y - w_z,         x_z + w_y,
      x_y + w_z,         1.0 - (x_x + z_z), y_z - w_x,
      x_z - w_y,         y_z + w_x,         1.0 - (x_x + y_y),
  };
}

inline std::array<mjtNum, 3> Mat9ToEuler(const std::array<mjtNum, 9>& mat) {
  constexpr mjtNum eps4 =
      std::numeric_limits<mjtNum>::epsilon() * static_cast<mjtNum>(4.0);
  mjtNum cy = std::sqrt(mat[8] * mat[8] + mat[5] * mat[5]);
  bool condition = cy > eps4;
  mjtNum rz =
      condition ? -std::atan2(mat[1], mat[0]) : -std::atan2(-mat[3], mat[4]);
  mjtNum ry = -std::atan2(-mat[2], cy);
  mjtNum rx = condition ? -std::atan2(mat[5], mat[8]) : 0.0;
  return {rx, ry, rz};
}

inline std::array<mjtNum, 3> QuatToEuler(const std::array<mjtNum, 4>& quat) {
  return Mat9ToEuler(QuatToMat(quat));
}

inline std::array<mjtNum, 4> QuatConjugate(const std::array<mjtNum, 4>& quat) {
  return {quat[0], -quat[1], -quat[2], -quat[3]};
}

inline std::array<mjtNum, 4> QuatMul(const std::array<mjtNum, 4>& lhs,
                                     const std::array<mjtNum, 4>& rhs) {
  return {
      lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] - lhs[3] * rhs[3],
      lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[3] - lhs[3] * rhs[2],
      lhs[0] * rhs[2] - lhs[1] * rhs[3] + lhs[2] * rhs[0] + lhs[3] * rhs[1],
      lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1] + lhs[3] * rhs[0],
  };
}

inline mjtNum QuatAngle(const std::array<mjtNum, 4>& quat) {
  mjtNum sin_half =
      std::sqrt(quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
  return static_cast<mjtNum>(2.0) * std::atan2(sin_half, quat[0]);
}

inline mjtNum QuatDistance(const std::array<mjtNum, 4>& lhs,
                           const std::array<mjtNum, 4>& rhs) {
  return std::abs(QuatAngle(QuatMul(rhs, QuatConjugate(lhs))));
}

inline mjtNum RoundReferenceTime(mjtNum time) {
  return std::round(time * static_cast<mjtNum>(10000.0)) /
         static_cast<mjtNum>(10000.0);
}

}  // namespace detail

class MyoDMTrackEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "object_name"_.Bind(std::string()),
        "reference_path"_.Bind(std::string()),
        "reference_time"_.Bind(std::vector<double>{}),
        "reference_robot"_.Bind(std::vector<double>{}),
        "reference_robot_vel"_.Bind(std::vector<double>{}),
        "reference_object"_.Bind(std::vector<double>{}),
        "reference_robot_init"_.Bind(std::vector<double>{}),
        "reference_object_init"_.Bind(std::vector<double>{}),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0), "robot_dim"_.Bind(0), "object_dim"_.Bind(7),
        "robot_horizon"_.Bind(0), "object_horizon"_.Bind(0),
        "reference_has_robot_vel"_.Bind(false), "motion_start_time"_.Bind(0.0),
        "motion_extrapolation"_.Bind(true), "reward_pose_w"_.Bind(0.0),
        "reward_object_w"_.Bind(1.0), "reward_bonus_w"_.Bind(1.0),
        "reward_penalty_w"_.Bind(-2.0), "terminate_obj_fail"_.Bind(true),
        "terminate_pose_fail"_.Bind(false),
        "test_playback_reference"_.Bind(false),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_ctrl"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_act_dot"_.Bind(std::vector<double>{}),
        "test_reset_integration_state"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_reference_time"_.Bind(std::vector<double>{}),
        "test_reference_robot"_.Bind(std::vector<double>{}),
        "test_reference_robot_vel"_.Bind(std::vector<double>{}),
        "test_reference_object"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:pose"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:object"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:bonus"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:penalty"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:target_pos"_.Bind(Spec<mjtNum>({3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using MyoDMTrackEnvSpec = EnvSpec<MyoDMTrackEnvFns>;
using MyoDMTrackPixelEnvFns = PixelObservationEnvFns<MyoDMTrackEnvFns>;
using MyoDMTrackPixelEnvSpec = EnvSpec<MyoDMTrackPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class MyoDMTrackEnvBase : public Env<EnvSpecT>,
                          public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::spec_;

  struct ReferenceState {
    std::vector<mjtNum> robot;
    std::vector<mjtNum> robot_vel;
    std::vector<mjtNum> object;
  };

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum pose{0.0};
    mjtNum object{0.0};
    mjtNum bonus{0.0};
    mjtNum penalty{0.0};
    bool success{false};
    bool done{false};
  };

  enum class ReferenceType : std::uint8_t { kFixed, kRandom, kTrack };

  bool normalize_act_;
  bool motion_extrapolation_;
  bool reference_has_robot_vel_;
  bool terminate_obj_fail_;
  bool terminate_pose_fail_;
  mjtNum motion_start_time_;
  mjtNum reward_pose_w_;
  mjtNum reward_object_w_;
  mjtNum reward_bonus_w_;
  mjtNum reward_penalty_w_;
  mjtNum obj_err_scale_{50.0};
  mjtNum base_err_scale_{40.0};
  mjtNum lift_bonus_thresh_{0.02};
  mjtNum lift_bonus_mag_{1.0};
  mjtNum qpos_reward_weight_{0.35};
  mjtNum qpos_err_scale_{5.0};
  mjtNum qvel_reward_weight_{0.05};
  mjtNum qvel_err_scale_{0.1};
  mjtNum obj_fail_thresh_{0.25};
  mjtNum base_fail_thresh_{0.25};
  mjtNum qpos_fail_thresh_{0.75};
  std::string object_name_;
  int target_sid_{-1};
  int object_bid_{-1};
  int wrist_bid_{-1};
  int body_geom_id_{-1};
  mjtNum lift_z_{0.0};
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;
  std::vector<mjtNum> reference_time_;
  std::vector<mjtNum> reference_robot_;
  std::vector<mjtNum> reference_robot_vel_;
  std::vector<mjtNum> reference_object_;
  std::vector<mjtNum> reference_robot_init_;
  std::vector<mjtNum> reference_object_init_;
  int robot_dim_{0};
  int object_dim_{0};
  int robot_horizon_{0};
  int object_horizon_{0};
  int horizon_{0};
  int reference_index_cache_{0};
  int playback_reference_index_{0};
  ReferenceType reference_type_{ReferenceType::kFixed};
  detail::NumpyPcg64 reference_rng_{0};
  ReferenceState current_reference_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_ctrl_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_act_dot_;
  std::vector<mjtNum> test_reset_integration_state_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_reference_time_;
  std::vector<mjtNum> test_reference_robot_;
  std::vector<mjtNum> test_reference_robot_vel_;
  std::vector<mjtNum> test_reference_object_;
  bool test_playback_reference_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoDMTrackEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            detail::BuildTrackModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_],
                                        spec.config["object_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        motion_extrapolation_(spec.config["motion_extrapolation"_]),
        reference_has_robot_vel_(spec.config["reference_has_robot_vel"_]),
        terminate_obj_fail_(spec.config["terminate_obj_fail"_]),
        terminate_pose_fail_(spec.config["terminate_pose_fail"_]),
        motion_start_time_(spec.config["motion_start_time"_]),
        reward_pose_w_(spec.config["reward_pose_w"_]),
        reward_object_w_(spec.config["reward_object_w"_]),
        reward_bonus_w_(spec.config["reward_bonus_w"_]),
        reward_penalty_w_(spec.config["reward_penalty_w"_]),
        object_name_(spec.config["object_name"_]),
        robot_dim_(spec.config["robot_dim"_]),
        object_dim_(spec.config["object_dim"_]),
        robot_horizon_(spec.config["robot_horizon"_]),
        object_horizon_(spec.config["object_horizon"_]),
        reference_rng_(static_cast<std::uint64_t>(this->seed_)),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_ctrl_(detail::ToMjtVector(spec.config["test_reset_ctrl"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_act_dot_(
            detail::ToMjtVector(spec.config["test_reset_act_dot"_])),
        test_reset_integration_state_(
            detail::ToMjtVector(spec.config["test_reset_integration_state"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_reference_time_(
            detail::ToMjtVector(spec.config["test_reference_time"_])),
        test_reference_robot_(
            detail::ToMjtVector(spec.config["test_reference_robot"_])),
        test_reference_robot_vel_(
            detail::ToMjtVector(spec.config["test_reference_robot_vel"_])),
        test_reference_object_(
            detail::ToMjtVector(spec.config["test_reference_object"_])),
        test_playback_reference_(spec.config["test_playback_reference"_]) {
    LoadReference(spec);
    ValidateConfig();
    mj_forward(model_, data_);
    CacheObjects();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    detail::AdjustInitialQposForNormalizedActions(model_, data_,
                                                  normalize_act_);
    InitializeReferencePose();
    InitializeRobotEnv();
    PrimeReferenceRngForOracleResetBehavior();
    detail::RemoveTrackTemporaryFiles(model_path_);
  }

  ~MyoDMTrackEnvBase() override = default;

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
    reference_index_cache_ = 0;
    playback_reference_index_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    ApplyResetState();
    ReferenceState reference =
        test_playback_reference_
            ? PlaybackReferenceAtCurrentIndex()
            : ReferenceAt(data_->time + motion_start_time_);
    UpdateTargetSite(reference);
    CaptureResetState();
    RewardInfo reward = ComputeReward(reference);
    WriteState(reward, reference, true, 0.0);
  }

  void Step(const Action& action) override {
    if (test_playback_reference_) {
      InvalidateRenderCache();
      if (playback_reference_index_ + 1 < horizon_) {
        ++playback_reference_index_;
      }
      ReferenceState reference = PlaybackReferenceAtCurrentIndex();
      SetQposFromReference(reference);
      mj_forward(model_, data_);
      data_->time += Dt();
      ++elapsed_step_;
      RewardInfo reward = ComputeReward(reference);
      done_ = false;
      WriteState(reward, reference, false, reward.dense_reward);
      return;
    }
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
    detail::RefreshObservedMyoSuiteState(model_, data_);
    ++elapsed_step_;
    ReferenceState reference = ReferenceAt(data_->time + motion_start_time_);
    UpdateTargetSite(reference);
    RewardInfo reward = ComputeReward(reference);
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, reference, false, reward.dense_reward);
  }

 private:
  void LoadReference(const Spec& spec) {
    std::string reference_path = spec.config["reference_path"_];
    if (!reference_path.empty()) {
      auto arrays = detail::LoadStoredNpzArrays(
          myosuite::MyoSuiteAssetRoot(spec.config["base_path"_]) + "/" +
          reference_path);
      for (auto& [name, array] : arrays) {
        if (name == "time") {
          reference_time_.assign(array.values.begin(), array.values.end());
        } else if (name == "robot") {
          robot_horizon_ = array.shape[0];
          robot_dim_ = array.shape[1];
          reference_robot_.assign(array.values.begin(), array.values.end());
        } else if (name == "robot_vel") {
          reference_has_robot_vel_ = true;
          reference_robot_vel_.assign(array.values.begin(), array.values.end());
        } else if (name == "object") {
          object_horizon_ = array.shape[0];
          object_dim_ = array.shape[1];
          reference_object_.assign(array.values.begin(), array.values.end());
        } else if (name == "robot_init") {
          reference_robot_init_.assign(array.values.begin(),
                                       array.values.end());
        } else if (name == "object_init") {
          reference_object_init_.assign(array.values.begin(),
                                        array.values.end());
        }
      }
    } else {
      reference_time_ = detail::ToMjtVector(spec.config["reference_time"_]);
      reference_robot_ = detail::ToMjtVector(spec.config["reference_robot"_]);
      reference_robot_vel_ =
          detail::ToMjtVector(spec.config["reference_robot_vel"_]);
      reference_object_ = detail::ToMjtVector(spec.config["reference_object"_]);
      reference_robot_init_ =
          detail::ToMjtVector(spec.config["reference_robot_init"_]);
      reference_object_init_ =
          detail::ToMjtVector(spec.config["reference_object_init"_]);
    }
    if (!test_reference_time_.empty()) {
      int horizon = static_cast<int>(test_reference_time_.size());
      if (robot_dim_ <= 0 || object_dim_ <= 0) {
        throw std::runtime_error(
            "TrackEnv test reference dims are not initialized.");
      }
      if (static_cast<int>(test_reference_robot_.size()) !=
              horizon * robot_dim_ ||
          static_cast<int>(test_reference_object_.size()) !=
              horizon * object_dim_) {
        throw std::runtime_error(
            "TrackEnv test reference arrays have wrong size.");
      }
      reference_time_ = test_reference_time_;
      reference_robot_ = test_reference_robot_;
      reference_object_ = test_reference_object_;
      reference_robot_init_.assign(reference_robot_.begin(),
                                   reference_robot_.begin() + robot_dim_);
      reference_object_init_.assign(reference_object_.begin(),
                                    reference_object_.begin() + object_dim_);
      robot_horizon_ = horizon;
      object_horizon_ = horizon;
      if (reference_has_robot_vel_) {
        if (static_cast<int>(test_reference_robot_vel_.size()) !=
            horizon * robot_dim_) {
          throw std::runtime_error(
              "TrackEnv test reference robot_vel has wrong size.");
        }
        reference_robot_vel_ = test_reference_robot_vel_;
      } else {
        reference_robot_vel_.clear();
      }
    }
    if (reference_has_robot_vel_ && reference_robot_vel_.empty()) {
      reference_has_robot_vel_ = false;
    }
    horizon_ = std::max(robot_horizon_, object_horizon_);
    if (robot_horizon_ > 2 || object_horizon_ > 2) {
      reference_type_ = ReferenceType::kTrack;
    } else if (robot_horizon_ == 2 || object_horizon_ == 2) {
      reference_type_ = ReferenceType::kRandom;
    } else {
      reference_type_ = ReferenceType::kFixed;
    }
    if (reference_robot_init_.empty() && !reference_robot_.empty()) {
      reference_robot_init_.assign(reference_robot_.begin(),
                                   reference_robot_.begin() + robot_dim_);
    }
    if (reference_object_init_.empty() && !reference_object_.empty()) {
      reference_object_init_.assign(reference_object_.begin(),
                                    reference_object_.begin() + object_dim_);
    }
  }

  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->nu != spec_.config["action_dim"_] ||
        model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error("TrackEnv config dims do not match model.");
    }
    if (robot_dim_ <= 0 || object_dim_ != 7 || reference_time_.empty()) {
      throw std::runtime_error("TrackEnv reference metadata is incomplete.");
    }
    if (static_cast<int>(reference_robot_.size()) !=
            robot_horizon_ * robot_dim_ ||
        static_cast<int>(reference_object_.size()) !=
            object_horizon_ * object_dim_) {
      throw std::runtime_error("TrackEnv reference arrays have wrong size.");
    }
    if (reference_has_robot_vel_ &&
        static_cast<int>(reference_robot_vel_.size()) !=
            robot_horizon_ * robot_dim_) {
      throw std::runtime_error("TrackEnv robot_vel has wrong size.");
    }
    int expected_obs = model_->nq + model_->nv + robot_dim_ +
                       (reference_has_robot_vel_ ? robot_dim_ : 1) + 3 +
                       model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("TrackEnv obs_dim does not match reference.");
    }
  }

  void CacheObjects() {
    target_sid_ = mj_name2id(model_, mjOBJ_SITE, "target");
    object_bid_ = mj_name2id(model_, mjOBJ_BODY, object_name_.c_str());
    wrist_bid_ = mj_name2id(model_, mjOBJ_BODY, "lunate");
    body_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "body");
    if (target_sid_ == -1 || object_bid_ == -1 || wrist_bid_ == -1) {
      throw std::runtime_error("TrackEnv object ids missing.");
    }
    if (body_geom_id_ >= 0) {
      model_->geom_rgba[body_geom_id_ * 4 + 3] = 0.0;
    }
    lift_z_ = data_->xipos[object_bid_ * 3 + 2] + lift_bonus_thresh_;
  }

  void InitializeReferencePose() {
    if (static_cast<int>(reference_robot_init_.size()) != robot_dim_ ||
        static_cast<int>(reference_object_init_.size()) != object_dim_) {
      throw std::runtime_error("TrackEnv init reference has wrong size.");
    }
    ReferenceState reference;
    reference.robot = reference_robot_init_;
    reference.object = reference_object_init_;
    SetQposFromReference(reference);
    mju_zero(data_->qvel, model_->nv);
    mj_forward(model_, data_);
  }

  void ApplyResetState() {
    if (!test_reset_integration_state_.empty()) {
      mj_setState(model_, data_, test_reset_integration_state_.data(),
                  mjSTATE_INTEGRATION);
      mj_forward(model_, data_);
      mj_step1(model_, data_);
      return;
    }
    bool has_test_reset_override =
        !test_reset_qpos_.empty() || !test_reset_qvel_.empty() ||
        !test_reset_ctrl_.empty() || !test_reset_act_.empty() ||
        !test_reset_act_dot_.empty() || !test_reset_qacc_warmstart_.empty();
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    }
    mj_forward(model_, data_);
    if (!test_reset_ctrl_.empty()) {
      detail::RestoreVector(test_reset_ctrl_, data_->ctrl);
    }
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
    }
    if (rerun_forward) {
      mj_forward(model_, data_);
    }
    if (has_test_reset_override) {
      // Official BaseV0 reset leaves dm_control Physics ready for the next
      // legacy-step transition, including actuator-derivative fields such as
      // act_dot. TrackEnv needs the same reset-sync treatment as Challenge
      // tasks; otherwise the first mj_step2 can drift on some seeds/actions.
      mj_step1(model_, data_);
    }
  }

  const mjtNum* RobotRow(int index) const {
    return reference_robot_.data() + index * robot_dim_;
  }

  void SetQposFromReference(const ReferenceState& reference) {
    for (int i = 0; i < robot_dim_; ++i) {
      data_->qpos[i] = reference.robot[i];
    }
    for (int axis = 0; axis < 3; ++axis) {
      data_->qpos[robot_dim_ + axis] = reference.object[axis];
    }
    std::array<mjtNum, 4> quat{reference.object[3], reference.object[4],
                               reference.object[5], reference.object[6]};
    auto euler = detail::QuatToEuler(quat);
    for (int axis = 0; axis < 3; ++axis) {
      data_->qpos[robot_dim_ + 3 + axis] = euler[axis];
    }
  }

  const mjtNum* RobotVelRow(int index) const {
    return reference_has_robot_vel_
               ? reference_robot_vel_.data() + index * robot_dim_
               : nullptr;
  }

  const mjtNum* ObjectRow(int index) const {
    return reference_object_.data() + index * object_dim_;
  }

  ReferenceState PlaybackReferenceAtCurrentIndex() {
    current_reference_.robot.resize(robot_dim_);
    current_reference_.object.resize(object_dim_);
    if (reference_has_robot_vel_) {
      current_reference_.robot_vel.resize(robot_dim_);
    } else {
      current_reference_.robot_vel.assign(1, 0.0);
    }
    int index = std::clamp(playback_reference_index_, 0, horizon_ - 1);
    std::copy_n(RobotRow(std::min(index, robot_horizon_ - 1)), robot_dim_,
                current_reference_.robot.data());
    std::copy_n(ObjectRow(std::min(index, object_horizon_ - 1)), object_dim_,
                current_reference_.object.data());
    if (reference_has_robot_vel_) {
      std::copy_n(RobotVelRow(std::min(index, robot_horizon_ - 1)), robot_dim_,
                  current_reference_.robot_vel.data());
    }
    return current_reference_;
  }

  void PrimeReferenceRngForOracleResetBehavior() {
    if (reference_type_ != ReferenceType::kRandom) {
      return;
    }
    // Upstream TrackEnv construction goes through env_base.MujocoEnv._setup(),
    // which performs one zero-action step to build observation_space. For
    // random references that eagerly consumes one ref.get_reference() sample
    // before the user's first reset(). Mirror that here so the first native
    // reset observes the same sampled target under the same seed.
    (void)ReferenceAt(motion_start_time_);
  }

  ReferenceState ReferenceAt(mjtNum raw_time) {
    mjtNum time = detail::RoundReferenceTime(raw_time);
    current_reference_.robot.resize(robot_dim_);
    current_reference_.object.resize(object_dim_);
    if (reference_has_robot_vel_) {
      current_reference_.robot_vel.resize(robot_dim_);
    } else {
      current_reference_.robot_vel.assign(1, 0.0);
    }
    if (reference_type_ == ReferenceType::kFixed) {
      std::copy_n(RobotRow(0), robot_dim_, current_reference_.robot.data());
      std::copy_n(ObjectRow(0), object_dim_, current_reference_.object.data());
      if (reference_has_robot_vel_) {
        std::copy_n(RobotVelRow(0), robot_dim_,
                    current_reference_.robot_vel.data());
      }
      return current_reference_;
    }
    if (reference_type_ == ReferenceType::kRandom) {
      for (int i = 0; i < robot_dim_; ++i) {
        current_reference_.robot[i] =
            reference_rng_.UniformMjt(RobotRow(0)[i], RobotRow(1)[i]);
      }
      if (reference_has_robot_vel_) {
        for (int i = 0; i < robot_dim_; ++i) {
          current_reference_.robot_vel[i] =
              reference_rng_.UniformMjt(RobotVelRow(0)[i], RobotVelRow(1)[i]);
        }
      }
      for (int i = 0; i < object_dim_; ++i) {
        current_reference_.object[i] =
            reference_rng_.UniformMjt(ObjectRow(0)[i], ObjectRow(1)[i]);
      }
      return current_reference_;
    }
    int index = reference_index_cache_;
    if (motion_extrapolation_ && time >= reference_time_.back()) {
      index = horizon_ - 1;
    } else {
      if (time == reference_time_[reference_index_cache_]) {
        index = reference_index_cache_;
      } else if (reference_index_cache_ + 1 < horizon_ &&
                 time == reference_time_[reference_index_cache_ + 1]) {
        index = reference_index_cache_ + 1;
      } else if (reference_index_cache_ + 1 < horizon_ &&
                 time > reference_time_[reference_index_cache_] &&
                 time < reference_time_[reference_index_cache_ + 1]) {
        index = reference_index_cache_;
      } else {
        auto upper = std::upper_bound(reference_time_.begin(),
                                      reference_time_.end(), time);
        if (upper == reference_time_.begin()) {
          index = 0;
        } else {
          index = static_cast<int>(upper - reference_time_.begin() - 1);
        }
        if (index < 0) {
          index = 0;
        } else if (index >= horizon_) {
          index = horizon_ - 1;
        }
      }
    }
    reference_index_cache_ = index;
    std::copy_n(RobotRow(std::min(index, robot_horizon_ - 1)), robot_dim_,
                current_reference_.robot.data());
    std::copy_n(ObjectRow(std::min(index, object_horizon_ - 1)), object_dim_,
                current_reference_.object.data());
    if (reference_has_robot_vel_) {
      std::copy_n(RobotVelRow(std::min(index, robot_horizon_ - 1)), robot_dim_,
                  current_reference_.robot_vel.data());
    }
    return current_reference_;
  }

  void UpdateTargetSite(const ReferenceState& reference) {
    for (int axis = 0; axis < 3; ++axis) {
      model_->site_pos[target_sid_ * 3 + axis] = reference.object[axis];
    }
    mj_forward(model_, data_);
  }

  RewardInfo ComputeReward(const ReferenceState& reference) const {
    RewardInfo reward;
    mjtNum qpos_sq = 0.0;
    mjtNum qvel_sq = 0.0;
    for (int i = 0; i < robot_dim_; ++i) {
      mjtNum delta = data_->qpos[i] - reference.robot[i];
      qpos_sq += delta * delta;
      if (reference_has_robot_vel_) {
        mjtNum dv = data_->qvel[i] - reference.robot_vel[i];
        qvel_sq += dv * dv;
      }
    }
    std::array<mjtNum, 3> obj_com{};
    std::array<mjtNum, 3> wrist{};
    for (int axis = 0; axis < 3; ++axis) {
      obj_com[axis] = data_->xipos[object_bid_ * 3 + axis];
      wrist[axis] = data_->xipos[wrist_bid_ * 3 + axis];
    }
    mjtNum obj_com_sq = 0.0;
    mjtNum base_sq = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      mjtNum delta = obj_com[axis] - reference.object[axis];
      obj_com_sq += delta * delta;
      mjtNum base_delta = obj_com[axis] - wrist[axis];
      base_sq += base_delta * base_delta;
    }
    auto obj_quat = detail::Mat9ToQuat(data_->ximat + object_bid_ * 9);
    std::array<mjtNum, 4> target_quat{reference.object[3], reference.object[4],
                                      reference.object[5], reference.object[6]};
    mjtNum obj_rot_err = detail::QuatDistance(obj_quat, target_quat) /
                         static_cast<mjtNum>(3.14159265358979323846);
    mjtNum obj_reward =
        std::exp(-obj_err_scale_ * (std::sqrt(obj_com_sq) + 0.1 * obj_rot_err));
    bool lift_bonus = reference.object[2] >= lift_z_ && obj_com[2] >= lift_z_;
    mjtNum pose_reward =
        qpos_reward_weight_ * std::exp(-qpos_err_scale_ * qpos_sq);
    mjtNum qvel_reward =
        qvel_reward_weight_ * std::exp(-qvel_err_scale_ * qvel_sq);
    mjtNum base_reward = std::exp(-base_err_scale_ * std::sqrt(base_sq));
    bool obj_term = terminate_obj_fail_ &&
                    (obj_com_sq >= obj_fail_thresh_ * obj_fail_thresh_ ||
                     base_sq >= base_fail_thresh_ * base_fail_thresh_);
    bool pose_term = terminate_pose_fail_ && qpos_sq >= qpos_fail_thresh_;
    reward.pose = pose_reward + qvel_reward;
    reward.object = obj_reward + base_reward;
    reward.bonus = lift_bonus ? lift_bonus_mag_ : 0.0;
    reward.penalty = static_cast<mjtNum>(obj_term || pose_term);
    reward.done = obj_term || pose_term;
    reward.dense_reward =
        reward_pose_w_ * reward.pose + reward_object_w_ * reward.object +
        reward_bonus_w_ * reward.bonus + reward_penalty_w_ * reward.penalty;
    return reward;
  }

  void WriteState(const RewardInfo& reward, const ReferenceState& reference,
                  bool reset, mjtNum reward_value) {
    auto state = Allocate();
    state["reward"_] = reward_value;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs = state["obs"_];
      mjtNum* buffer = PrepareObservation("obs", &obs);
      for (int i = 0; i < model_->nq; ++i) {
        *(buffer++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv; ++i) {
        *(buffer++) = data_->qvel[i];
      }
      for (int i = 0; i < robot_dim_; ++i) {
        *(buffer++) = data_->qpos[i] - reference.robot[i];
      }
      if (reference_has_robot_vel_) {
        for (int i = 0; i < robot_dim_; ++i) {
          *(buffer++) = data_->qvel[i] - reference.robot_vel[i];
        }
      } else {
        *(buffer++) = 0.0;
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) =
            data_->xipos[object_bid_ * 3 + axis] - reference.object[axis];
      }
      for (int i = 0; i < model_->na; ++i) {
        *(buffer++) = data_->act[i];
      }
      CommitObservation("obs", &obs, reset);
    }
    state["info:pose"_] = reward.pose;
    state["info:object"_] = reward.object;
    state["info:bonus"_] = reward.bonus;
    state["info:penalty"_] = reward.penalty;
    state["info:success"_] = reward.success;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:target_pos"_].Assign(model_->site_pos + target_sid_ * 3, 3);
  }
};

using MyoDMTrackEnv = MyoDMTrackEnvBase<MyoDMTrackEnvSpec, false>;
using MyoDMTrackPixelEnv = MyoDMTrackEnvBase<MyoDMTrackPixelEnvSpec, true>;
using MyoDMTrackEnvPool = AsyncEnvPool<MyoDMTrackEnv>;
using MyoDMTrackPixelEnvPool = AsyncEnvPool<MyoDMTrackPixelEnv>;

}  // namespace myosuite_envpool

#endif  // ENVPOOL_MUJOCO_MYOSUITE_MYODM_H_
