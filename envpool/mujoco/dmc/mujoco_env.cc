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

#include "envpool/mujoco/dmc/mujoco_env.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "pugixml.hpp"

namespace mujoco_dmc {

std::string XMLAssetPath(const std::string& directory,
                         const std::string& filename) {
  if (filename.empty() || directory.empty() || filename[0] == '/' ||
      (filename.size() > 1 && filename[1] == ':')) {
    return filename;
  }
  if (directory.back() == '/') {
    return directory + filename;
  }
  return directory + "/" + filename;
}

void AddReferencedAssetsToVFS(mjVFS* vfs, const std::string& base_path,
                              const std::string& raw_xml) {
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_string(raw_xml.c_str());
  if (result.status != pugi::status_ok) {
    return;
  }
  pugi::xml_node compiler = doc.child("mujoco").child("compiler");
  std::string mesh_dir = compiler.attribute("meshdir").as_string();
  std::string texture_dir = compiler.attribute("texturedir").as_string();
  std::set<std::string> asset_paths;
  for (pugi::xml_node asset = doc.child("mujoco").child("asset").first_child();
       !asset.empty(); asset = asset.next_sibling()) {
    pugi::xml_attribute file = asset.attribute("file");
    if (file.empty()) {
      continue;
    }
    std::string directory =
        std::strcmp(asset.name(), "texture") == 0 ? texture_dir : mesh_dir;
    asset_paths.insert(XMLAssetPath(directory, file.as_string()));
  }
  for (const std::string& asset_path : asset_paths) {
    std::string path = base_path;
    path.append("/mujoco/assets_dmc/");
    path.append(asset_path);
    std::ifstream ifs(path, std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    mj_addBufferVFS(vfs, asset_path.c_str(), content.data(), content.size());
  }
}

MujocoEnv::MujocoEnv(const std::string& base_path, const std::string& raw_xml,
                     int n_sub_steps, int max_episode_steps, int frame_stack,
                     int render_width, int render_height, int render_camera_id)
    : n_sub_steps_(n_sub_steps),
      max_episode_steps_(max_episode_steps),
      elapsed_step_(max_episode_steps + 1),
      frame_stack_buffer_(frame_stack),
      pixel_frame_stack_buffer_(frame_stack),
      render_width_(render_width),
      render_height_(render_height),
      render_camera_id_(render_camera_id) {
  // initialize vfs from common assets and raw xml
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/wrapper/core.py#L158
  // https://github.com/deepmind/mujoco/blob/main/python/mujoco/structs.cc
  // MjModelWrapper::LoadXML
  std::unique_ptr<mjVFS, void (*)(mjVFS*)> vfs(new mjVFS, [](mjVFS* vfs) {
    mj_deleteVFS(vfs);
    delete vfs;
  });
  mj_defaultVFS(vfs.get());
  // save raw_xml into vfs
  std::string model_filename("model_.xml");
  mj_addBufferVFS(vfs.get(), model_filename.c_str(), raw_xml.data(),
                  raw_xml.size());
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/common/__init__.py#L28
  std::vector<std::string> common_assets_name(
      {"./common/materials.xml", "./common/skybox.xml", "./common/visual.xml"});
  for (const auto& asset_name : common_assets_name) {
    std::string content = GetFileContent(base_path, asset_name);
    mj_addBufferVFS(vfs.get(), asset_name.c_str(), content.data(),
                    content.size());
  }
  AddReferencedAssetsToVFS(vfs.get(), base_path, raw_xml);
  // create model and data
  model_ = mj_loadXML(model_filename.c_str(), vfs.get(), error_.data(), 1000);
  if (model_ == nullptr) {
    throw std::runtime_error(error_.data());
  }
  data_ = mj_makeData(model_);
#ifdef ENVPOOL_TEST
  qpos0_.reset(new mjtNum[model_->nq]);
#endif
}

MujocoEnv::~MujocoEnv() {
  mj_deleteModel(model_);
  mj_deleteData(data_);
}

// rl control Environment
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py#L77
void MujocoEnv::ControlReset() {
  has_cached_render_ = false;
  elapsed_step_ = 0;
  discount_ = 1.0;
  done_ = false;
  TaskInitializeEpisodeMjcf();
  // attention: no keyframe_id
  PhysicsReset();  // first mj_forward
  TaskInitializeEpisode();
  PhysicsAfterReset();  // second mj_forward
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py#L94
void MujocoEnv::ControlStep(const mjtNum* action) {
  has_cached_render_ = false;
  TaskBeforeStep(action);
  PhysicsStep(n_sub_steps_, action);
  TaskAfterStep();
  reward_ = TaskGetReward();
  if (++elapsed_step_ >= max_episode_steps_) {
    discount_ = 1.0;
    done_ = true;
  } else if (TaskShouldTerminateEpisode()) {
    // copy from composer task
    // this is different because we don't use None in C++
    discount_ = 0.0;
    done_ = true;
  } else {
    discount_ = TaskGetDiscount();
    done_ = false;
  }
}

// Task
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/base.py#L73
void MujocoEnv::TaskBeforeStep(const mjtNum* action) {
  PhysicsSetControl(action);
}
float MujocoEnv::TaskGetReward() {
  throw std::runtime_error("GetReward not implemented");
}
float MujocoEnv::TaskGetDiscount() { return 1.0; }
bool MujocoEnv::TaskShouldTerminateEpisode() { return false; }

// Physics
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L263
void MujocoEnv::PhysicsReset(int keyframe_id) {
  if (keyframe_id < 0) {
    mj_resetData(model_, data_);
  } else {
    // actually no one steps to this line
    assert(keyframe_id < model_->nkey);
    mj_resetDataKeyframe(model_, data_, keyframe_id);
  }

  // PhysicsAfterReset may be overwritten?
  int old_flags = model_->opt.disableflags;
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  PhysicsForward();
  model_->opt.disableflags = old_flags;
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L286
void MujocoEnv::PhysicsAfterReset() {
  int old_flags = model_->opt.disableflags;
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  PhysicsForward();
  model_->opt.disableflags = old_flags;
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L138
void MujocoEnv::PhysicsSetControl(const mjtNum* control) {
  std::memcpy(data_->ctrl, control, sizeof(mjtNum) * model_->nu);
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L292
void MujocoEnv::PhysicsForward() { mj_forward(model_, data_); }

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L146
void MujocoEnv::PhysicsStep(int nstep, const mjtNum* action) {
  TaskBeforeSubStep(action);
  if (model_->opt.integrator != mjINT_RK4) {
    mj_step2(model_, data_);
  } else {
    mj_step(model_, data_);
  }
  TaskAftersSubStep();
  if (nstep > 1) {
    for (int i = 0; i < nstep - 1; ++i) {
      TaskBeforeSubStep(action);
      mj_step(model_, data_);
      TaskAftersSubStep();
    }
  }
  mj_step1(model_, data_);
}

// randomizer
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/utils/randomizers.py#L35
void MujocoEnv::RandomizeLimitedAndRotationalJoints(std::mt19937* gen) {
  for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
    int joint_type = model_->jnt_type[joint_id];
    mjtByte is_limited = model_->jnt_limited[joint_id];
    mjtNum range_min = model_->jnt_range[joint_id * 2 + 0];
    mjtNum range_max = model_->jnt_range[joint_id * 2 + 1];
    int qpos_offset = model_->jnt_qposadr[joint_id];
    if (is_limited != 0) {
      if (joint_type == mjJNT_HINGE || joint_type == mjJNT_SLIDE) {
        data_->qpos[qpos_offset] = RandUniform(range_min, range_max)(*gen);
      } else if (joint_type == mjJNT_BALL) {
        // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/utils/randomizers.py#L23
        std::array<mjtNum, 3> axis = {RandNormal(0, 1)(*gen),
                                      RandNormal(0, 1)(*gen),
                                      RandNormal(0, 1)(*gen)};
        auto norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] +
                              axis[2] * axis[2]);
        axis = {axis[0] / norm, axis[1] / norm, axis[2] / norm};
        auto angle = RandUniform(0, range_max)(*gen);
        mju_axisAngle2Quat(data_->qpos + qpos_offset, axis.data(), angle);
      }
    } else if (joint_type == mjJNT_HINGE) {
      data_->qpos[qpos_offset] = RandUniform(-M_PI, M_PI)(*gen);
    } else if (joint_type == mjJNT_BALL || joint_type == mjJNT_FREE) {
      std::array<mjtNum, 4> quat;
      if (joint_type == mjJNT_BALL) {
        quat = {RandNormal(0, 1)(*gen), RandNormal(0, 1)(*gen),
                RandNormal(0, 1)(*gen), RandNormal(0, 1)(*gen)};
      } else {
        quat = {RandUniform(0, 1)(*gen), RandUniform(0, 1)(*gen),
                RandUniform(0, 1)(*gen), RandUniform(0, 1)(*gen)};
      }
      auto norm = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                            quat[2] * quat[2] + quat[3] * quat[3]);
      int extra_offset = joint_type == mjJNT_BALL ? 0 : 3;
      data_->qpos[qpos_offset + extra_offset + 0] = quat[0] / norm;
      data_->qpos[qpos_offset + extra_offset + 1] = quat[1] / norm;
      data_->qpos[qpos_offset + extra_offset + 2] = quat[2] / norm;
      data_->qpos[qpos_offset + extra_offset + 3] = quat[3] / norm;
    }
  }
}

}  // namespace mujoco_dmc
