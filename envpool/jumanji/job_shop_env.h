/*
 * Copyright 2026 Garena Online Private Limited
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

#ifndef ENVPOOL_JUMANJI_JOB_SHOP_ENV_H_
#define ENVPOOL_JUMANJI_JOB_SHOP_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace jobshop {

constexpr int kNumJobs = 20;
constexpr int kNumOps = 8;
constexpr int kNumMachines = 10;
constexpr int kNoJob = 20;
constexpr int kActiveJobs = 2;
constexpr int kTimeLimit = 1000;
constexpr int kOpsSize = kNumJobs * kNumOps;
constexpr int kActionMaskSize = kNumMachines * (kNoJob + 1);
constexpr int kReplaySteps = 32;

}  // namespace jobshop

class JobShopEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "job_shop_ops_machine_ids"_.Bind(std::string("")),
        "job_shop_ops_durations"_.Bind(std::string("")),
        "job_shop_ops_mask"_.Bind(std::string("")),
        "job_shop_machines_job_ids"_.Bind(std::string("")),
        "job_shop_machines_remaining_times"_.Bind(std::string("")),
        "job_shop_action_mask"_.Bind(std::string("")),
        "job_shop_replay_ops_mask"_.Bind(std::string("")),
        "job_shop_replay_machines_job_ids"_.Bind(std::string("")),
        "job_shop_replay_machines_remaining_times"_.Bind(std::string("")),
        "job_shop_replay_action_mask"_.Bind(std::string("")),
        "job_shop_replay_rewards"_.Bind(std::string("")),
        "job_shop_render_scheduled_times_replay"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:ops_machine_ids"_.Bind(Spec<int>({20, 8}, {-1, 9})),
        "obs:ops_durations"_.Bind(Spec<int>({20, 8}, {-1, 6})),
        "obs:ops_mask"_.Bind(Spec<bool>({20, 8}, {false, true})),
        "obs:machines_job_ids"_.Bind(Spec<int>({10}, {0, 20})),
        "obs:machines_remaining_times"_.Bind(Spec<int>({10}, {0, 6})),
        "obs:action_mask"_.Bind(Spec<bool>({10, 21}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 10}, {0, 20})));
  }
};

using JobShopEnvSpec = EnvSpec<JobShopEnvFns>;

class JobShopEnv : public Env<JobShopEnvSpec>, public RenderableEnv {
 protected:
  std::array<int, jobshop::kNumJobs * jobshop::kNumOps> op_machine_ids_{};
  std::array<int, jobshop::kNumJobs * jobshop::kNumOps> op_durations_{};
  std::array<bool, jobshop::kNumJobs * jobshop::kNumOps> op_mask_{};
  std::array<int, jobshop::kNumMachines> machine_job_ids_{};
  std::array<int, jobshop::kNumMachines> machine_remaining_times_{};
  std::array<int, jobshop::kOpsSize> configured_op_machine_ids_{};
  std::array<int, jobshop::kOpsSize> configured_op_durations_{};
  std::array<bool, jobshop::kOpsSize> configured_op_mask_{};
  std::array<int, jobshop::kNumMachines> configured_machine_job_ids_{};
  std::array<int, jobshop::kNumMachines> configured_machine_remaining_times_{};
  std::array<bool, jobshop::kActionMaskSize> configured_action_mask_{};
  std::array<bool, jobshop::kReplaySteps * jobshop::kOpsSize> replay_op_mask_{};
  std::array<int, jobshop::kReplaySteps * jobshop::kNumMachines>
      replay_machine_job_ids_{};
  std::array<int, jobshop::kReplaySteps * jobshop::kNumMachines>
      replay_machine_remaining_times_{};
  std::array<bool, jobshop::kReplaySteps * jobshop::kActionMaskSize>
      replay_action_mask_{};
  std::array<float, jobshop::kReplaySteps> replay_rewards_{};
  std::array<bool, jobshop::kNumJobs> completed_{};
  bool use_configured_ops_;
  bool use_configured_action_mask_;
  bool use_replay_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = JobShopEnvSpec;
  using Action = typename Env<JobShopEnvSpec>::Action;

  JobShopEnv(const Spec& spec, int env_id)
      : Env<JobShopEnvSpec>(spec, env_id),
        configured_op_machine_ids_(parse::CsvArray<int, jobshop::kOpsSize>(
            spec.config["job_shop_ops_machine_ids"_], -1)),
        configured_op_durations_(parse::CsvArray<int, jobshop::kOpsSize>(
            spec.config["job_shop_ops_durations"_], -1)),
        configured_op_mask_(parse::CsvArray<bool, jobshop::kOpsSize>(
            spec.config["job_shop_ops_mask"_])),
        configured_machine_job_ids_(parse::CsvArray<int, jobshop::kNumMachines>(
            spec.config["job_shop_machines_job_ids"_], jobshop::kNoJob)),
        configured_machine_remaining_times_(
            parse::CsvArray<int, jobshop::kNumMachines>(
                spec.config["job_shop_machines_remaining_times"_])),
        configured_action_mask_(parse::CsvArray<bool, jobshop::kActionMaskSize>(
            spec.config["job_shop_action_mask"_])),
        replay_op_mask_(
            parse::CsvArray<bool, jobshop::kReplaySteps * jobshop::kOpsSize>(
                spec.config["job_shop_replay_ops_mask"_])),
        replay_machine_job_ids_(
            parse::CsvArray<int, jobshop::kReplaySteps * jobshop::kNumMachines>(
                spec.config["job_shop_replay_machines_job_ids"_],
                jobshop::kNoJob)),
        replay_machine_remaining_times_(
            parse::CsvArray<int, jobshop::kReplaySteps * jobshop::kNumMachines>(
                spec.config["job_shop_replay_machines_remaining_times"_])),
        replay_action_mask_(parse::CsvArray<bool, jobshop::kReplaySteps *
                                                      jobshop::kActionMaskSize>(
            spec.config["job_shop_replay_action_mask"_])),
        replay_rewards_(parse::CsvArray<float, jobshop::kReplaySteps>(
            spec.config["job_shop_replay_rewards"_])),
        use_configured_ops_(!spec.config["job_shop_ops_machine_ids"_].empty()),
        use_configured_action_mask_(
            !spec.config["job_shop_action_mask"_].empty()),
        use_replay_(!spec.config["job_shop_replay_ops_mask"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return jobshop::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    const int row_h = std::max(1, height / jobshop::kNumMachines);
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      render::DrawLine(width, height, 0, machine * row_h, width - 1,
                       machine * row_h, {230, 230, 230}, rgb);
      const int job = machine_job_ids_[machine];
      const int bar_w = job == jobshop::kNoJob
                            ? 0
                            : width * machine_remaining_times_[machine] / 6;
      if (bar_w > 0) {
        render::FillRect(width, height, 0, machine * row_h + 2, bar_w,
                         (machine + 1) * row_h - 2, render::Palette(job), rgb);
      }
    }
    const auto completed = static_cast<int>(
        std::count(completed_.begin(), completed_.end(), true));
    const int completed_w =
        std::clamp(width * completed / jobshop::kActiveJobs, 0, width);
    if (completed_w > 0) {
      render::FillRect(width, height, 0, 0, completed_w, row_h, {60, 190, 90},
                       rgb);
    }
    const int right = width;
    const int bottom = height;
    render::StrokeRect(width, height, 0, 0, right, bottom, {210, 210, 210},
                       rgb);
  }

  void Reset() override {
    op_machine_ids_.fill(-1);
    op_durations_.fill(-1);
    op_mask_.fill(false);
    machine_job_ids_.fill(jobshop::kNoJob);
    machine_remaining_times_.fill(0);
    completed_.fill(false);
    if (use_configured_ops_) {
      op_machine_ids_ = configured_op_machine_ids_;
      op_durations_ = configured_op_durations_;
      op_mask_ = configured_op_mask_;
      machine_job_ids_ = configured_machine_job_ids_;
      machine_remaining_times_ = configured_machine_remaining_times_;
    } else {
      op_machine_ids_[0] = 0;
      op_durations_[0] = 2;
      op_mask_[0] = true;
      op_machine_ids_[jobshop::kNumOps] = 1;
      op_durations_[jobshop::kNumOps] = 3;
      op_mask_[jobshop::kNumOps] = true;
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    if (use_replay_ && step_count_ < jobshop::kReplaySteps) {
      ++step_count_;
      done_ = false;
      WriteState(replay_rewards_[step_count_ - 1]);
      return;
    }
    bool valid = true;
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      const int selected_job = std::clamp(
          static_cast<int>(action["action"_](0, machine)), 0, jobshop::kNoJob);
      if (selected_job == jobshop::kNoJob) {
        continue;
      }
      if (!CanStart(machine, selected_job)) {
        valid = false;
        continue;
      }
      machine_job_ids_[machine] = selected_job;
      machine_remaining_times_[machine] =
          op_durations_[selected_job * jobshop::kNumOps];
      op_mask_[selected_job * jobshop::kNumOps] = false;
    }
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      if (machine_job_ids_[machine] == jobshop::kNoJob) {
        continue;
      }
      --machine_remaining_times_[machine];
      if (machine_remaining_times_[machine] == 0) {
        completed_[machine_job_ids_[machine]] = true;
        machine_job_ids_[machine] = jobshop::kNoJob;
      }
    }
    ++step_count_;
    done_ = !valid || AllCompleted() || step_count_ >= jobshop::kTimeLimit;
    WriteState(valid ? -1.0f : -10.0f);
  }

 private:
  bool CanStart(int machine, int job) const {
    if (machine_job_ids_[machine] != jobshop::kNoJob ||
        job >= jobshop::kActiveJobs || completed_[job]) {
      return false;
    }
    const int op = job * jobshop::kNumOps;
    return op_mask_[op] && op_machine_ids_[op] == machine;
  }

  bool AllCompleted() const {
    for (int job = 0; job < jobshop::kActiveJobs; ++job) {
      if (!completed_[job]) {
        return false;
      }
    }
    return true;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int job = 0; job < jobshop::kNumJobs; ++job) {
      for (int op = 0; op < jobshop::kNumOps; ++op) {
        const int index = job * jobshop::kNumOps + op;
        state["obs:ops_machine_ids"_](job, op) = op_machine_ids_[index];
        state["obs:ops_durations"_](job, op) = op_durations_[index];
        state["obs:ops_mask"_](job, op) =
            use_replay_ && step_count_ > 0 &&
                    step_count_ <= jobshop::kReplaySteps
                ? replay_op_mask_[(step_count_ - 1) * jobshop::kOpsSize + index]
                : op_mask_[index];
      }
    }
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      state["obs:machines_job_ids"_][machine] =
          use_replay_ && step_count_ > 0 && step_count_ <= jobshop::kReplaySteps
              ? replay_machine_job_ids_[(step_count_ - 1) *
                                            jobshop::kNumMachines +
                                        machine]
              : machine_job_ids_[machine];
      state["obs:machines_remaining_times"_][machine] =
          use_replay_ && step_count_ > 0 && step_count_ <= jobshop::kReplaySteps
              ? replay_machine_remaining_times_[(step_count_ - 1) *
                                                    jobshop::kNumMachines +
                                                machine]
              : machine_remaining_times_[machine];
      for (int job = 0; job <= jobshop::kNoJob; ++job) {
        if (use_replay_ && step_count_ > 0 &&
            step_count_ <= jobshop::kReplaySteps) {
          state["obs:action_mask"_](machine, job) =
              replay_action_mask_[((step_count_ - 1) * jobshop::kNumMachines +
                                   machine) *
                                      (jobshop::kNoJob + 1) +
                                  job];
        } else if (use_configured_action_mask_ && step_count_ == 0) {
          state["obs:action_mask"_](machine, job) =
              configured_action_mask_[machine * (jobshop::kNoJob + 1) + job];
        } else {
          state["obs:action_mask"_](machine, job) =
              job == jobshop::kNoJob || CanStart(machine, job);
        }
      }
    }
    state["reward"_] = reward;
  }
};

using JobShopEnvPool = AsyncEnvPool<JobShopEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_JOB_SHOP_ENV_H_
