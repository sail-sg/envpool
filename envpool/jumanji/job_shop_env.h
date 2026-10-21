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
#include <random>
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
constexpr int kTimeLimit = 1000;
constexpr int kOpsSize = kNumJobs * kNumOps;
constexpr int kActionMaskSize = kNumMachines * (kNoJob + 1);

}  // namespace jobshop

class JobShopEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("job_shop_ops_machine_ids"_.Bind(std::string("")),
                    "job_shop_ops_durations"_.Bind(std::string("")),
                    "job_shop_ops_mask"_.Bind(std::string("")),
                    "job_shop_machines_job_ids"_.Bind(std::string("")),
                    "job_shop_machines_remaining_times"_.Bind(std::string("")),
                    "job_shop_action_mask"_.Bind(std::string("")));
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
  bool use_configured_ops_;
  bool use_configured_action_mask_;
  std::array<int, jobshop::kOpsSize> start_times_{};
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
        use_configured_ops_(!spec.config["job_shop_ops_machine_ids"_].empty()),
        use_configured_action_mask_(
            !spec.config["job_shop_action_mask"_].empty()) {}

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
    int horizon = std::max(1, step_count_);
    for (int op = 0; op < jobshop::kOpsSize; ++op) {
      if (start_times_[op] >= 0) {
        horizon = std::max(horizon, start_times_[op] + op_durations_[op]);
      }
    }
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      const int top = machine * height / jobshop::kNumMachines;
      const int bottom = (machine + 1) * height / jobshop::kNumMachines;
      render::DrawLine(width, height, 0, top, width - 1, top, {230, 230, 230},
                       rgb);
      for (int op = 0; op < jobshop::kOpsSize; ++op) {
        if (start_times_[op] >= 0 && op_machine_ids_[op] == machine) {
          const int left = start_times_[op] * width / horizon;
          const int right =
              (start_times_[op] + op_durations_[op]) * width / horizon;
          render::FillRect(width, height, left, top + 2, right, bottom - 2,
                           render::Palette(op / jobshop::kNumOps), rgb);
        }
      }
    }
    const int now = std::min(width - 1, step_count_ * width / horizon);
    render::DrawLine(width, height, now, 0, now, height - 1, render::kBlack,
                     rgb);
  }

  void Reset() override {
    start_times_.fill(-1);
    op_machine_ids_.fill(-1);
    op_durations_.fill(-1);
    op_mask_.fill(false);
    machine_job_ids_.fill(jobshop::kNoJob);
    machine_remaining_times_.fill(0);
    if (use_configured_ops_) {
      op_machine_ids_ = configured_op_machine_ids_;
      op_durations_ = configured_op_durations_;
      op_mask_ = configured_op_mask_;
      machine_job_ids_ = configured_machine_job_ids_;
      machine_remaining_times_ = configured_machine_remaining_times_;
    } else {
      for (int job = 0; job < jobshop::kNumJobs; ++job) {
        const int operations =
            std::uniform_int_distribution<int>(1, jobshop::kNumOps)(gen_);
        for (int op = 0; op < operations; ++op) {
          const int index = job * jobshop::kNumOps + op;
          op_machine_ids_[index] = std::uniform_int_distribution<int>(
              0, jobshop::kNumMachines - 1)(gen_);
          op_durations_[index] = std::uniform_int_distribution<int>(1, 6)(gen_);
          op_mask_[index] = true;
        }
      }
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    bool valid = true;
    std::array<int, jobshop::kNumMachines> operations{};
    operations.fill(-1);
    // Validate against the pre-step mask before assigning any machines.
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      const int job = static_cast<int>(action["action"_](0, machine));
      if (job != jobshop::kNoJob) {
        const bool can_start =
            job >= 0 && job < jobshop::kNumJobs && CanStart(machine, job);
        valid = valid && can_start;
        if (can_start) {
          operations[machine] = NextOperation(job);
        }
      }
    }
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      const int job = static_cast<int>(action["action"_](0, machine));
      if (operations[machine] >= 0) {
        const int index = job * jobshop::kNumOps + operations[machine];
        machine_job_ids_[machine] = job;
        machine_remaining_times_[machine] = op_durations_[index];
        op_mask_[index] = false;
        start_times_[index] = step_count_;
      } else if (machine_remaining_times_[machine] == 0) {
        machine_job_ids_[machine] = jobshop::kNoJob;
      }
      machine_remaining_times_[machine] =
          std::max(0, machine_remaining_times_[machine] - 1);
    }
    ++step_count_;
    const bool idle =
        std::all_of(machine_job_ids_.begin(), machine_job_ids_.end(),
                    [](int job) { return job == jobshop::kNoJob; });
    done_ =
        !valid || idle || AllCompleted() || step_count_ >= jobshop::kTimeLimit;
    WriteState(!valid || idle ? -static_cast<float>(jobshop::kNumJobs *
                                                    jobshop::kNumOps * 6)
                              : -1.0f);
  }

 private:
  int NextOperation(int job) const {
    for (int op = 0; op < jobshop::kNumOps; ++op) {
      if (op_mask_[job * jobshop::kNumOps + op]) {
        return op;
      }
    }
    return -1;
  }

  bool JobBusy(int job) const {
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      if (machine_job_ids_[machine] == job &&
          machine_remaining_times_[machine] > 0) {
        return true;
      }
    }
    return false;
  }

  bool CanStart(int machine, int job) const {
    const int op = NextOperation(job);
    return machine_remaining_times_[machine] == 0 && op >= 0 && !JobBusy(job) &&
           op_machine_ids_[job * jobshop::kNumOps + op] == machine;
  }

  bool AllCompleted() const {
    return std::none_of(op_mask_.begin(), op_mask_.end(),
                        [](bool pending) { return pending; }) &&
           std::all_of(machine_remaining_times_.begin(),
                       machine_remaining_times_.end(),
                       [](int time) { return time == 0; });
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int job = 0; job < jobshop::kNumJobs; ++job) {
      for (int op = 0; op < jobshop::kNumOps; ++op) {
        const int index = job * jobshop::kNumOps + op;
        state["obs:ops_machine_ids"_](job, op) = op_machine_ids_[index];
        state["obs:ops_durations"_](job, op) = op_durations_[index];
        state["obs:ops_mask"_](job, op) = op_mask_[index];
      }
    }
    for (int machine = 0; machine < jobshop::kNumMachines; ++machine) {
      state["obs:machines_job_ids"_][machine] = machine_job_ids_[machine];
      state["obs:machines_remaining_times"_][machine] =
          machine_remaining_times_[machine];
      for (int job = 0; job <= jobshop::kNoJob; ++job) {
        if (use_configured_action_mask_ && step_count_ == 0) {
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
