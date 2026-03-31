/*
 * Copyright 2021 Garena Online Private Limited
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

#ifndef ENVPOOL_CORE_ASYNC_ENVPOOL_H_
#define ENVPOOL_CORE_ASYNC_ENVPOOL_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "ThreadPool.h"
#include "envpool/core/action_buffer_queue.h"
#include "envpool/core/array.h"
#include "envpool/core/env.h"
#include "envpool/core/envpool.h"
#include "envpool/core/shared_thread_pool.h"
#include "envpool/core/spec.h"
#include "envpool/core/state_buffer_queue.h"
/**
 * Async EnvPool
 *
 * batch-action -> action buffer queue -> threadpool -> state buffer queue
 *
 * ThreadPool is tailored with EnvPool, so here we don't use the existing
 * third_party ThreadPool (which is really slow).
 */
template <typename Env>
class AsyncEnvPool : public EnvPool<typename Env::Spec> {
 protected:
  std::size_t num_envs_;
  std::size_t batch_;
  std::size_t max_num_players_;
  std::size_t num_threads_;
  bool is_sync_;
  std::atomic<int> stop_;
  std::atomic<std::size_t> stepping_env_num_;
  std::unique_ptr<StateBufferQueue> state_buffer_queue_;
  std::vector<std::unique_ptr<Env>> envs_;
  std::vector<std::atomic<int>> stepping_env_;
  std::chrono::duration<double> dur_send_, dur_recv_, dur_send_all_;
  std::shared_ptr<SharedThreadPool> shared_thread_pool_;
  // Guard pending-task accounting with the same mutex used by teardown waits
  // so the final completion notification cannot be missed.
  std::size_t pending_tasks_ = 0;
  std::mutex pending_tasks_mutex_;
  std::condition_variable pending_tasks_cv_;
  bool claimed_capacity_ = false;

  void FinishTask() {
    bool notify = false;
    {
      std::lock_guard<std::mutex> lock(pending_tasks_mutex_);
      notify = --pending_tasks_ == 0;
    }
    if (notify) {
      pending_tasks_cv_.notify_all();
    }
  }

  template <typename Fn>
  std::function<void()> WrapTask(Fn&& fn) {
    {
      std::lock_guard<std::mutex> lock(pending_tasks_mutex_);
      ++pending_tasks_;
    }
    return [this, task = std::forward<Fn>(fn)]() mutable {
      struct TaskGuard {
        AsyncEnvPool* pool;

        ~TaskGuard() { pool->FinishTask(); }
      } guard{this};
      if (this->stop_ == 1) {
        return;
      }
      task();
    };
  }

  void WaitForPendingTasks() {
    std::unique_lock<std::mutex> lock(pending_tasks_mutex_);
    pending_tasks_cv_.wait(lock, [this] { return pending_tasks_ == 0; });
  }

  template <typename V>
  void SendImpl(V&& action) {
    int* env_id = static_cast<int*>(action[0].Data());
    int shared_offset = action[0].Shape(0);
    std::vector<std::function<void()>> actions;
    actions.reserve(shared_offset);
    std::shared_ptr<std::vector<Array>> action_batch =
        std::make_shared<std::vector<Array>>(std::forward<V>(action));
    for (int i = 0; i < shared_offset; ++i) {
      int eid = env_id[i];
      envs_[eid]->SetAction(action_batch, i);
      actions.emplace_back(WrapTask([this, eid, i] {
        int order = is_sync_ ? i : -1;
        bool reset = envs_[eid]->IsDone();
        envs_[eid]->EnvStep(state_buffer_queue_.get(), order, reset, false);
      }));
    }
    if (is_sync_) {
      stepping_env_num_ += shared_offset;
    }
    auto start = std::chrono::system_clock::now();
    shared_thread_pool_->EnqueueBulk(actions);
    dur_send_ += std::chrono::system_clock::now() - start;
  }

 public:
  using Spec = typename Env::Spec;
  using Action = typename Env::Action;
  using State = typename Env::State;

  explicit AsyncEnvPool(
      const Spec& spec,
      std::shared_ptr<SharedThreadPool> shared_thread_pool = nullptr)
      : EnvPool<Spec>(spec),
        num_envs_(spec.config["num_envs"_]),
        batch_(spec.config["batch_size"_] <= 0 ? num_envs_
                                               : spec.config["batch_size"_]),
        max_num_players_(spec.config["max_num_players"_]),
        num_threads_(spec.config["num_threads"_]),
        is_sync_(batch_ == num_envs_ && max_num_players_ == 1),
        stop_(0),
        stepping_env_num_(0),
        state_buffer_queue_(new StateBufferQueue(
            batch_, num_envs_, max_num_players_,
            spec.state_spec.template AllValues<ShapeSpec>())),
        envs_(num_envs_),
        shared_thread_pool_(std::move(shared_thread_pool)) {
    std::size_t processor_count =
        std::max<std::size_t>(1, std::thread::hardware_concurrency());
    if (num_threads_ == 0) {
      num_threads_ = std::min(batch_, processor_count);
    }
    if (shared_thread_pool_ == nullptr) {
      shared_thread_pool_ = std::make_shared<SharedThreadPool>(
          num_threads_, num_envs_, spec.config["thread_affinity_offset"_]);
    }
    shared_thread_pool_->ClaimCapacity(num_envs_);
    claimed_capacity_ = true;
    try {
      ThreadPool init_pool(std::min(processor_count, num_envs_));
      std::vector<std::future<void>> result;
      for (std::size_t i = 0; i < num_envs_; ++i) {
        result.emplace_back(init_pool.enqueue(
            [i, spec, this] { envs_[i].reset(new Env(spec, i)); }));
      }
      for (auto& f : result) {
        f.get();
      }
    } catch (...) {
      shared_thread_pool_->ReleaseCapacity(num_envs_);
      claimed_capacity_ = false;
      throw;
    }
  }

  ~AsyncEnvPool() override {
    stop_ = 1;
    try {
      WaitForPendingTasks();
      if (claimed_capacity_) {
        shared_thread_pool_->ReleaseCapacity(num_envs_);
      }
    } catch (...) {
      claimed_capacity_ = false;
    }
    // LOG(INFO) << "envpool send: " << dur_send_.count();
    // LOG(INFO) << "envpool recv: " << dur_recv_.count();
  }

  void Send(const Action& action) {
    SendImpl(action.template AllValues<Array>());
  }
  void Send(const std::vector<Array>& action) override { SendImpl(action); }
  void Send(std::vector<Array>&& action) override { SendImpl(action); }

  std::vector<Array> Recv() override {
    int additional_wait = 0;
    if (is_sync_ && stepping_env_num_ < batch_) {
      additional_wait = batch_ - stepping_env_num_;
    }
    auto start = std::chrono::system_clock::now();
    auto ret = state_buffer_queue_->Wait(additional_wait);
    dur_recv_ += std::chrono::system_clock::now() - start;
    if (is_sync_) {
      stepping_env_num_ -= ret[0].Shape(0);
    }
    return ret;
  }

  Array Render(const Array& env_ids, int width, int height,
               int camera_id) override {
    WaitForPendingTasks();

    TArray<int> tenv_ids(env_ids);
    int batch = tenv_ids.Shape(0);
    if (batch <= 0) {
      throw std::invalid_argument("render env_ids must not be empty");
    }
    auto* first_renderable =
        dynamic_cast<RenderableEnv*>(envs_[tenv_ids[0]].get());
    if (first_renderable == nullptr) {
      throw std::runtime_error("render not implemented for this environment");
    }
    auto [render_width, render_height] =
        first_renderable->RenderSize(width, height);
    if (render_width <= 0 || render_height <= 0) {
      throw std::invalid_argument(
          "resolved render width and height must be positive");
    }
    ShapeSpec spec(sizeof(unsigned char),
                   {batch, render_height, render_width, 3});
    Array rendered(spec);
    auto* rgb = static_cast<unsigned char*>(rendered.Data());
    const auto frame_size = static_cast<std::size_t>(render_width) *
                            render_height * 3 * sizeof(unsigned char);
    for (int i = 0; i < batch; ++i) {
      int eid = tenv_ids[i];
      auto* renderable = dynamic_cast<RenderableEnv*>(envs_[eid].get());
      if (renderable == nullptr) {
        throw std::runtime_error("render not implemented for this environment");
      }
      auto [env_width, env_height] = renderable->RenderSize(width, height);
      if (env_width != render_width || env_height != render_height) {
        throw std::runtime_error(
            "render size must be consistent across env_ids");
      }
      renderable->Render(render_width, render_height, camera_id,
                         rgb + static_cast<std::size_t>(i) * frame_size);
    }
    return rendered;
  }

  void Reset(const Array& env_ids) override {
    TArray<int> tenv_ids(env_ids);
    int shared_offset = tenv_ids.Shape(0);
    std::vector<std::function<void()>> actions;
    actions.reserve(shared_offset);
    for (int i = 0; i < shared_offset; ++i) {
      int eid = tenv_ids[i];
      actions.emplace_back(WrapTask([this, eid, i] {
        int order = is_sync_ ? i : -1;
        envs_[eid]->EnvStep(state_buffer_queue_.get(), order, true, true);
      }));
    }
    if (is_sync_) {
      stepping_env_num_ += shared_offset;
    }
    shared_thread_pool_->EnqueueBulk(actions);
  }
};

#endif  // ENVPOOL_CORE_ASYNC_ENVPOOL_H_
