/*
 * Copyright 2023 Garena Online Private Limited
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

#ifndef ENVPOOL_CORE_SHARED_THREAD_POOL_H_
#define ENVPOOL_CORE_SHARED_THREAD_POOL_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "envpool/core/action_buffer_queue.h"

class SharedThreadPool {
 protected:
  std::unique_ptr<ActionBufferQueue> action_buffer_queue_;
  std::size_t num_threads_;
  std::size_t num_envs_capacity_;
  std::size_t claimed_num_envs_;
  std::mutex m_;
  std::vector<std::thread> workers_;
  std::atomic_bool stop_;

 public:
  explicit SharedThreadPool(std::size_t num_threads,
                            std::size_t num_envs_capacity,
                            int thread_affinity_offset)
      : num_threads_(num_threads),
        num_envs_capacity_(num_envs_capacity),
        claimed_num_envs_(0),
        stop_(false) {
    std::size_t processor_count = std::max<std::size_t>(
        1, std::thread::hardware_concurrency());
    if (num_threads_ == 0) {
      num_threads_ = std::min(processor_count, num_envs_capacity);
    }
    action_buffer_queue_ =
        std::make_unique<ActionBufferQueue>(num_envs_capacity_);
    for (std::size_t i = 0; i < num_threads_; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> fn = action_buffer_queue_->Dequeue();
          if (stop_) {
            break;
          }
          fn();
        }
      });
    }
    if (thread_affinity_offset >= 0) {
#ifdef __linux__
      for (std::size_t tid = 0; tid < num_threads_; ++tid) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        std::size_t cid = (thread_affinity_offset + tid) % processor_count;
        CPU_SET(cid, &cpuset);
        pthread_setaffinity_np(workers_[tid].native_handle(), sizeof(cpu_set_t),
                               &cpuset);
      }
#else
      (void)thread_affinity_offset;
#endif
    }
  }

  void ClaimCapacity(std::size_t num_envs) {
    std::unique_lock<std::mutex> l(m_);
    if (claimed_num_envs_ + num_envs > num_envs_capacity_) {
      throw std::runtime_error(
          "Shared thread pool capacity exceeded. Did you create more envs than "
          "the num_envs_capacity specified?");
    }
    claimed_num_envs_ += num_envs;
  }

  void ReleaseCapacity(std::size_t num_envs) {
    std::unique_lock<std::mutex> l(m_);
    if (claimed_num_envs_ < num_envs) {
      throw std::runtime_error(
          "Shared thread pool released more env capacity than it claimed.");
    }
    claimed_num_envs_ -= num_envs;
  }

  void EnqueueBulk(const std::vector<std::function<void()>>& action) {
    if (action.empty()) {
      return;
    }
    action_buffer_queue_->EnqueueBulk(action);
  }

  ~SharedThreadPool() {
    stop_ = true;
    std::vector<std::function<void()>> empty_actions(workers_.size(), [] {});
    action_buffer_queue_->EnqueueBulk(empty_actions);
    for (auto& worker : workers_) {
      worker.join();
    }
  }
};

#endif  // ENVPOOL_CORE_SHARED_THREAD_POOL_H_
