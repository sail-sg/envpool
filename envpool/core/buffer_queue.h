#ifndef ENVPOOL_CORE_BUFFER_QUEUE_H_
#define ENVPOOL_CORE_BUFFER_QUEUE_H_

#include <list>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/spec.h"
#include "envpool/core/state_buffer.h"

class BufferQueue {
 protected:
  std::size_t batch_;
  std::size_t max_num_players_;
  std::vector<ShapeSpec> specs_;
  std::size_t queue_size_;
  std::vector<std::unique_ptr<StateBuffer>> queue_;
  std::atomic<std::size_t> head_;
  std::atomic<std::size_t> tail_;
  std::atomic<uint64_t> alloc_count_;

 public:
  BufferQueue(std::size_t batch_env, std::size_t num_envs,
              std::size_t max_num_players, const std::vector<ShapeSpec>& specs)
      : batch_(batch_env),
        max_num_players_(max_num_players),
        specs_(specs),
        queue_size_(
            (num_envs / batch_env + (std::size_t)2) *
            (std::size_t)2),  // two times enough buffer for all the envs
        queue_(queue_size_),  // circular buffer
        head_(0),
        tail_(0),
        alloc_count_(0) {
    // Only initialize first half of the buffer
    // At the consumption of each block, the first consumping thread
    // will allocate a new state buffer and append to the tail.
    tail_ = num_envs / batch_env + 2;
    for (std::size_t i = 0; i < tail_; ++i) {
      queue_[i].reset(new StateBuffer(batch_, max_num_players_, specs_));
      DLOG(INFO) << "Allocate at " << i;
    }
  }

  /**
   * Allocate slice of memory for the current env to write.
   * This function is used from the producer side.
   * It is safe to access from multiple threads.
   */
  StateBuffer::WritableSlice Allocate(std::size_t num_players) {
    std::size_t pos = alloc_count_.fetch_add(1);
    std::size_t offset = (pos / batch_) % queue_size_;
    if (pos % batch_ == 0) {
      // At the time a new statebuffer is accessed, the first visitor allocate
      // a new state buffer and put it at the back of the queue.
      std::size_t insert_pos = tail_.fetch_add(1);
      std::size_t insert_offset = insert_pos % queue_size_;
      queue_[insert_offset].reset(
          new StateBuffer(batch_, max_num_players_, specs_));
      DLOG(INFO) << "Allocate at " << insert_offset;
    }
    return queue_[offset]->Allocate(num_players);
  }

  /**
   * Wait for the state buffer at the head to be ready.
   * This function can only be accessed from one thread.
   *
   * BIG CAVEATE:
   * Wait should be accessed from only one thread.
   * If Wait is accessed from multiple threads, it is only safe if the finish
   * time of each state buffer is in the same order as the allocation time.
   */
  std::vector<Array> Wait() {
    std::size_t pos = head_.fetch_add(std::size_t(1));
    std::size_t offset = pos % queue_size_;
    DLOG(INFO) << "Wait at " << offset;
    auto arr = queue_[offset]->Wait();
    queue_[offset].reset(nullptr);
    DLOG(INFO) << "Reset at " << offset;
    return arr;
  }
};

#endif  // ENVPOOL_CORE_BUFFER_QUEUE_H_
