#ifndef ENVPOOL_CORE_STATE_BUFFER_H_
#define ENVPOOL_CORE_STATE_BUFFER_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/dict.h"
#include "envpool/core/spec.h"

/**
 * Buffer of a batch of states, which is used as an intermediate storage device
 * for the environments to write their state outputs of each step.
 * There's a quota for how many envs' results are stored in this buffer,
 * which is controlled by the batch argments in the constructor.
 */
class StateBuffer {
 protected:
  std::size_t batch_;
  std::size_t max_num_players_;
  std::vector<ShapeSpec> specs_;
  std::vector<bool> is_player_state_;
  std::vector<Array> arrays_;
  std::atomic<uint64_t> offsets_{0};
  std::atomic<std::size_t> alloc_count_{0};
  std::atomic<std::size_t> done_count_{0};
  bool done_ = false;
  std::mutex mu_;
  std::condition_variable cv_;

 public:
  /**
   * Return type of StateBuffer.Allocate is a slice of each state arrays that
   * can be written by the caller. When writing is done, the caller should
   * invoke done write.
   */
  struct WritableSlice {
    std::vector<Array> arr;
    std::function<void()> done_write;
  };

  /**
   * Create a StateBuffer instance with the player_specs and shared_specs
   * provided.
   */
  StateBuffer(std::size_t batch, std::size_t max_num_players,
              const std::vector<ShapeSpec>& specs)
      : batch_(batch),
        max_num_players_(max_num_players),
        specs_(Transform(specs,
                         [=](ShapeSpec s) {
                           if (s.shape.size() > 0 && s.shape[0] == -1) {
                             // If first dim is num_players
                             s.shape[0] = batch * max_num_players;
                             return s;
                           } else {
                             return s.Batch(batch);
                           }
                         })),
        is_player_state_(Transform(specs,
                                   [](const ShapeSpec& s) {
                                     return (s.shape.size() > 0 &&
                                             s.shape[0] == -1);
                                   })),
        arrays_(MakeArray(specs_)) {}

  /**
   * Tries to allocate a piece of memory without lock.
   * If this buffer runs out of quota, a out_of_range exception is thrown.
   * Externally, caller has to catch the exception and handle accordingly.
   */
  WritableSlice Allocate(std::size_t num_players) {
    DCHECK_LE(num_players, max_num_players_);
    std::size_t alloc_count = alloc_count_.fetch_add(1);
    if (alloc_count < batch_) {
      // Make a increment atomically on two uint32_t simultaneously
      // This avoids lock
      uint64_t increment = ((uint64_t)num_players) << 32 | (uint32_t)1;
      uint64_t offsets = offsets_.fetch_add(increment);
      uint32_t player_offset = (uint32_t)(offsets >> 32);
      uint32_t shared_offset = (uint32_t)offsets;
      DCHECK_LE((std::size_t)shared_offset + 1, batch_);
      DCHECK_LE((std::size_t)(player_offset + num_players),
                batch_ * max_num_players_);
      std::vector<Array> state(arrays_.size());
      for (std::size_t i = 0; i < arrays_.size(); ++i) {
        const Array& a = arrays_[i];
        if (is_player_state_[i]) {
          state[i] = a.Slice(player_offset, player_offset + num_players);
        } else {
          state[i] = a[shared_offset];
        }
      }
      return WritableSlice{.arr = std::move(state),
                           .done_write = [this]() { Done(); }};
    } else {
      DLOG(INFO) << "Allocation failed, continue to the next block of memory";
      throw std::out_of_range("StateBuffer out of storage");
    }
  }

  std::pair<uint32_t, uint32_t> Offsets() const {
    uint32_t player_offset = (uint32_t)(offsets_ >> 32);
    uint32_t shared_offset = (uint32_t)offsets_;
    return {player_offset, shared_offset};
  }

  /**
   * When the allocated memory has been filled, the user of the memory will call
   * this callback to notify StateBuffer that its part has been written.
   */
  void Done() {
    std::size_t done_count = done_count_.fetch_add(1);
    if (done_count == batch_ - 1) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        done_ = true;
      }
      cv_.notify_all();
    }
  }

  /**
   * Blocks until the entire buffer is ready, aka, all quota has been
   * distributed out, and all user has called done.
   */
  std::vector<Array> Wait() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this]() { return done_; });
    // when things are all done, compact the buffer.
    uint32_t player_offset = (uint32_t)(offsets_ >> 32);
    uint32_t shared_offset = (uint32_t)offsets_;
    DCHECK_EQ((std::size_t)shared_offset, batch_)
        << "When this StateBuffer is ready, the shared state arrays should be "
           "used up.";
    std::vector<Array> ret(arrays_.size());
    for (std::size_t i = 0; i < arrays_.size(); ++i) {
      const Array& a = arrays_[i];
      if (is_player_state_[i]) {
        ret[i] = a.Truncate(player_offset);
      } else {
        ret[i] = a;
      }
    }
    return ret;
  }
};

#endif  // ENVPOOL_CORE_STATE_BUFFER_H_
