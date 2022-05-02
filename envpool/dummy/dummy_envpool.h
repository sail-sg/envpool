// Copyright 2021 Garena Online Private Limited
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

#ifndef ENVPOOL_DUMMY_DUMMY_ENVPOOL_H_
#define ENVPOOL_DUMMY_DUMMY_ENVPOOL_H_

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace dummy {

class DummyEnvFns {
 public:
  /**
   * Returns a dict, keys are the names of the configurable variables of this
   * env, values stores the default values of the corresponding variable.
   *
   * EnvPool will append to your configuration some common fields, currently
   * there are the envpool specific configurations that defines the behavior of
   * envpool.
   *
   * 1. num_envs: number of envs to be launched in envpool
   * 2. batch_size: the batch_size when interacting with the envpool
   * 3. num_threads: the number of threads to run all the envs
   * 4. thread_affinity_offset: sets the thread affinity of the threads
   * 5. base_path: contains the path of the envpool python package
   * 6. seed: random seed
   *
   * These's also single env specific configurations
   *
   * 7. max_num_players: defines the number of players in a single env.
   *
   */
  static decltype(auto) DefaultConfig() {
    return MakeDict("state_num"_.Bind(10), "action_num"_.Bind(6));
  }

  /**
   * Returns a dict, keys are the names of the states of this env,
   * values are the ArraySpec of the state (as each state is stored in an
   * array).
   *
   * The array spec can be created by calling `Spec<dtype>(shape, bounds)`.
   *
   * Similarly, envpool also append to this state spec, there're:
   *
   * 1. info:env_id: a int array that has shape [batch_size], when there's a
   * batch of states, it tells the user from which `env_id` that these states
   * come from.
   * 2. info:players.env_id: This is similar to `env_id`, but it has a shape of
   * [total_num_player], where the `total_num_player` is the total number of
   * players summed.
   *
   * For example, if in one batch we have states from envs [1, 3, 4],
   * in env 1 there're players [1, 2], in env 2 there're players [2, 3, 4],
   * in env 3 there're players [1]. Then:
   * `info:env_id == [1, 3, 4]`
   * `info:players.env_id == [1, 1, 3, 3, 3, 4]`
   *
   * 3. elapsed_step: the total elapsed steps of the envs.
   * 4. done: whether it is the end of episode for each env.
   */
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<int>({-1, conf["state_num"_]})),
                    "dyn"_.Bind(Spec<Container<int>>(
                        {-1}, Spec<int>({-1, conf["state_num"_]}))),
                    "info:players.done"_.Bind(Spec<bool>({-1})),
                    "info:players.id"_.Bind(
                        Spec<int>({-1}, {0, conf["max_num_players"_]})));
  }

  /**
   * Returns a dict, keys are the names of the actions of this env,
   * values are the ArraySpec of the actions (each action is stored in an
   * array).
   *
   * Similarly, envpool also append to this state spec, there're:
   *
   * 1. env_id
   * 2. players.env_id
   *
   * Their meanings are the same as in the `StateSpec`.
   */
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("players.action"_.Bind(Spec<int>({-1})),
                    "players.id"_.Bind(Spec<int>({-1})));
  }
};

/**
 * Create an DummyEnvSpec by passing the above functions to EnvSpec.
 */
using DummyEnvSpec = EnvSpec<DummyEnvFns>;

/**
 * The main part of the single env.
 * It inherits and implements the interfaces defined in Env specialized by the
 * DummyEnvSpec we defined above.
 */
class DummyEnv : public Env<DummyEnvSpec> {
 protected:
  int state_;

 public:
  /**
   * Initilize the env, in this function we perform tasks like loading the game
   * rom etc.
   */
  DummyEnv(const Spec& spec, int env_id)
      : Env<DummyEnvSpec>(spec, env_id), state_(0) {
    if (seed_ < 1) {
      seed_ = 1;
    }
  }

  /**
   * Reset this single env, this has the same meaning as the openai gym's reset
   * The reset function usually returns the state after reset, here, we first
   * call `Allocate` to create the state (which is managed by envpool), and
   * populate it with the returning state.
   */
  void Reset() override {
    state_ = 0;
    int num_players =
        max_num_players_ <= 1 ? 1 : state_ % (max_num_players_ - 1) + 1;

    // Ask envpool to allocate a piece of memory where we can write the state
    // after reset.
    auto state = Allocate(num_players);

    // write the information of the next state into the state.
    for (int i = 0; i < num_players; ++i) {
      state["info:players.id"_][i] = i;
      state["info:players.done"_][i] = IsDone();
      state["obs"_](i, 0) = state_;
      state["obs"_](i, 1) = 0;
      state["reward"_][i] = -i;
    }
  }

  /**
   * Step is the central function of a single env.
   * It takes an action, executes the env, and returns the next state.
   *
   * Similar to Reset, Step also return the state through `Allocate` function.
   *
   */
  void Step(const Action& action) override {
    ++state_;
    int num_players =
        max_num_players_ <= 1 ? 1 : state_ % (max_num_players_ - 1) + 1;

    // Ask envpool to allocate a piece of memory where we can write the state
    // after reset.
    auto state = Allocate(num_players);

    // Parse the action, and execute the env (dummy env has nothing to do)
    int action_num = action["players.env_id"_].Shape(0);
    for (int i = 0; i < action_num; ++i) {
      if (static_cast<int>(action["players.env_id"_][i]) != env_id_) {
        action_num = 0;
      }
    }

    // write the information of the next state into the state.
    for (int i = 0; i < num_players; ++i) {
      state["info:players.id"_][i] = i;
      state["info:players.done"_][i] = IsDone();
      state["obs"_](i, 0) = state_;
      state["obs"_](i, 1) = action_num;
      state["reward"_][i] = -i;
    }
  }

  /**
   * Whether the single env has ended the current episode.
   */
  bool IsDone() override { return state_ >= seed_; }
};

/**
 * Pass the DummyEnv we defined above as an template parameter to the
 * AsyncEnvPool template, it gives us a parallelized version of the single env.
 */
using DummyEnvPool = AsyncEnvPool<DummyEnv>;

}  // namespace dummy

#endif  // ENVPOOL_DUMMY_DUMMY_ENVPOOL_H_
