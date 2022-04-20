#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"  // env-specific definition of config and state/action spec
class MujocoEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.bind(200),
                    "reward_threshold"_.bind(195.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.bind(Spec<float>({4})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // the last argument in Spec is for the range definition
    return MakeDict("action"_.bind(Spec<int>({-1}, {0, 1})));
  }
};

// this line will concat common config and common state/action spec
typedef class EnvSpec<MujocoEnvFns> MujocoEnvSpec;

class MujocoEnv : public Env<MujocoEnvSpec> {
 protected:
 public:
  EnvSpec() {}
  bool IsDone() override {}
  void Reset() override {}
  void Step(const Action& action) override {}

 private:
  void WriteObs(State& state, float reward) {}
};

typedef AsyncEnvPool<MujocoEnv> MujocoPool;