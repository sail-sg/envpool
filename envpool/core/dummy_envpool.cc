#include <string>
#include <vector>

#include "envpool/core/dict.h"
#include "envpool/core/envpool.h"
#include "envpool/core/py_envpool.h"
#include "envpool/core/spec.h"

namespace dummy {

class DummyEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("abc"_.bind(1), "xyz"_.bind(std::string("dddd")),
                    "123"_.bind(10.));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("players.abc"_.bind(Spec<int>({2})),
                    "players.cde"_.bind(Spec<float>({1, 3})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("players.do1"_.bind(Spec<int>({1, 2})),
                    "players.do2"_.bind(Spec<float>({2, 3})));
  }
};

typedef class EnvSpec<DummyEnvFns> DummyEnvSpec;

class DummyEnvPool : public EnvPool<DummyEnvSpec> {
 public:
  explicit DummyEnvPool(const DummyEnvSpec& spec)
      : EnvPool<DummyEnvSpec>(spec) {}

 protected:
  void Send(const std::vector<Array>& action) override {
    LOG(INFO) << static_cast<int>(action[0](0, 0, 0));
    LOG(INFO) << static_cast<int>(action[0](0, 0, 1));
    LOG(INFO) << static_cast<int>(action[0](0, 1, 1));
    LOG(INFO) << static_cast<int>(action[0](0, 1, 2));
    LOG(INFO) << static_cast<int>(action[0](0, 1));
  }
  std::vector<Array> Recv() override {
    auto arr = MakeArray(spec_.state_spec.values());
    arr[0][0] = 2;
    arr[0][1] = 10;
    return arr;
  }
};

}  // namespace dummy

typedef PyEnvSpec<dummy::DummyEnvSpec> DummyEnvSpec;
typedef PyEnvPool<dummy::DummyEnvPool> DummyEnvPool;

PYBIND11_MODULE(dummy_envpool, m) { REGISTER(m, DummyEnvSpec, DummyEnvPool) }
