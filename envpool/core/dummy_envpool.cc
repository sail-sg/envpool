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
    return MakeDict("obs:players.abc"_.bind(Spec<int>({2})),
                    "info:players.cde"_.bind(Spec<float>({1, 3})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("players.do1"_.bind(Spec<int>({1, 2, 4})),
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
    Array do1 =
        Action(const_cast<std::vector<Array>*>(&action))["players.do1"_];
    LOG(INFO) << static_cast<int>(do1(0, 0, 0));
    LOG(INFO) << static_cast<int>(do1(0, 0, 1));
    LOG(INFO) << static_cast<int>(do1(0, 1, 1));
    LOG(INFO) << static_cast<int>(do1(0, 1, 2));
    LOG(INFO) << static_cast<int>(do1(0, 0));
    LOG(INFO) << static_cast<int>(do1(0, 1));
    LOG(INFO) << static_cast<int>(do1(1, 0));
    LOG(INFO) << static_cast<int>(do1(1, 1));
  }
  std::vector<Array> Recv() override {
    auto arr = MakeArray(
        Transform(spec_.state_spec.values<ShapeSpec>(), [=](ShapeSpec s) {
          if (s.shape.size() > 0 && s.shape[0] == -1) {
            s.shape[0] = 1;
          }
          return s;
        }));
    Array abc = State(&arr)["obs:players.abc"_];
    abc[0] = 2;
    abc[1] = 10;
    return arr;
  }
};

}  // namespace dummy

typedef PyEnvSpec<dummy::DummyEnvSpec> DummyEnvSpec;
typedef PyEnvPool<dummy::DummyEnvPool> DummyEnvPool;

PYBIND11_MODULE(dummy_envpool, m) { REGISTER(m, DummyEnvSpec, DummyEnvPool) }
