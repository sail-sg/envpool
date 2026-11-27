// Copyright 2026 Garena Online Private Limited
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/py_envpool.h"
#include "envpool/xminigrid/xminigrid.h"

namespace xminigrid {

class Pool : public AsyncEnvPool<XMiniGridEnv> {
 public:
  using AsyncEnvPool::AsyncEnvPool;

  void SetRuleSet(const std::shared_ptr<RuleSet>& ruleset,
                  std::optional<int> env_id) {
    spec.Validate(*ruleset);
    if (env_id) {
      envs_.at(*env_id)->SetRuleSet(ruleset);
    } else {
      for (auto& env : envs_) {
        env->SetRuleSet(ruleset);
      }
    }
  }
};

using XMiniGridEnvSpec = PyEnvSpec<XMiniGridSpec>;
using XMiniGridEnvPool = PyEnvPool<Pool>;

PYBIND11_MODULE(xminigrid_envpool, m) {
  py::class_<RuleSet, std::shared_ptr<RuleSet>>(
      m, "RuleSet",
      "An encoded XLand goal, ordered production rules and initial tiles.")
      .def(py::init<Goal, std::vector<Rule>, std::vector<Tile>>(),
           py::arg("goal") = Goal{}, py::arg("rules") = std::vector<Rule>{},
           py::arg("init_tiles") = std::vector<Tile>{})
      .def_readonly("goal", &RuleSet::goal)
      .def_readonly("rules", &RuleSet::rules)
      .def_readonly("init_tiles", &RuleSet::init_tiles);

  REGISTER(m, XMiniGridEnvSpec, XMiniGridEnvPool)
  py::reinterpret_borrow<py::class_<XMiniGridEnvPool>>(
      m.attr("_XMiniGridEnvPool"))
      .def("set_ruleset", &Pool::SetRuleSet, py::arg("ruleset").none(false),
           py::arg("env_id") = py::none(),
           "Use this ruleset from the next reset of one slot or all slots.");

  py::list tasks;
  for (const auto& t : kTasks) {
    tasks.append(py::make_tuple(t.id, t.view_size, t.max_steps, t.num_objects));
  }
  m.attr("TASKS") = tasks;
}

}  // namespace xminigrid
