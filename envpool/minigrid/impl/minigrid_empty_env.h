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

#ifndef ENVPOOL_MINIGRID_IMPL_MINIGRID_EMPTY_ENV_H_
#define ENVPOOL_MINIGRID_IMPL_MINIGRID_EMPTY_ENV_H_

#include "envpool/minigrid/impl/minigrid_env.h"

namespace minigrid {

class MiniGridEmptyEnv : public MiniGridEnv {
 public:
  MiniGridEmptyEnv(int size, std::pair<int, int> agent_start_pos,
                   int agent_start_dir, int max_steps, int agent_view_size);
  void GenGrid() override;
};

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_MINIGRID_EMPTY_ENV_H_