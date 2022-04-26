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

#include "envpool/dummy/dummy_envpool.h"

#include "envpool/core/py_envpool.h"

/**
 * Wrap the `DummyEnvSpec` and `DummyEnvPool` with the corresponding `PyEnvSpec`
 * and `PyEnvPool` template.
 */
using DummyEnvSpec = PyEnvSpec<dummy::DummyEnvSpec>;
using DummyEnvPool = PyEnvPool<dummy::DummyEnvPool>;

/**
 * Finally, call the REGISTER macro to expose them to python
 */
PYBIND11_MODULE(dummy_envpool, m) { REGISTER(m, DummyEnvSpec, DummyEnvPool) }
