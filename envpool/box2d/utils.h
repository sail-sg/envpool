/*
 * Copyright 2022 Garena Online Private Limited
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

#ifndef ENVPOOL_BOX2D_UTILS_H_
#define ENVPOOL_BOX2D_UTILS_H_

#include <box2d/box2d.h>

// this function is to pass clang-tidy conversion check
b2Vec2 Vec2(double x, double y);

#endif  // ENVPOOL_BOX2D_UTILS_H_
