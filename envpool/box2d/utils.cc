// Copyright 2022 Garena Online Private Limited
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

#include "envpool/box2d/utils.h"

namespace box2d {

b2Vec2 Vec2(double x, double y) {
  return b2Vec2(static_cast<float>(x), static_cast<float>(y));
}

float Sign(double val, double eps) {
  if (val > eps) {
    return 1;
  }
  if (val < -eps) {
    return -1;
  }
  return 0;
}

}  // namespace box2d
