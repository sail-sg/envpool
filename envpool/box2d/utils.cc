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

std::array<float, 2> RotateRad(const std::array<float, 2>& vec, float angle) {
  return {std::cos(angle) * vec[0] - std::sin(angle) * vec[1],
          std::sin(angle) * vec[0] + std::cos(angle) * vec[1]};
}

b2Vec2 RotateRad(const b2Vec2& v, float angle) {
  return {std::cos(angle) * v.x - std::sin(angle) * v.y,
          std::sin(angle) * v.x + std::cos(angle) * v.y};
}

b2Vec2 Multiply(const b2Transform& trans, const b2Vec2& v) {
  float x = (trans.q.c * v.x - trans.q.s * v.y) + trans.p.x;
  float y = (trans.q.s * v.x + trans.q.c * v.y) + trans.p.y;
  return b2Vec2(x, y);
}

}  // namespace box2d
