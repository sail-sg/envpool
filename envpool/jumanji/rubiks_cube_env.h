/*
 * Copyright 2026 Garena Online Private Limited
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

#ifndef ENVPOOL_JUMANJI_RUBIKS_CUBE_ENV_H_
#define ENVPOOL_JUMANJI_RUBIKS_CUBE_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace rubiks_cube {

constexpr int kFaces = 6;
constexpr int kSize = 3;
constexpr int kStickerCount = kFaces * kSize * kSize;

using Cube = std::array<std::int8_t, kStickerCount>;

inline int Offset(int face, int row, int col) {
  return face * kSize * kSize + row * kSize + col;
}

inline Cube SolvedCube() {
  Cube cube{};
  for (int face = 0; face < kFaces; ++face) {
    for (int row = 0; row < kSize; ++row) {
      for (int col = 0; col < kSize; ++col) {
        cube[Offset(face, row, col)] = static_cast<std::int8_t>(face);
      }
    }
  }
  return cube;
}

inline Cube ParseCube(const std::string& text) {
  Cube cube = SolvedCube();
  if (text.empty()) {
    return cube;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kStickerCount) {
    cube[index++] = static_cast<std::int8_t>(std::stoi(token));
  }
  return cube;
}

inline bool IsSolved(const Cube& cube) {
  for (int face = 0; face < kFaces; ++face) {
    const auto value = cube[Offset(face, 0, 0)];
    for (int row = 0; row < kSize; ++row) {
      for (int col = 0; col < kSize; ++col) {
        if (cube[Offset(face, row, col)] != value) {
          return false;
        }
      }
    }
  }
  return true;
}

inline void RotateFaceClockwise(Cube* cube, int face) {
  Cube before = *cube;
  for (int row = 0; row < kSize; ++row) {
    for (int col = 0; col < kSize; ++col) {
      (*cube)[Offset(face, row, col)] =
          before[Offset(face, kSize - 1 - col, row)];
    }
  }
}

inline void AdjacentIndices(int face, std::array<int, 12>* faces,
                            std::array<int, 12>* rows,
                            std::array<int, 12>* cols) {
  std::array<int, 4> adjacent{};
  std::array<int, 12> r{};
  std::array<int, 12> c{};
  if (face == 0) {
    adjacent = {1, 4, 3, 2};
    r = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    c = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  } else if (face == 1) {
    adjacent = {0, 2, 5, 4};
    r = {2, 2, 2, 0, 1, 2, 0, 0, 0, 2, 1, 0};
    c = {0, 1, 2, 0, 0, 0, 2, 1, 0, 2, 2, 2};
  } else if (face == 2) {
    adjacent = {0, 3, 5, 1};
    r = {2, 1, 0, 0, 1, 2, 2, 1, 0, 2, 1, 0};
    c = {2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2};
  } else if (face == 3) {
    adjacent = {0, 4, 5, 2};
    r = {0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 0};
    c = {2, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2};
  } else if (face == 4) {
    adjacent = {0, 1, 5, 3};
    r = {0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0};
    c = {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2};
  } else {
    adjacent = {1, 2, 3, 4};
    r = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    c = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  }
  for (int group = 0; group < 4; ++group) {
    for (int i = 0; i < kSize; ++i) {
      const int index = group * kSize + i;
      (*faces)[index] = adjacent[group];
      (*rows)[index] = r[index];
      (*cols)[index] = c[index];
    }
  }
}

inline int AmountTurns(int amount_index) {
  if (amount_index == 0) {
    return 1;
  }
  if (amount_index == 1) {
    return -1;
  }
  return 2;
}

inline void Rotate(Cube* cube, int face, int amount_index) {
  const int amount = AmountTurns(amount_index);
  const int face_turns = (amount % 4 + 4) % 4;
  for (int i = 0; i < face_turns; ++i) {
    RotateFaceClockwise(cube, face);
  }
  std::array<int, 12> faces{};
  std::array<int, 12> rows{};
  std::array<int, 12> cols{};
  AdjacentIndices(face, &faces, &rows, &cols);
  std::array<std::int8_t, 12> values{};
  for (int i = 0; i < 12; ++i) {
    values[i] = (*cube)[Offset(faces[i], rows[i], cols[i])];
  }
  const int shift = ((kSize * amount) % 12 + 12) % 12;
  for (int i = 0; i < 12; ++i) {
    (*cube)[Offset(faces[i], rows[i], cols[i])] = values[(i - shift + 12) % 12];
  }
}

}  // namespace rubiks_cube

template <int TimeLimit, int DefaultScrambles>
class RubiksCubeEnvFnsBase {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("rubiks_cube_num_scrambles"_.Bind(DefaultScrambles),
                    "rubiks_cube_initial_cube"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs:cube"_.Bind(Spec<std::int8_t>({6, 3, 3}, {0, 5})),
                    "obs:step_count"_.Bind(Spec<int>({}, {0, TimeLimit})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 3}, {{0, 0, 0}, {5, 0, 2}})));
  }
};

using RubiksCubeEnvFns = RubiksCubeEnvFnsBase<200, 100>;
using RubiksCubePartlyScrambledEnvFns = RubiksCubeEnvFnsBase<20, 20>;
using RubiksCubeEnvSpec = EnvSpec<RubiksCubeEnvFns>;
using RubiksCubePartlyScrambledEnvSpec =
    EnvSpec<RubiksCubePartlyScrambledEnvFns>;  // NOLINT

template <typename SpecT, int TimeLimit>
class RubiksCubeEnvBase : public Env<SpecT>, public RenderableEnv {
 protected:
  rubiks_cube::Cube cube_{};
  rubiks_cube::Cube configured_cube_{};
  bool use_configured_cube_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = SpecT;
  using Action = typename Env<SpecT>::Action;

  RubiksCubeEnvBase(const Spec& spec, int env_id)
      : Env<SpecT>(spec, env_id),
        configured_cube_(
            rubiks_cube::ParseCube(spec.config["rubiks_cube_initial_cube"_])),
        use_configured_cube_(
            !spec.config["rubiks_cube_initial_cube"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return TimeLimit + 1; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    constexpr std::array<render::Color, 6> colors = {{
        {255, 255, 255},
        {255, 214, 0},
        {0, 82, 255},
        {0, 155, 72},
        {255, 88, 0},
        {183, 18, 52},
    }};
    constexpr std::array<std::array<int, 2>, 6> face_pos = {{
        {{1, 0}},
        {{0, 1}},
        {{1, 1}},
        {{2, 1}},
        {{3, 1}},
        {{1, 2}},
    }};
    const int face_side = std::max(1, std::min(width / 4, height / 3) - 4);
    const int x_origin = (width - 4 * face_side) / 2;
    const int y_origin = (height - 3 * face_side) / 2;
    for (int face = 0; face < 6; ++face) {
      const int face_left = x_origin + face_pos[face][0] * face_side;
      const int face_top = y_origin + face_pos[face][1] * face_side;
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          const int value = cube_[rubiks_cube::Offset(face, row, col)];
          const int left = face_left + col * face_side / 3;
          const int right = face_left + (col + 1) * face_side / 3;
          const int top = face_top + row * face_side / 3;
          const int bottom = face_top + (row + 1) * face_side / 3;
          render::FillRect(width, height, left + 1, top + 1, right - 1,
                           bottom - 1, colors[value], rgb);
          render::StrokeRect(width, height, left, top, right, bottom,
                             render::kBlack, rgb);
        }
      }
    }
  }

  void Reset() override {
    cube_ = use_configured_cube_ ? configured_cube_ : rubiks_cube::SolvedCube();
    step_count_ = 0;
    done_ = false;
    std::uniform_int_distribution<int> face_dist(0, 5);
    std::uniform_int_distribution<int> amount_dist(0, 2);
    const int scrambles = this->spec_.config["rubiks_cube_num_scrambles"_];
    if (!use_configured_cube_) {
      for (int i = 0; i < scrambles; ++i) {
        rubiks_cube::Rotate(&cube_, face_dist(this->gen_),
                            amount_dist(this->gen_));
      }
    }
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int face =
        std::clamp(static_cast<int>(action["action"_](0, 0)), 0, 5);
    const int amount =
        std::clamp(static_cast<int>(action["action"_](0, 2)), 0, 2);
    rubiks_cube::Rotate(&cube_, face, amount);
    ++step_count_;
    const bool solved = rubiks_cube::IsSolved(cube_);
    done_ = solved || step_count_ >= TimeLimit;
    WriteState(solved ? 1.0f : 0.0f);
  }

 private:
  void WriteState(float reward) {
    auto state = this->Allocate();
    for (int face = 0; face < 6; ++face) {
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          state["obs:cube"_](face, row, col) =
              cube_[rubiks_cube::Offset(face, row, col)];
        }
      }
    }
    state["obs:step_count"_] = step_count_;
    state["reward"_] = reward;
  }
};

using RubiksCubeEnv = RubiksCubeEnvBase<RubiksCubeEnvSpec, 200>;
using RubiksCubePartlyScrambledEnv =
    RubiksCubeEnvBase<RubiksCubePartlyScrambledEnvSpec, 20>;  // NOLINT
using RubiksCubeEnvPool = AsyncEnvPool<RubiksCubeEnv>;
using RubiksCubePartlyScrambledEnvPool =
    AsyncEnvPool<RubiksCubePartlyScrambledEnv>;  // NOLINT

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_RUBIKS_CUBE_ENV_H_
