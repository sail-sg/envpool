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

#ifndef ENVPOOL_PGX_SHOGI_H_
#define ENVPOOL_PGX_SHOGI_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/pgx/board_games.h"

namespace pgx {
namespace shogi {

constexpr int kBoardSize = 9;
constexpr int kSquares = 81;
constexpr int kActions = 27 * 81;
constexpr int kMaxTerminationSteps = 512;
constexpr int kEmpty = -1;
constexpr int kPawn = 0;
constexpr int kLance = 1;
constexpr int kKnight = 2;
constexpr int kSilver = 3;
constexpr int kBishop = 4;
constexpr int kRook = 5;
constexpr int kGold = 6;
constexpr int kKing = 7;
constexpr int kProPawn = 8;
constexpr int kProLance = 9;
constexpr int kProKnight = 10;
constexpr int kProSilver = 11;
constexpr int kHorse = 12;
constexpr int kDragon = 13;
constexpr int kOppPawn = 14;
constexpr int kOppKing = 21;

using Board = std::array<int, kSquares>;
using Hand = std::array<std::array<int, 7>, 2>;
using Mask = std::array<bool, kActions>;

constexpr Board kInitBoard = {
    15, -1, 14, -1, -1, -1, 0,  -1, 1,  16, 18, 14, -1, -1, -1, 0,  5,
    2,  17, -1, 14, -1, -1, -1, 0,  -1, 3,  20, -1, 14, -1, -1, -1, 0,
    -1, 6,  21, -1, 14, -1, -1, -1, 0,  -1, 7,  20, -1, 14, -1, -1, -1,
    0,  -1, 6,  17, -1, 14, -1, -1, -1, 0,  -1, 3,  16, 19, 14, -1, -1,
    -1, 0,  4,  2,  15, -1, 14, -1, -1, -1, 0,  -1, 1,
};

struct Action {
  bool is_drop{false};
  int piece{kEmpty};
  int to{0};
  int from{0};
  bool is_promotion{false};
};

inline bool IsOwn(int piece) { return piece >= kPawn && piece < kOppPawn; }

inline bool IsOpponent(int piece) { return piece >= kOppPawn; }

inline int FlipPiece(int piece) {
  return piece >= 0 ? (piece + 14) % 28 : piece;
}

inline Board FlipBoard(const Board& board) {
  Board out{};
  for (int i = 0; i < kSquares; ++i) {
    out[i] = FlipPiece(board[kSquares - 1 - i]);
  }
  return out;
}

inline Hand FlipHand(const Hand& hand) { return {hand[1], hand[0]}; }

inline int X(int pos) { return pos / kBoardSize; }

inline int Y(int pos) { return pos % kBoardSize; }

inline int Sign(int value) {
  if (value < 0) {
    return -1;
  }
  if (value > 0) {
    return 1;
  }
  return 0;
}

inline bool CanMoveTo(int piece, int from, int to) {
  if (from == to || from < 0 || to < 0 || from >= kSquares || to >= kSquares) {
    return false;
  }
  const int dx = X(to) - X(from);
  const int dy = Y(to) - Y(from);
  switch (piece) {
    case kPawn:
      return dx == 0 && dy == -1;
    case kLance:
      return dx == 0 && dy < 0;
    case kKnight:
      return (dx == -1 || dx == 1) && dy == -2;
    case kSilver:
      return ((dx >= -1 && dx <= 1) && dy == -1) ||
             ((dx == -1 || dx == 1) && dy == 1);
    case kBishop:
      return dx == dy || dx == -dy;
    case kRook:
      return dx == 0 || dy == 0;
    case kGold:
    case kProPawn:
    case kProLance:
    case kProKnight:
    case kProSilver:
      return ((dx >= -1 && dx <= 1) && (dy == 0 || dy == -1)) ||
             (dx == 0 && dy == 1);
    case kKing:
      return std::abs(dx) <= 1 && std::abs(dy) <= 1;
    case kHorse:
      return (std::abs(dx) <= 1 && std::abs(dy) <= 1) || dx == dy || dx == -dy;
    case kDragon:
      return (std::abs(dx) <= 1 && std::abs(dy) <= 1) || dx == 0 || dy == 0;
    default:
      return false;
  }
}

inline bool IsSlidingMove(int piece, int from, int to) {
  const int dx = X(to) - X(from);
  const int dy = Y(to) - Y(from);
  if (piece == kLance) {
    return dx == 0 && std::abs(dy) > 1;
  }
  if (piece == kBishop || piece == kHorse) {
    return std::abs(dx) == std::abs(dy) && std::abs(dx) > 1;
  }
  if (piece == kRook || piece == kDragon) {
    return (dx == 0 || dy == 0) && (std::abs(dx) > 1 || std::abs(dy) > 1);
  }
  return false;
}

inline bool PathClear(const Board& board, int piece, int from, int to) {
  if (!IsSlidingMove(piece, from, to)) {
    return true;
  }
  const int dx = Sign(X(to) - X(from));
  const int dy = Sign(Y(to) - Y(from));
  int x = X(from) + dx;
  int y = Y(from) + dy;
  while (x != X(to) || y != Y(to)) {
    if (board[x * kBoardSize + y] != kEmpty) {
      return false;
    }
    x += dx;
    y += dy;
  }
  return true;
}

inline std::array<int, 8> LegalFromCandidates(int direction, int to) {
  std::array<int, 8> out{};
  out.fill(-1);
  int dx = 0;
  int dy = 0;
  switch (direction) {
    case 0:
      dx = 0;
      dy = 1;
      break;
    case 1:
      dx = -1;
      dy = 1;
      break;
    case 2:
      dx = 1;
      dy = 1;
      break;
    case 3:
      dx = -1;
      dy = 0;
      break;
    case 4:
      dx = 1;
      dy = 0;
      break;
    case 5:
      dx = 0;
      dy = -1;
      break;
    case 6:
      dx = -1;
      dy = -1;
      break;
    case 7:
      dx = 1;
      dy = -1;
      break;
    case 8:
      dx = -1;
      dy = 2;
      break;
    case 9:
      dx = 1;
      dy = 2;
      break;
    default:
      break;
  }
  int x = X(to);
  int y = Y(to);
  for (int i = 0; i < 8; ++i) {
    x += dx;
    y += dy;
    if (x < 0 || x >= kBoardSize || y < 0 || y >= kBoardSize) {
      break;
    }
    out[i] = x * kBoardSize + y;
    if (direction == 8 || direction == 9) {
      break;
    }
  }
  return out;
}

inline Action DecodeAction(const Board& board, int label) {
  const int direction = label / kSquares;
  const int to = label % kSquares;
  const bool is_drop = direction >= 20;
  const bool is_promotion = direction >= 10 && direction < 20;
  Action action;
  action.is_drop = is_drop;
  action.is_promotion = is_promotion;
  action.to = to;
  if (is_drop) {
    action.piece = direction - 20;
    return action;
  }
  const auto candidates = LegalFromCandidates(direction % 10, to);
  for (int from : candidates) {
    if (from >= 0 && IsOwn(board[from])) {
      action.from = from;
      action.piece = board[from];
      return action;
    }
  }
  action.from = -1;
  action.piece = kEmpty;
  return action;
}

}  // namespace shogi

class ShogiEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("shogi")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({-1, 9, 9, 119})),
                    "info:board"_.Bind(Spec<int>({9, 9}, {-1, 27})),
                    "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
                    "info:hand"_.Bind(Spec<int>({2, 7})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({2187})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})),
                    "info:turn"_.Bind(Spec<int>({}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 2186})));
  }
};

using ShogiEnvSpec = EnvSpec<ShogiEnvFns>;

class ShogiEnv : public Env<ShogiEnvSpec>, public RenderableEnv {
 public:
  using Spec = ShogiEnvSpec;
  using Action = typename Env<ShogiEnvSpec>::Action;

  ShogiEnv(const Spec& spec, int env_id) : Env<ShogiEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    const int first_player = static_cast<int>(gen_() & 1U);
    player_order_ = {first_player, 1 - first_player};
    current_player_ = first_player;
    color_ = 0;
    board_ = shogi::kInitBoard;
    hand_ = {};
    step_count_ = 0;
    done_ = false;
    rewards_ = {0.0f, 0.0f};
    UpdateLegalActionMask();
    WriteState(rewards_);
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal = act < 0 || act >= shogi::kActions ||
                         (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act < shogi::kActions) {
      StepGame(act);
    }
    if (illegal) {
      done_ = true;
      rewards_ = board_games::IllegalRewards(loser);
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
    } else if (done_) {
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
    }
    WriteState(rewards_);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::DrawGrid(rgb, width, height, 9, 9);
    for (int pos = 0; pos < shogi::kSquares; ++pos) {
      const int piece = board_[pos];
      if (piece == shogi::kEmpty) {
        continue;
      }
      const int row = 8 - shogi::Y(pos);
      const int col = shogi::X(pos);
      const int cx = col * width / 9 + width / 18;
      const int cy = row * height / 9 + height / 18;
      const bool mine = shogi::IsOwn(piece);
      board_games::DrawCircle(rgb, width, height, cx, cy,
                              std::min(width, height) / 28,
                              mine ? board_games::Rgb{35, 35, 35}
                                   : board_games::Rgb{232, 232, 224});
    }
  }

 private:
  shogi::Board board_{};
  shogi::Hand hand_{};
  shogi::Mask legal_action_mask_{};
  std::array<float, 2> rewards_{};
  std::array<int, 2> player_order_{0, 1};
  int color_{0};
  int current_player_{0};
  int step_count_{0};
  bool done_{true};

  static bool IsPseudoLegalMove(const shogi::Board& board, int from, int to) {
    if (from < 0 || to < 0 || from >= shogi::kSquares ||
        to >= shogi::kSquares) {
      return false;
    }
    const int piece = board[from];
    if (!shogi::IsOwn(piece) || shogi::IsOwn(board[to])) {
      return false;
    }
    return shogi::CanMoveTo(piece, from, to) &&
           shogi::PathClear(board, piece, from, to);
  }

  static bool IsChecked(const shogi::Board& board) {
    int king_pos = -1;
    for (int i = 0; i < shogi::kSquares; ++i) {
      if (board[i] == shogi::kKing) {
        king_pos = i;
        break;
      }
    }
    if (king_pos < 0) {
      return false;
    }
    const shogi::Board flipped = shogi::FlipBoard(board);
    const int target = shogi::kSquares - 1 - king_pos;
    for (int from = 0; from < shogi::kSquares; ++from) {
      if (IsPseudoLegalMove(flipped, from, target)) {
        return true;
      }
    }
    return false;
  }

  static bool IsLegalMoveWoPro(const shogi::Board& board, int from, int to) {
    if (!IsPseudoLegalMove(board, from, to)) {
      return false;
    }
    shogi::Board next = board;
    next[to] = next[from];
    next[from] = shogi::kEmpty;
    return !IsChecked(next);
  }

  static bool IsNoPromotionLegal(const shogi::Board& board, int from, int to) {
    if (from < 0 || from >= shogi::kSquares || to < 0 ||
        to >= shogi::kSquares) {
      return false;
    }
    const int piece = board[from];
    bool illegal =
        (piece == shogi::kPawn || piece == shogi::kLance) && shogi::Y(to) == 0;
    illegal = illegal || (piece == shogi::kKnight && shogi::Y(to) < 2);
    return !illegal;
  }

  static bool IsPromotionLegal(const shogi::Board& board, int from, int to) {
    if (from < 0 || from >= shogi::kSquares || to < 0 ||
        to >= shogi::kSquares) {
      return false;
    }
    const int piece = board[from];
    bool illegal = piece >= shogi::kGold && piece <= shogi::kDragon;
    illegal = illegal || (shogi::Y(from) >= 3 && shogi::Y(to) >= 3);
    return !illegal;
  }

  bool IsLegalDropWoPiece(int to) const {
    if (to < 0 || to >= shogi::kSquares || board_[to] != shogi::kEmpty) {
      return false;
    }
    shogi::Board next = board_;
    next[to] = shogi::kPawn;
    return !IsChecked(next);
  }

  bool IsLegalDropWoIgnoringCheck(int piece, int to) const {
    if (piece < 0 || piece >= 7 || to < 0 || to >= shogi::kSquares ||
        board_[to] != shogi::kEmpty || hand_[0][piece] <= 0) {
      return false;
    }
    if (piece == shogi::kPawn) {
      for (int y = 0; y < shogi::kBoardSize; ++y) {
        if (board_[shogi::X(to) * shogi::kBoardSize + y] == shogi::kPawn) {
          return false;
        }
      }
    }
    if ((piece == shogi::kPawn || piece == shogi::kLance) &&
        shogi::Y(to) == 0) {
      return false;
    }
    if (piece == shogi::kKnight && shogi::Y(to) < 2) {
      return false;
    }
    return true;
  }

  static std::array<int, 8> Around(int pos) {
    static constexpr std::array<int, 8> k_dx = {-1, -1, 0, 1, 1, 1, 0, -1};
    static constexpr std::array<int, 8> k_dy = {0, -1, -1, -1, 0, 1, 1, 1};
    std::array<int, 8> out{};
    out.fill(-1);
    for (int i = 0; i < 8; ++i) {
      const int x = shogi::X(pos) + k_dx[i];
      const int y = shogi::Y(pos) + k_dy[i];
      if (x >= 0 && x < shogi::kBoardSize && y >= 0 && y < shogi::kBoardSize) {
        out[i] = x * shogi::kBoardSize + y;
      }
    }
    return out;
  }

  bool IsDropPawnMate(int* pawn_to) const {
    int opp_king_pos = -1;
    for (int i = 0; i < shogi::kSquares; ++i) {
      if (board_[i] == shogi::kOppKing) {
        opp_king_pos = i;
        break;
      }
    }
    if (opp_king_pos < 0) {
      *pawn_to = 0;
      return false;
    }
    const int to = opp_king_pos + 1;
    *pawn_to = to;
    if (to < 0 || to >= shogi::kSquares) {
      return false;
    }
    shogi::Board with_pawn = board_;
    with_pawn[to] = shogi::kPawn;
    const shogi::Board flipped = shogi::FlipBoard(with_pawn);
    const int flipped_to = shogi::kSquares - 1 - to;

    bool can_capture_pawn = false;
    for (int from = 0; from < shogi::kSquares; ++from) {
      can_capture_pawn =
          can_capture_pawn || IsLegalMoveWoPro(flipped, from, flipped_to);
    }

    bool can_king_escape = false;
    const int king_from = shogi::kSquares - 1 - opp_king_pos;
    for (int escape_to : Around(king_from)) {
      can_king_escape =
          can_king_escape || IsLegalMoveWoPro(flipped, king_from, escape_to);
    }
    return !(can_capture_pawn || can_king_escape);
  }

  bool IsLegalMoveLabel(int label) const {
    const shogi::Action action = shogi::DecodeAction(board_, label);
    if (action.is_drop) {
      return false;
    }
    const int base_direction = (label / shogi::kSquares) % 10;
    const int base_label = base_direction * shogi::kSquares + action.to;
    const shogi::Action base = shogi::DecodeAction(board_, base_label);
    bool ok = IsLegalMoveWoPro(board_, base.from, base.to);
    ok = ok &&
         (action.is_promotion ? IsPromotionLegal(board_, base.from, base.to)
                              : IsNoPromotionLegal(board_, base.from, base.to));
    return ok;
  }

  void UpdateLegalActionMask() {
    std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), false);
    for (int label = 0; label < 20 * shogi::kSquares; ++label) {
      legal_action_mask_[label] = IsLegalMoveLabel(label);
    }
    for (int to = 0; to < shogi::kSquares; ++to) {
      const bool drop_square_ok = IsLegalDropWoPiece(to);
      for (int piece = 0; piece < 7; ++piece) {
        const int label = (20 + piece) * shogi::kSquares + to;
        legal_action_mask_[label] =
            drop_square_ok && IsLegalDropWoIgnoringCheck(piece, to);
      }
    }

    int pawn_to = 0;
    const bool drop_pawn_mate = IsDropPawnMate(&pawn_to);
    const int pawn_label = 20 * shogi::kSquares + pawn_to;
    if (pawn_to >= 0 && pawn_to < shogi::kSquares) {
      legal_action_mask_[pawn_label] =
          legal_action_mask_[pawn_label] && !drop_pawn_mate;
    }
  }

  void StepGame(int label) {
    const shogi::Action action = shogi::DecodeAction(board_, label);
    if (action.is_drop) {
      board_[action.to] = action.piece;
      --hand_[0][action.piece];
    } else {
      const int captured = board_[action.to];
      if (captured != shogi::kEmpty) {
        ++hand_[0][((captured + 14) % 28) % 8];
      }
      int piece = action.is_promotion ? action.piece + 8 : action.piece;
      board_[action.from] = shogi::kEmpty;
      board_[action.to] = piece;
    }
    board_ = shogi::FlipBoard(board_);
    hand_ = shogi::FlipHand(hand_);
    color_ = 1 - color_;
    current_player_ = player_order_[color_];
    ++step_count_;
    UpdateLegalActionMask();
    const bool has_legal =
        std::any_of(legal_action_mask_.begin(), legal_action_mask_.end(),
                    [](bool value) { return value; });
    done_ = !has_legal || step_count_ >= shogi::kMaxTerminationSteps;
    rewards_ = {0.0f, 0.0f};
    if (!has_legal) {
      const std::array<float, 2> color_rewards =
          color_ == 0 ? std::array<float, 2>{-1.0f, 1.0f}
                      : std::array<float, 2>{1.0f, -1.0f};
      rewards_[player_order_[0]] = color_rewards[0];
      rewards_[player_order_[1]] = color_rewards[1];
    }
  }

  static bool HasEffect(const shogi::Board& board, int from, int to) {
    if (from < 0 || from >= shogi::kSquares || !shogi::IsOwn(board[from])) {
      return false;
    }
    const int piece = board[from];
    return shogi::CanMoveTo(piece, from, to) &&
           shogi::PathClear(board, piece, from, to);
  }

  static void BuildFeaturePlanes(
      const shogi::Board& board, bool reverse_flat,
      std::array<std::array<bool, shogi::kSquares>, 14>* piece_planes,
      std::array<std::array<bool, shogi::kSquares>, 14>* effect_planes,
      std::array<std::array<bool, shogi::kSquares>, 3>* effect_sum_planes) {
    for (auto& plane : *piece_planes) {
      plane.fill(false);
    }
    for (auto& plane : *effect_planes) {
      plane.fill(false);
    }
    std::array<int, shogi::kSquares> effect_count{};
    effect_count.fill(0);
    for (int from = 0; from < shogi::kSquares; ++from) {
      const int piece = board[from];
      if (!shogi::IsOwn(piece)) {
        continue;
      }
      const int piece_sq = reverse_flat ? shogi::kSquares - 1 - from : from;
      (*piece_planes)[piece][piece_sq] = true;
      for (int to = 0; to < shogi::kSquares; ++to) {
        if (HasEffect(board, from, to)) {
          const int effect_sq = reverse_flat ? shogi::kSquares - 1 - to : to;
          (*effect_planes)[piece][effect_sq] = true;
          ++effect_count[effect_sq];
        }
      }
    }
    for (auto& plane : *effect_sum_planes) {
      plane.fill(false);
    }
    for (int sq = 0; sq < shogi::kSquares; ++sq) {
      for (int n = 0; n < 3; ++n) {
        (*effect_sum_planes)[n][sq] = effect_count[sq] >= n + 1;
      }
    }
  }

  static int ObsSquare(int row, int col) {
    return (shogi::kBoardSize - 1 - col) * shogi::kBoardSize + row;
  }

  static void SetObsPlane(State* state, int player, int channel,
                          const std::array<bool, shogi::kSquares>& flat) {
    for (int row = 0; row < shogi::kBoardSize; ++row) {
      for (int col = 0; col < shogi::kBoardSize; ++col) {
        (*state)["obs"_](player, row, col, channel) = flat[ObsSquare(row, col)];
      }
    }
  }

  static void SetConstantObsPlane(State* state, int player, int channel,
                                  bool value) {
    for (int row = 0; row < shogi::kBoardSize; ++row) {
      for (int col = 0; col < shogi::kBoardSize; ++col) {
        (*state)["obs"_](player, row, col, channel) = value;
      }
    }
  }

  void WriteHandPlanes(State* state, int player, int* channel,
                       const std::array<int, 7>& hand) const {
    for (int n = 1; n <= 8; ++n) {
      SetConstantObsPlane(state, player, (*channel)++, hand[shogi::kPawn] >= n);
    }
    for (int piece :
         {shogi::kLance, shogi::kKnight, shogi::kSilver, shogi::kGold}) {
      for (int n = 1; n <= 4; ++n) {
        SetConstantObsPlane(state, player, (*channel)++, hand[piece] >= n);
      }
    }
    for (int piece : {shogi::kBishop, shogi::kRook}) {
      for (int n = 1; n <= 2; ++n) {
        SetConstantObsPlane(state, player, (*channel)++, hand[piece] >= n);
      }
    }
  }

  void WriteObservationForPlayer(State* state, int player) const {
    shogi::Board view_board = board_;
    shogi::Hand view_hand = hand_;
    shogi::Board opp_board = shogi::FlipBoard(board_);
    if (player != current_player_) {
      view_board = opp_board;
      view_hand = shogi::FlipHand(hand_);
      opp_board = board_;
    }

    std::array<std::array<bool, shogi::kSquares>, 14> my_piece{};
    std::array<std::array<bool, shogi::kSquares>, 14> my_effect{};
    std::array<std::array<bool, shogi::kSquares>, 3> my_effect_sum{};
    std::array<std::array<bool, shogi::kSquares>, 14> opp_piece{};
    std::array<std::array<bool, shogi::kSquares>, 14> opp_effect{};
    std::array<std::array<bool, shogi::kSquares>, 3> opp_effect_sum{};
    BuildFeaturePlanes(view_board, false, &my_piece, &my_effect,
                       &my_effect_sum);
    BuildFeaturePlanes(opp_board, true, &opp_piece, &opp_effect,
                       &opp_effect_sum);

    int channel = 0;
    for (const auto& plane : my_piece) {
      SetObsPlane(state, player, channel++, plane);
    }
    for (const auto& plane : my_effect) {
      SetObsPlane(state, player, channel++, plane);
    }
    for (const auto& plane : my_effect_sum) {
      SetObsPlane(state, player, channel++, plane);
    }
    for (const auto& plane : opp_piece) {
      SetObsPlane(state, player, channel++, plane);
    }
    for (const auto& plane : opp_effect) {
      SetObsPlane(state, player, channel++, plane);
    }
    for (const auto& plane : opp_effect_sum) {
      SetObsPlane(state, player, channel++, plane);
    }
    WriteHandPlanes(state, player, &channel, view_hand[0]);
    WriteHandPlanes(state, player, &channel, view_hand[1]);
    SetConstantObsPlane(state, player, channel++, IsChecked(view_board));
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    for (int row = 0; row < shogi::kBoardSize; ++row) {
      for (int col = 0; col < shogi::kBoardSize; ++col) {
        state["info:board"_](row, col) =
            board_[(shogi::kBoardSize - 1 - col) * shogi::kBoardSize + row];
      }
    }
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 7; ++col) {
        state["info:hand"_](row, col) = hand_[row][col];
      }
    }
    for (int i = 0; i < shogi::kActions; ++i) {
      state["info:legal_action_mask"_][i] = legal_action_mask_[i];
    }
    state["info:current_player"_] = current_player_;
    state["info:turn"_] = color_;
    for (int player = 0; player < 2; ++player) {
      state["info:players.id"_][player] = player;
      state["reward"_][player] = rewards[player];
      WriteObservationForPlayer(&state, player);
    }
  }
};

using ShogiEnvPool = AsyncEnvPool<ShogiEnv>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_SHOGI_H_
