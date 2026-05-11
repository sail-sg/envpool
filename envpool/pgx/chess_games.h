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

#ifndef ENVPOOL_PGX_CHESS_GAMES_H_
#define ENVPOOL_PGX_CHESS_GAMES_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/pgx/board_games.h"

namespace pgx {
namespace gardner_chess {

constexpr int kSize = 5;
constexpr int kSquares = 25;
constexpr int kPlanes = 49;
constexpr int kActions = kSquares * kPlanes;
constexpr int kHistory = 8;
constexpr int kMaxTerminationSteps = 256;
constexpr int kMaxPieces = 10;

enum Piece {
  kEmpty = 0,
  kPawn = 1,
  kKnight = 2,
  kBishop = 3,
  kRook = 4,
  kQueen = 5,
  kKing = 6,
};

using Board = std::array<int, kSquares>;
using BoardHistory = std::array<Board, kHistory>;
using Hash = std::array<uint32_t, 2>;

constexpr Board kInitBoard = {4,  1,  0, -1, -4, 2,  1,  0, -1, -2, 3,  1, 0,
                              -1, -3, 5, 1,  0,  -1, -5, 6, 1,  0,  -1, -6};
constexpr Hash kInitHash = {2025569903U, 1172890342U};

inline int Row(int pos) { return pos % kSize; }
inline int Col(int pos) { return pos / kSize; }
inline int Pos(int row, int col) { return col * kSize + row; }
inline int FlipPos(int pos) {
  return pos < 0 ? pos : (pos / kSize) * kSize + (kSize - 1 - (pos % kSize));
}

struct Action {
  int from = -1;
  int to = -1;
  int underpromotion = -1;
};

struct Tables {
  std::array<std::array<int, kPlanes>, kSquares> to_map{};
  std::array<std::array<int, kSquares>, kSquares> plane_map{};
  std::array<std::array<std::array<int, 16>, kSquares>, 7> can_move{};
  std::array<std::array<int, 24>, kSquares> can_move_any{};
  std::array<std::array<std::array<int, 3>, kSquares>, kSquares> between{};
};

inline Tables MakeTables() {
  Tables tables;
  for (auto& row : tables.to_map) {
    row.fill(-1);
  }
  for (auto& row : tables.plane_map) {
    row.fill(-1);
  }
  for (auto& piece : tables.can_move) {
    for (auto& row : piece) {
      row.fill(-1);
    }
  }
  for (auto& row : tables.can_move_any) {
    row.fill(-1);
  }
  for (auto& row : tables.between) {
    for (auto& col : row) {
      col.fill(-1);
    }
  }

  for (int from = 0; from < kSquares; ++from) {
    if (Row(from) == 3) {
      for (int plane = 0; plane < 9; ++plane) {
        const int to = from + std::array<int, 3>{1, 6, -4}[plane % 3];
        if (0 <= to && to < kSquares) {
          tables.to_map[from][plane] = to;
        }
      }
    }
  }

  static constexpr std::array<int, 40> kDr = {
      -4, -3, -2, -1, 1, 2, 3, 4, 0,  0,  0,  0,  0,  0, 0,  0, -4, -3, -2, -1,
      1,  2,  3,  4,  4, 3, 2, 1, -1, -2, -3, -4, -1, 1, -2, 2, -1, 1,  -2, 2};
  static constexpr std::array<int, 40> kDc = {
      0, 0, 0,  0,  0,  0,  0,  0,  -4, -3, -2, -1, 1,  2,
      3, 4, -4, -3, -2, -1, 1,  2,  3,  4,  -4, -3, -2, -1,
      1, 2, 3,  4,  -2, -2, -1, -1, 2,  2,  1,  1};
  for (int from = 0; from < kSquares; ++from) {
    for (int plane = 9; plane < kPlanes; ++plane) {
      const int row = Row(from) + kDr[plane - 9];
      const int col = Col(from) + kDc[plane - 9];
      if (0 <= row && row < kSize && 0 <= col && col < kSize) {
        const int to = Pos(row, col);
        tables.to_map[from][plane] = to;
        tables.plane_map[from][to] = plane;
      }
    }
  }

  for (int from = 0; from < kSquares; ++from) {
    std::array<std::vector<int>, 7> legal_dest;
    const int r0 = Row(from);
    const int c0 = Col(from);
    for (int to = 0; to < kSquares; ++to) {
      if (from == to) {
        continue;
      }
      const int r1 = Row(to);
      const int c1 = Col(to);
      if (r1 - r0 == 1 && std::abs(c1 - c0) <= 1) {
        legal_dest[kPawn].push_back(to);
      }
      if ((std::abs(r1 - r0) == 1 && std::abs(c1 - c0) == 2) ||
          (std::abs(r1 - r0) == 2 && std::abs(c1 - c0) == 1)) {
        legal_dest[kKnight].push_back(to);
      }
      if (std::abs(r1 - r0) == std::abs(c1 - c0)) {
        legal_dest[kBishop].push_back(to);
      }
      if (std::abs(r1 - r0) == 0 || std::abs(c1 - c0) == 0) {
        legal_dest[kRook].push_back(to);
      }
      if (std::abs(r1 - r0) == 0 || std::abs(c1 - c0) == 0 ||
          std::abs(r1 - r0) == std::abs(c1 - c0)) {
        legal_dest[kQueen].push_back(to);
      }
      if (std::abs(r1 - r0) <= 1 && std::abs(c1 - c0) <= 1) {
        legal_dest[kKing].push_back(to);
      }
    }
    for (int piece = 1; piece <= 6; ++piece) {
      for (int i = 0; i < static_cast<int>(legal_dest[piece].size()); ++i) {
        tables.can_move[piece][from][i] = legal_dest[piece][i];
      }
    }
    std::vector<int> any;
    for (int to : legal_dest[kQueen]) {
      any.push_back(to);
    }
    for (int to : legal_dest[kKnight]) {
      if (std::find(any.begin(), any.end(), to) == any.end()) {
        any.push_back(to);
      }
    }
    for (int i = 0; i < static_cast<int>(any.size()); ++i) {
      tables.can_move_any[from][i] = any[i];
    }
  }

  for (int from = 0; from < kSquares; ++from) {
    for (int to = 0; to < kSquares; ++to) {
      const int r0 = Row(from);
      const int c0 = Col(from);
      const int r1 = Row(to);
      const int c1 = Col(to);
      if (!(std::abs(r1 - r0) == 0 || std::abs(c1 - c0) == 0 ||
            std::abs(r1 - r0) == std::abs(c1 - c0))) {
        continue;
      }
      const int dr = std::max(-1, std::min(1, r1 - r0));
      const int dc = std::max(-1, std::min(1, c1 - c0));
      int row = r0;
      int col = c0;
      int out = 0;
      while (true) {
        row += dr;
        col += dc;
        if (row == r1 && col == c1) {
          break;
        }
        if (out < 3) {
          tables.between[from][to][out++] = Pos(row, col);
        }
      }
    }
  }
  return tables;
}

inline const Tables& GetTables() {
  static const Tables* tables = new Tables(MakeTables());
  return *tables;
}

inline Action FromLabel(int label) {
  const int from = label / kPlanes;
  const int plane = label % kPlanes;
  return {from, GetTables().to_map[from][plane], plane >= 9 ? -1 : plane / 3};
}

inline int ToLabel(const Action& action) {
  return action.from * kPlanes + GetTables().plane_map[action.from][action.to];
}

inline Board FlipBoard(const Board& board) {
  Board flipped{};
  for (int pos = 0; pos < kSquares; ++pos) {
    flipped[FlipPos(pos)] = -board[pos];
  }
  return flipped;
}

inline BoardHistory FlipHistory(const BoardHistory& history) {
  BoardHistory out{};
  for (int i = 0; i < kHistory; ++i) {
    out[i] = FlipBoard(history[i]);
  }
  return out;
}

inline uint64_t BoardKey(const Board& board, int turn) {
  uint64_t key = static_cast<uint64_t>(turn + 1);
  for (int value : board) {
    key = key * 1315423911ULL + static_cast<uint64_t>(value + 7);
  }
  return key;
}

}  // namespace gardner_chess

class GardnerChessEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("gardner_chess")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<float>({-1, 5, 5, 115})),
                    "info:board"_.Bind(Spec<int>({5, 5})),
                    "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
                    "info:fullmove_count"_.Bind(Spec<int>({})),
                    "info:halfmove_count"_.Bind(Spec<int>({})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({1225})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})),
                    "info:turn"_.Bind(Spec<int>({}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 1224})));
  }
};

using GardnerChessEnvSpec = EnvSpec<GardnerChessEnvFns>;

class GardnerChessEnv : public Env<GardnerChessEnvSpec>, public RenderableEnv {
 public:
  using Spec = GardnerChessEnvSpec;
  using Action = typename Env<GardnerChessEnvSpec>::Action;

  GardnerChessEnv(const Spec& spec, int env_id)
      : Env<GardnerChessEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    current_player_ = static_cast<int>(gen_() & 1U);
    turn_ = 0;
    board_ = gardner_chess::kInitBoard;
    board_history_.fill({});
    board_history_[0] = board_;
    seen_keys_.fill(0);
    seen_keys_[0] = gardner_chess::BoardKey(board_, turn_);
    halfmove_count_ = 0;
    fullmove_count_ = 1;
    step_count_ = 0;
    rewards_ = {0.0f, 0.0f};
    done_ = false;
    UpdateLegalActionMask();
    WriteState(rewards_);
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal = act < 0 || act >= gardner_chess::kActions ||
                         (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act < gardner_chess::kActions) {
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
    return {width > 0 ? width : 200, height > 0 ? height : 200};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::DrawGrid(rgb, width, height, 5, 5);
    for (int pos = 0; pos < gardner_chess::kSquares; ++pos) {
      const int piece = board_[pos];
      if (piece == 0) {
        continue;
      }
      const int row = 4 - gardner_chess::Row(pos);
      const int col = gardner_chess::Col(pos);
      const int cx = col * width / 5 + width / 10;
      const int cy = row * height / 5 + height / 10;
      board_games::DrawCircle(rgb, width, height, cx, cy,
                              std::min(width, height) / 16,
                              piece > 0 ? board_games::Rgb{35, 35, 35}
                                        : board_games::Rgb{235, 235, 225});
    }
  }

 private:
  gardner_chess::Board board_{};
  gardner_chess::BoardHistory board_history_{};
  std::array<uint64_t, gardner_chess::kMaxTerminationSteps + 1> seen_keys_{};
  std::array<bool, gardner_chess::kActions> legal_action_mask_{};
  std::array<float, 2> rewards_{};
  int current_player_{0};
  int turn_{0};
  int halfmove_count_{0};
  int fullmove_count_{1};
  int step_count_{0};
  bool done_{true};

  static bool IsSlidingPiece(int piece) {
    piece = std::abs(piece);
    return piece == gardner_chess::kBishop || piece == gardner_chess::kRook ||
           piece == gardner_chess::kQueen;
  }

  bool IsPseudoLegal(const gardner_chess::Board& board,
                     const gardner_chess::Action& action) const {
    if (action.from < 0 || action.from >= gardner_chess::kSquares ||
        action.to < 0 || action.to >= gardner_chess::kSquares) {
      return false;
    }
    const int piece = board[action.from];
    if (piece < 0 || board[action.to] > 0) {
      return false;
    }
    const auto& moves = gardner_chess::GetTables().can_move[piece][action.from];
    if (std::find(moves.begin(), moves.end(), action.to) == moves.end()) {
      return false;
    }
    if (IsSlidingPiece(piece)) {
      for (int between :
           gardner_chess::GetTables().between[action.from][action.to]) {
        if (between >= 0 && board[between] != gardner_chess::kEmpty) {
          return false;
        }
      }
    }
    if (piece == gardner_chess::kPawn) {
      const bool same_file =
          gardner_chess::Col(action.to) == gardner_chess::Col(action.from);
      if (same_file && board[action.to] < 0) {
        return false;
      }
      if (!same_file && board[action.to] >= 0) {
        return false;
      }
    }
    return true;
  }

  bool IsAttacking(const gardner_chess::Board& board, int pos) const {
    for (int from : gardner_chess::GetTables().can_move_any[pos]) {
      if (from >= 0 && IsPseudoLegal(board, {from, pos, -1})) {
        return true;
      }
    }
    return false;
  }

  bool IsChecking(const gardner_chess::Board& board) const {
    int king = -1;
    for (int i = 0; i < gardner_chess::kSquares; ++i) {
      if (board[i] == -gardner_chess::kKing) {
        king = i;
        break;
      }
    }
    return king >= 0 && IsAttacking(board, king);
  }

  gardner_chess::Board ApplyMove(gardner_chess::Board board,
                                 const gardner_chess::Action& action) const {
    int piece = board[action.from];
    if (piece == gardner_chess::kPawn && gardner_chess::Row(action.from) == 3 &&
        action.underpromotion < 0) {
      piece = gardner_chess::kQueen;
    }
    if (action.underpromotion >= 0) {
      static constexpr std::array<int, 3> kUnderpromotions = {
          gardner_chess::kRook,
          gardner_chess::kBishop,
          gardner_chess::kKnight,
      };
      piece = kUnderpromotions[action.underpromotion];
    }
    board[action.from] = gardner_chess::kEmpty;
    board[action.to] = piece;
    return board;
  }

  bool IsLegal(const gardner_chess::Action& action) const {
    if (!IsPseudoLegal(board_, action)) {
      return false;
    }
    const gardner_chess::Board next =
        gardner_chess::FlipBoard(ApplyMove(board_, action));
    return !IsChecking(next);
  }

  bool HasInsufficientPieces() const {
    int pieces = 0;
    int pawn_rook_queen = 0;
    int bishops = 0;
    int bishops_on_black = 0;
    for (int pos = 0; pos < gardner_chess::kSquares; ++pos) {
      const int piece = std::abs(board_[pos]);
      if (piece == 0) {
        continue;
      }
      ++pieces;
      if (piece >= gardner_chess::kRook || piece == gardner_chess::kPawn) {
        ++pawn_rook_queen;
      }
      if (piece == gardner_chess::kBishop) {
        ++bishops;
        if ((pos % 2) == 0) {
          ++bishops_on_black;
        }
      }
    }
    pawn_rook_queen -= 2;
    return pieces <= 2 || (pieces == 3 && pawn_rook_queen == 0) ||
           (pieces == bishops + 2 &&
            (bishops_on_black == bishops || bishops_on_black == 0));
  }

  int RepetitionCount(uint64_t key) const {
    return static_cast<int>(
               std::count(seen_keys_.begin(), seen_keys_.end(), key)) -
           1;
  }

  void UpdateLegalActionMask() {
    std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), false);
    for (int from = 0; from < gardner_chess::kSquares; ++from) {
      const int piece = board_[from];
      if (piece <= 0) {
        continue;
      }
      for (int to : gardner_chess::GetTables().can_move[piece][from]) {
        if (to < 0) {
          continue;
        }
        gardner_chess::Action action{from, to, -1};
        if (IsLegal(action)) {
          legal_action_mask_[gardner_chess::ToLabel(action)] = true;
        }
      }
    }
    for (int from : {3, 8, 13, 18, 23}) {
      for (int plane = 0; plane < 9; ++plane) {
        const int label = from * gardner_chess::kPlanes + plane;
        const gardner_chess::Action action = gardner_chess::FromLabel(label);
        if (board_[action.from] == gardner_chess::kPawn && action.to >= 0 &&
            IsLegal({action.from, action.to, -1})) {
          legal_action_mask_[label] = true;
        }
      }
    }
  }

  void UpdateHistory() {
    for (int i = gardner_chess::kHistory - 1; i > 0; --i) {
      board_history_[i] = board_history_[i - 1];
    }
    board_history_[0] = board_;
    for (int i = gardner_chess::kMaxTerminationSteps; i > 0; --i) {
      seen_keys_[i] = seen_keys_[i - 1];
    }
    seen_keys_[0] = gardner_chess::BoardKey(board_, turn_);
  }

  void StepGame(int label) {
    const gardner_chess::Action action = gardner_chess::FromLabel(label);
    const int piece = board_[action.from];
    const bool captured = action.to >= 0 && board_[action.to] < 0;
    halfmove_count_ =
        captured || piece == gardner_chess::kPawn ? 0 : halfmove_count_ + 1;
    fullmove_count_ += turn_ == 1 ? 1 : 0;
    board_ = ApplyMove(board_, action);
    board_ = gardner_chess::FlipBoard(board_);
    board_history_ = gardner_chess::FlipHistory(board_history_);
    turn_ = 1 - turn_;
    current_player_ = 1 - current_player_;
    ++step_count_;
    UpdateHistory();
    UpdateLegalActionMask();

    const bool has_legal =
        std::any_of(legal_action_mask_.begin(), legal_action_mask_.end(),
                    [](bool value) { return value; });
    done_ = !has_legal || halfmove_count_ >= 100 || HasInsufficientPieces() ||
            RepetitionCount(seen_keys_[0]) >= 2 ||
            step_count_ >= gardner_chess::kMaxTerminationSteps;
    const bool is_checkmate =
        !has_legal && IsChecking(gardner_chess::FlipBoard(board_));
    rewards_ = {0.0f, 0.0f};
    if (is_checkmate) {
      rewards_ = {1.0f, 1.0f};
      rewards_[current_player_] = -1.0f;
    }
  }

  void WriteObservationForPlayer(State* state, int player) const {
    const bool current_view = current_player_ == player;
    gardner_chess::BoardHistory history =
        current_view ? board_history_
                     : gardner_chess::FlipHistory(board_history_);
    const int color = current_view ? turn_ : 1 - turn_;
    int channel = 0;
    for (int h = 0; h < gardner_chess::kHistory; ++h) {
      for (int piece = 1; piece <= 6; ++piece, ++channel) {
        for (int row = 0; row < 5; ++row) {
          for (int col = 0; col < 5; ++col) {
            const int pos = col * 5 + (4 - row);
            (*state)["obs"_](player, row, col, channel) =
                history[h][pos] == piece ? 1.0f : 0.0f;
          }
        }
      }
      for (int piece = 1; piece <= 6; ++piece, ++channel) {
        for (int row = 0; row < 5; ++row) {
          for (int col = 0; col < 5; ++col) {
            const int pos = col * 5 + (4 - row);
            (*state)["obs"_](player, row, col, channel) =
                history[h][pos] == -piece ? 1.0f : 0.0f;
          }
        }
      }
      const float rep0 = 1.0f;
      const float rep1 = 0.0f;
      for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 5; ++col) {
          (*state)["obs"_](player, row, col, channel) = rep0;
          (*state)["obs"_](player, row, col, channel + 1) = rep1;
        }
      }
      channel += 2;
    }
    for (int row = 0; row < 5; ++row) {
      for (int col = 0; col < 5; ++col) {
        (*state)["obs"_](player, row, col, channel) = static_cast<float>(color);
        (*state)["obs"_](player, row, col, channel + 1) =
            static_cast<float>(step_count_) /
            static_cast<float>(gardner_chess::kMaxTerminationSteps);
        (*state)["obs"_](player, row, col, channel + 2) =
            static_cast<float>(halfmove_count_) / 100.0f;
      }
    }
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    for (int player = 0; player < 2; ++player) {
      WriteObservationForPlayer(&state, player);
      state["info:players.id"_][player] = player;
    }
    for (int row = 0; row < 5; ++row) {
      for (int col = 0; col < 5; ++col) {
        state["info:board"_](row, col) = board_[col * 5 + (4 - row)];
      }
    }
    for (int i = 0; i < gardner_chess::kActions; ++i) {
      state["info:legal_action_mask"_][i] = legal_action_mask_[i];
    }
    state["info:current_player"_] = current_player_;
    state["info:fullmove_count"_] = fullmove_count_;
    state["info:halfmove_count"_] = halfmove_count_;
    state["info:turn"_] = turn_;
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
    }
  }
};

using GardnerChessEnvPool = AsyncEnvPool<GardnerChessEnv>;

namespace chess {

constexpr int kSize = 8;
constexpr int kSquares = 64;
constexpr int kPlanes = 73;
constexpr int kActions = kSquares * kPlanes;
constexpr int kHistory = 8;
constexpr int kMaxTerminationSteps = 512;

enum Piece {
  kEmpty = 0,
  kPawn = 1,
  kKnight = 2,
  kBishop = 3,
  kRook = 4,
  kQueen = 5,
  kKing = 6,
};

using Board = std::array<int, kSquares>;
using BoardHistory = std::array<Board, kHistory>;
using CastlingRights = std::array<std::array<bool, 2>, 2>;

constexpr Board kInitBoard = {
    4, 1, 0, 0, 0, 0, -1, -4, 2, 1, 0, 0, 0, 0, -1, -2,
    3, 1, 0, 0, 0, 0, -1, -3, 5, 1, 0, 0, 0, 0, -1, -5,
    6, 1, 0, 0, 0, 0, -1, -6, 3, 1, 0, 0, 0, 0, -1, -3,
    2, 1, 0, 0, 0, 0, -1, -2, 4, 1, 0, 0, 0, 0, -1, -4,
};

inline int Row(int pos) { return pos % kSize; }
inline int Col(int pos) { return pos / kSize; }
inline int Pos(int row, int col) { return col * kSize + row; }
inline int FlipPos(int pos) {
  return pos < 0 ? pos : (pos / kSize) * kSize + (kSize - 1 - (pos % kSize));
}

struct Action {
  int from = -1;
  int to = -1;
  int underpromotion = -1;
};

struct Tables {
  std::array<std::array<int, kPlanes>, kSquares> from_plane{};
  std::array<std::array<int, kSquares>, kSquares> to_plane{};
  std::array<std::array<std::array<bool, kSquares>, kSquares>, 7> can_move{};
  std::array<std::array<std::array<int, 6>, kSquares>, kSquares> between{};
};

inline Tables MakeTables() {
  Tables tables;
  for (auto& row : tables.from_plane) {
    row.fill(-1);
  }
  for (auto& row : tables.to_plane) {
    row.fill(-1);
  }
  for (auto& piece : tables.can_move) {
    for (auto& row : piece) {
      row.fill(false);
    }
  }
  for (auto& row : tables.between) {
    for (auto& col : row) {
      col.fill(-1);
    }
  }

  std::array<int, 64> dr{};
  std::array<int, 64> dc{};
  int n = 0;
  for (int d = -7; d < 0; ++d) {
    dr[n] = d;
    dc[n++] = 0;
  }
  for (int d = 1; d <= 7; ++d) {
    dr[n] = d;
    dc[n++] = 0;
  }
  for (int d = -7; d < 0; ++d) {
    dr[n] = 0;
    dc[n++] = d;
  }
  for (int d = 1; d <= 7; ++d) {
    dr[n] = 0;
    dc[n++] = d;
  }
  for (int d = -7; d < 0; ++d) {
    dr[n] = d;
    dc[n++] = d;
  }
  for (int d = 1; d <= 7; ++d) {
    dr[n] = d;
    dc[n++] = d;
  }
  for (int d = 7; d >= 1; --d) {
    dr[n] = d;
    dc[n++] = -d;
  }
  for (int d = -1; d >= -7; --d) {
    dr[n] = d;
    dc[n++] = -d;
  }
  static constexpr std::array<int, 8> kKnightDr = {-1, 1, -2, 2, -1, 1, -2, 2};
  static constexpr std::array<int, 8> kKnightDc = {-2, -2, -1, -1, 2, 2, 1, 1};
  for (int i = 0; i < 8; ++i) {
    dr[n] = kKnightDr[i];
    dc[n++] = kKnightDc[i];
  }

  for (int from = 0; from < kSquares; ++from) {
    for (int plane = 0; plane < kPlanes; ++plane) {
      if (plane < 9) {
        const int to = Row(from) == 6
                           ? from + std::array<int, 3>{1, 9, -7}[plane % 3]
                           : -1;
        if (0 <= to && to < kSquares) {
          tables.from_plane[from][plane] = to;
        }
      } else {
        const int row = Row(from) + dr[plane - 9];
        const int col = Col(from) + dc[plane - 9];
        if (0 <= row && row < kSize && 0 <= col && col < kSize) {
          const int to = Pos(row, col);
          tables.from_plane[from][plane] = to;
          tables.to_plane[from][to] = plane;
        }
      }
    }
  }

  for (int from = 0; from < kSquares; ++from) {
    const int r0 = Row(from);
    const int c0 = Col(from);
    for (int to = 0; to < kSquares; ++to) {
      if (from == to) {
        continue;
      }
      const int r1 = Row(to);
      const int c1 = Col(to);
      if ((r1 - r0 == 1 && std::abs(c1 - c0) <= 1) ||
          (r0 == 1 && r1 == 3 && c1 == c0)) {
        tables.can_move[kPawn][from][to] = true;
      }
      if ((std::abs(r1 - r0) == 1 && std::abs(c1 - c0) == 2) ||
          (std::abs(r1 - r0) == 2 && std::abs(c1 - c0) == 1)) {
        tables.can_move[kKnight][from][to] = true;
      }
      if (std::abs(r1 - r0) == std::abs(c1 - c0)) {
        tables.can_move[kBishop][from][to] = true;
      }
      if (r1 == r0 || c1 == c0) {
        tables.can_move[kRook][from][to] = true;
      }
      if (r1 == r0 || c1 == c0 || std::abs(r1 - r0) == std::abs(c1 - c0)) {
        tables.can_move[kQueen][from][to] = true;
      }
      if (std::abs(r1 - r0) <= 1 && std::abs(c1 - c0) <= 1) {
        tables.can_move[kKing][from][to] = true;
      }
    }
  }

  for (int from = 0; from < kSquares; ++from) {
    for (int to = 0; to < kSquares; ++to) {
      const int r0 = Row(from);
      const int c0 = Col(from);
      const int r1 = Row(to);
      const int c1 = Col(to);
      if (!(r1 == r0 || c1 == c0 || std::abs(r1 - r0) == std::abs(c1 - c0))) {
        continue;
      }
      const int step_r = std::max(-1, std::min(1, r1 - r0));
      const int step_c = std::max(-1, std::min(1, c1 - c0));
      int row = r0;
      int col = c0;
      int out = 0;
      while (true) {
        row += step_r;
        col += step_c;
        if (row == r1 && col == c1) {
          break;
        }
        if (out < 6) {
          tables.between[from][to][out++] = Pos(row, col);
        }
      }
    }
  }
  return tables;
}

inline const Tables& GetTables() {
  static const Tables* tables = new Tables(MakeTables());
  return *tables;
}

inline Action FromLabel(int label) {
  const int from = label / kPlanes;
  const int plane = label % kPlanes;
  return {from, GetTables().from_plane[from][plane],
          plane >= 9 ? -1 : plane / 3};
}

inline int ToLabel(const Action& action) {
  return action.from * kPlanes + GetTables().to_plane[action.from][action.to];
}

inline Board FlipBoard(const Board& board) {
  Board flipped{};
  for (int pos = 0; pos < kSquares; ++pos) {
    flipped[FlipPos(pos)] = -board[pos];
  }
  return flipped;
}

inline BoardHistory FlipHistory(const BoardHistory& history) {
  BoardHistory out{};
  for (int i = 0; i < kHistory; ++i) {
    out[i] = FlipBoard(history[i]);
  }
  return out;
}

inline uint64_t BoardKey(const Board& board, int color,
                         const CastlingRights& castling_rights,
                         int en_passant) {
  uint64_t key = static_cast<uint64_t>(color + 1) * 131U +
                 static_cast<uint64_t>(en_passant + 2);
  for (const auto& row : castling_rights) {
    for (bool right : row) {
      key = key * 131U + static_cast<uint64_t>(right);
    }
  }
  for (int value : board) {
    key = key * 1315423911ULL + static_cast<uint64_t>(value + 7);
  }
  return key;
}

}  // namespace chess

class ChessEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("chess")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<float>({-1, 8, 8, 119})),
                    "info:board"_.Bind(Spec<int>({8, 8})),
                    "info:castling_rights"_.Bind(Spec<bool>({2, 2})),
                    "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
                    "info:en_passant"_.Bind(Spec<int>({}, {-1, 63})),
                    "info:fullmove_count"_.Bind(Spec<int>({})),
                    "info:halfmove_count"_.Bind(Spec<int>({})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({4672})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})),
                    "info:turn"_.Bind(Spec<int>({}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 4671})));
  }
};

using ChessEnvSpec = EnvSpec<ChessEnvFns>;

class ChessEnv : public Env<ChessEnvSpec>, public RenderableEnv {
 public:
  using Spec = ChessEnvSpec;
  using Action = typename Env<ChessEnvSpec>::Action;

  ChessEnv(const Spec& spec, int env_id) : Env<ChessEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    const int first_player = static_cast<int>(gen_() & 1U);
    player_order_ = {first_player, 1 - first_player};
    color_ = 0;
    current_player_ = player_order_[color_];
    board_ = chess::kInitBoard;
    castling_rights_ = {{{true, true}, {true, true}}};
    en_passant_ = -1;
    board_history_.fill({});
    board_history_[0] = board_;
    seen_keys_.fill(0);
    seen_keys_[0] =
        chess::BoardKey(board_, color_, castling_rights_, en_passant_);
    halfmove_count_ = 0;
    fullmove_count_ = 1;
    step_count_ = 0;
    rewards_ = {0.0f, 0.0f};
    done_ = false;
    UpdateLegalActionMask();
    WriteState(rewards_);
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal = act < 0 || act >= chess::kActions ||
                         (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act < chess::kActions) {
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
    board_games::DrawGrid(rgb, width, height, 8, 8);
    for (int pos = 0; pos < chess::kSquares; ++pos) {
      const int piece = board_[pos];
      if (piece == 0) {
        continue;
      }
      const int row = 7 - chess::Row(pos);
      const int col = chess::Col(pos);
      const int cx = col * width / 8 + width / 16;
      const int cy = row * height / 8 + height / 16;
      board_games::DrawCircle(rgb, width, height, cx, cy,
                              std::min(width, height) / 24,
                              piece > 0 ? board_games::Rgb{35, 35, 35}
                                        : board_games::Rgb{235, 235, 225});
    }
  }

 private:
  chess::Board board_{};
  chess::BoardHistory board_history_{};
  chess::CastlingRights castling_rights_{};
  std::array<uint64_t, chess::kMaxTerminationSteps + 1> seen_keys_{};
  std::array<bool, chess::kActions> legal_action_mask_{};
  std::array<float, 2> rewards_{};
  std::array<int, 2> player_order_{0, 1};
  int color_{0};
  int current_player_{0};
  int en_passant_{-1};
  int halfmove_count_{0};
  int fullmove_count_{1};
  int step_count_{0};
  bool done_{true};

  static bool IsSlidingPiece(int piece) {
    piece = std::abs(piece);
    return piece == chess::kBishop || piece == chess::kRook ||
           piece == chess::kQueen;
  }

  bool PathClear(const chess::Board& board, int from, int to) const {
    if (!IsSlidingPiece(board[from]) && std::abs(board[from]) != chess::kPawn) {
      return true;
    }
    for (int between : chess::GetTables().between[from][to]) {
      if (between >= 0 && board[between] != chess::kEmpty) {
        return false;
      }
    }
    return true;
  }

  bool IsPseudoLegal(const chess::Board& board,
                     const chess::Action& action) const {
    if (action.from < 0 || action.from >= chess::kSquares || action.to < 0 ||
        action.to >= chess::kSquares) {
      return false;
    }
    const int piece = board[action.from];
    if (piece <= 0 || board[action.to] > 0) {
      return false;
    }
    if (!chess::GetTables().can_move[piece][action.from][action.to]) {
      return false;
    }
    if (!PathClear(board, action.from, action.to)) {
      return false;
    }
    if (piece == chess::kPawn) {
      const bool same_file = chess::Col(action.to) == chess::Col(action.from);
      const bool pawn_should =
          (same_file && board[action.to] == chess::kEmpty) ||
          (!same_file && board[action.to] < 0);
      if (!pawn_should) {
        return false;
      }
    }
    return true;
  }

  bool IsAttacked(const chess::Board& board, int pos) const {
    for (int to = 0; to < chess::kSquares; ++to) {
      if (board[to] >= 0) {
        continue;
      }
      const int piece = std::abs(board[to]);
      if (!chess::GetTables().can_move[piece][pos][to]) {
        continue;
      }
      if (IsSlidingPiece(piece)) {
        bool clear = true;
        for (int between : chess::GetTables().between[pos][to]) {
          if (between >= 0 && board[between] != chess::kEmpty) {
            clear = false;
            break;
          }
        }
        if (!clear) {
          continue;
        }
      }
      if (piece == chess::kPawn && chess::Col(to) == chess::Col(pos)) {
        continue;
      }
      return true;
    }
    return false;
  }

  bool IsChecked(const chess::Board& board) const {
    int king = -1;
    for (int i = 0; i < chess::kSquares; ++i) {
      if (board[i] == chess::kKing) {
        king = i;
        break;
      }
    }
    return king >= 0 && IsAttacked(board, king);
  }

  chess::Board ApplyMove(chess::Board board, const chess::Action& action,
                         chess::CastlingRights* castling_rights,
                         int* en_passant, int* halfmove_count,
                         int* fullmove_count) const {
    int piece = board[action.from];
    const bool capture_en_passant =
        *en_passant >= 0 && piece == chess::kPawn && *en_passant == action.to;
    if (capture_en_passant) {
      board[action.to - 1] = chess::kEmpty;
    }
    const bool double_pawn_move =
        piece == chess::kPawn && std::abs(action.to - action.from) == 2;
    *en_passant = double_pawn_move ? (action.to + action.from) / 2 : -1;
    const bool captured = board[action.to] < 0 || double_pawn_move;
    *halfmove_count =
        captured || piece == chess::kPawn ? 0 : *halfmove_count + 1;
    *fullmove_count += color_ == 1 ? 1 : 0;

    if (piece == chess::kKing && action.from == 32 && action.to == 16) {
      board[0] = chess::kEmpty;
      board[24] = chess::kRook;
    }
    if (piece == chess::kKing && action.from == 32 && action.to == 48) {
      board[56] = chess::kEmpty;
      board[40] = chess::kRook;
    }

    (*castling_rights)[0][0] =
        (*castling_rights)[0][0] && action.from != 32 && action.from != 0;
    (*castling_rights)[0][1] =
        (*castling_rights)[0][1] && action.from != 32 && action.from != 56;
    (*castling_rights)[1][0] = (*castling_rights)[1][0] && action.to != 7;
    (*castling_rights)[1][1] = (*castling_rights)[1][1] && action.to != 63;

    if (piece == chess::kPawn && chess::Row(action.from) == 6 &&
        action.underpromotion < 0) {
      piece = chess::kQueen;
    }
    if (action.underpromotion >= 0) {
      static constexpr std::array<int, 3> kUnderpromotions = {
          chess::kRook,
          chess::kBishop,
          chess::kKnight,
      };
      piece = kUnderpromotions[action.underpromotion];
    }
    board[action.from] = chess::kEmpty;
    board[action.to] = piece;
    return board;
  }

  chess::Board ApplyMoveNoCounters(chess::Board board,
                                   const chess::Action& action) const {
    chess::CastlingRights rights = castling_rights_;
    int en_passant = en_passant_;
    int halfmove = halfmove_count_;
    int fullmove = fullmove_count_;
    return ApplyMove(board, action, &rights, &en_passant, &halfmove, &fullmove);
  }

  bool IsNormalMoveLegal(const chess::Action& action) const {
    if (!IsPseudoLegal(board_, action)) {
      return false;
    }
    return !IsChecked(ApplyMoveNoCounters(board_, action));
  }

  bool IsEnPassantLegal(const chess::Action& action) const {
    if (en_passant_ < 0 || action.to != en_passant_ || action.from < 0 ||
        action.from >= chess::kSquares || board_[action.from] != chess::kPawn ||
        board_[action.to - 1] != -chess::kPawn) {
      return false;
    }
    return !IsChecked(ApplyMoveNoCounters(board_, action));
  }

  bool CastlePathSafe(bool queen_side) const {
    const std::array<int, 3> queen_checks = {16, 24, 32};
    const std::array<int, 3> king_checks = {32, 40, 48};
    const auto& checks = queen_side ? queen_checks : king_checks;
    for (int pos : checks) {
      if (IsAttacked(board_, pos)) {
        return false;
      }
    }
    return true;
  }

  void UpdateLegalActionMask() {
    std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), false);
    for (int from = 0; from < chess::kSquares; ++from) {
      if (board_[from] <= 0) {
        continue;
      }
      for (int to = 0; to < chess::kSquares; ++to) {
        chess::Action action{from, to, -1};
        if (IsNormalMoveLegal(action)) {
          const int label = chess::ToLabel(action);
          if (label >= 0) {
            legal_action_mask_[label] = true;
          }
        }
      }
    }
    if (en_passant_ >= 0) {
      for (int from : {en_passant_ - 9, en_passant_ + 7}) {
        chess::Action action{from, en_passant_, -1};
        if (IsEnPassantLegal(action)) {
          const int label = chess::ToLabel(action);
          if (label >= 0) {
            legal_action_mask_[label] = true;
          }
        }
      }
    }
    const bool can_castle_queen =
        castling_rights_[0][0] && board_[0] == chess::kRook &&
        board_[8] == chess::kEmpty && board_[16] == chess::kEmpty &&
        board_[24] == chess::kEmpty && board_[32] == chess::kKing &&
        CastlePathSafe(true);
    const bool can_castle_king =
        castling_rights_[0][1] && board_[32] == chess::kKing &&
        board_[40] == chess::kEmpty && board_[48] == chess::kEmpty &&
        board_[56] == chess::kRook && CastlePathSafe(false);
    legal_action_mask_[2364] = legal_action_mask_[2364] || can_castle_queen;
    legal_action_mask_[2367] = legal_action_mask_[2367] || can_castle_king;

    for (int from : {6, 14, 22, 30, 38, 46, 54, 62}) {
      for (int plane = 0; plane < 9; ++plane) {
        const int label = from * chess::kPlanes + plane;
        const chess::Action action = chess::FromLabel(label);
        const int normal_label = chess::ToLabel({action.from, action.to, -1});
        if (board_[action.from] == chess::kPawn && action.to >= 0 &&
            normal_label >= 0 && legal_action_mask_[normal_label]) {
          legal_action_mask_[label] = true;
        }
      }
    }
  }

  bool HasInsufficientPieces() const {
    int pieces = 0;
    int pawn_rook_queen = 0;
    int bishops = 0;
    int bishops_on_black = 0;
    for (int pos = 0; pos < chess::kSquares; ++pos) {
      const int piece = std::abs(board_[pos]);
      if (piece == 0) {
        continue;
      }
      ++pieces;
      if (piece >= chess::kRook || piece == chess::kPawn) {
        ++pawn_rook_queen;
      }
      if (piece == chess::kBishop) {
        ++bishops;
        const int row = chess::Row(pos);
        const int col = chess::Col(pos);
        if ((row % 2 == 0 && col % 2 == 0) || (row % 2 == 1 && col % 2 == 1)) {
          ++bishops_on_black;
        }
      }
    }
    pawn_rook_queen -= 2;
    return pieces <= 2 || (pieces == 3 && pawn_rook_queen == 0) ||
           (pieces == bishops + 2 &&
            (bishops_on_black == bishops || bishops_on_black == 0));
  }

  int RepetitionCount(uint64_t key) const {
    return static_cast<int>(
               std::count(seen_keys_.begin(), seen_keys_.end(), key)) -
           1;
  }

  void FlipState() {
    board_ = chess::FlipBoard(board_);
    color_ = 1 - color_;
    en_passant_ = chess::FlipPos(en_passant_);
    std::swap(castling_rights_[0], castling_rights_[1]);
    board_history_ = chess::FlipHistory(board_history_);
  }

  void UpdateHistory() {
    for (int i = chess::kHistory - 1; i > 0; --i) {
      board_history_[i] = board_history_[i - 1];
    }
    board_history_[0] = board_;
    for (int i = chess::kMaxTerminationSteps; i > 0; --i) {
      seen_keys_[i] = seen_keys_[i - 1];
    }
    seen_keys_[0] =
        chess::BoardKey(board_, color_, castling_rights_, en_passant_);
  }

  void StepGame(int label) {
    const chess::Action action = chess::FromLabel(label);
    board_ = ApplyMove(board_, action, &castling_rights_, &en_passant_,
                       &halfmove_count_, &fullmove_count_);
    FlipState();
    current_player_ = player_order_[color_];
    ++step_count_;
    UpdateHistory();
    UpdateLegalActionMask();

    const bool has_legal =
        std::any_of(legal_action_mask_.begin(), legal_action_mask_.end(),
                    [](bool value) { return value; });
    done_ = !has_legal || halfmove_count_ >= 100 || HasInsufficientPieces() ||
            RepetitionCount(seen_keys_[0]) >= 2 ||
            step_count_ >= chess::kMaxTerminationSteps;
    const bool is_checkmate = !has_legal && IsChecked(board_);
    rewards_ = {0.0f, 0.0f};
    if (is_checkmate) {
      std::array<float, 2> color_rewards{1.0f, 1.0f};
      color_rewards[color_] = -1.0f;
      rewards_[player_order_[0]] = color_rewards[0];
      rewards_[player_order_[1]] = color_rewards[1];
    }
  }

  void WriteObservationForPlayer(State* state, int player) const {
    const bool current_view = current_player_ == player;
    chess::BoardHistory history =
        current_view ? board_history_ : chess::FlipHistory(board_history_);
    chess::CastlingRights rights = castling_rights_;
    if (!current_view) {
      std::swap(rights[0], rights[1]);
    }
    const int color = current_view ? color_ : 1 - color_;
    int channel = 0;
    for (int h = 0; h < chess::kHistory; ++h) {
      for (int piece = 1; piece <= 6; ++piece, ++channel) {
        for (int row = 0; row < 8; ++row) {
          for (int col = 0; col < 8; ++col) {
            const int pos = col * 8 + (7 - row);
            (*state)["obs"_](player, row, col, channel) =
                history[h][pos] == piece ? 1.0f : 0.0f;
          }
        }
      }
      for (int piece = 1; piece <= 6; ++piece, ++channel) {
        for (int row = 0; row < 8; ++row) {
          for (int col = 0; col < 8; ++col) {
            const int pos = col * 8 + (7 - row);
            (*state)["obs"_](player, row, col, channel) =
                history[h][pos] == -piece ? 1.0f : 0.0f;
          }
        }
      }
      for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
          (*state)["obs"_](player, row, col, channel) = 1.0f;
          (*state)["obs"_](player, row, col, channel + 1) = 0.0f;
        }
      }
      channel += 2;
    }
    for (int row = 0; row < 8; ++row) {
      for (int col = 0; col < 8; ++col) {
        (*state)["obs"_](player, row, col, channel) = static_cast<float>(color);
        (*state)["obs"_](player, row, col, channel + 1) =
            static_cast<float>(step_count_) /
            static_cast<float>(chess::kMaxTerminationSteps);
        for (int i = 0; i < 4; ++i) {
          (*state)["obs"_](player, row, col, channel + 2 + i) =
              rights[i / 2][i % 2] ? 1.0f : 0.0f;
        }
        (*state)["obs"_](player, row, col, channel + 6) =
            static_cast<float>(halfmove_count_) / 100.0f;
      }
    }
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    for (int player = 0; player < 2; ++player) {
      WriteObservationForPlayer(&state, player);
      state["info:players.id"_][player] = player;
    }
    for (int row = 0; row < 8; ++row) {
      for (int col = 0; col < 8; ++col) {
        state["info:board"_](row, col) = board_[col * 8 + (7 - row)];
      }
    }
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 2; ++col) {
        state["info:castling_rights"_](row, col) = castling_rights_[row][col];
      }
    }
    for (int i = 0; i < chess::kActions; ++i) {
      state["info:legal_action_mask"_][i] = legal_action_mask_[i];
    }
    state["info:current_player"_] = current_player_;
    state["info:en_passant"_] = en_passant_;
    state["info:fullmove_count"_] = fullmove_count_;
    state["info:halfmove_count"_] = halfmove_count_;
    state["info:turn"_] = color_;
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
    }
  }
};

using ChessEnvPool = AsyncEnvPool<ChessEnv>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_CHESS_GAMES_H_
