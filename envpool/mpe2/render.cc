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

#include <ft2build.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/mpe2/model.h"
#include FT_FREETYPE_H
#include "third_party/mpe2/font.h"

namespace mpe2 {
namespace {
struct Glyph {
  int left, top, width, height, advance;
  std::vector<unsigned char> alpha;
};

class Font {
 public:
  std::array<Glyph, 128> glyphs;
  Font() {
    FT_Library library;
    if (FT_Init_FreeType(&library) != 0) {
      throw std::runtime_error("Cannot initialize MPE2 font renderer");
    }
    FT_Face face;
    if (FT_New_Memory_Face(library, kFont, sizeof(kFont), 0, &face) != 0) {
      FT_Done_FreeType(library);
      throw std::runtime_error("Cannot load MPE2 font");
    }
    FT_Set_Char_Size(face, 0, 24 * 64, 72, 72);
    for (int i = 32; i < 127; ++i) {
      FT_Load_Char(face, i, FT_LOAD_RENDER | FT_LOAD_FORCE_AUTOHINT);
      auto& bitmap = face->glyph->bitmap;
      auto& g = glyphs[i];
      g = {face->glyph->bitmap_left,
           face->glyph->bitmap_top,
           static_cast<int>(bitmap.width),
           static_cast<int>(bitmap.rows),
           static_cast<int>(face->glyph->advance.x >> 6),
           {}};
      g.alpha.resize(g.width * g.height);
      for (int y = 0; y < g.height; ++y) {
        std::copy_n(bitmap.buffer + y * bitmap.pitch, g.width,
                    g.alpha.data() + y * g.width);
      }
    }
    FT_Done_Face(face);
    FT_Done_FreeType(library);
  }
};

std::vector<std::pair<int, int>> CircleExtents(int radius) {
  std::vector<std::pair<int, int>> spans(2 * radius, {0, radius});
  int x = 0;
  int y = radius;
  int error = 1 - radius;
  auto update = [&](int row, int half) {
    if (row >= 0 && row < 2 * radius) {
      // Nearest/furthest boundary pixels define the one-pixel outline. A
      // smaller filled circle has different rounding at shallow slopes.
      spans[row].first = std::max(spans[row].first, half);
      spans[row].second = std::min(spans[row].second, std::max(0, half - 1));
    }
  };
  while (x <= y) {
    update(radius - y, x);
    update(radius + y - 1, x);
    update(radius - x, y);
    update(radius + x - 1, y);
    ++x;
    if (error < 0) {
      error += 2 * x + 1;
    } else {
      --y;
      error += 2 * (x - y) + 1;
    }
  }
  return spans;
}

void Circle(unsigned char* rgb, int width, int height, int cx, int cy,
            int radius, const Color& color) {
  if (radius <= 0) {
    return;
  }
  auto spans = CircleExtents(radius);
  for (int row = 0; row < 2 * radius; ++row) {
    int y = cy - radius + row;
    if (y < 0 || y >= height) {
      continue;
    }
    auto [outside, inside] = spans[row];
    for (int x = std::max(0, cx - outside); x < std::min(width, cx + outside);
         ++x) {
      bool border = x < cx - inside || x >= cx + inside;
      for (int c = 0; c < 3; ++c) {
        rgb[(y * width + x) * 3 + c] =
            border ? 0 : static_cast<unsigned char>(color[c] * 200);
      }
    }
  }
}

void Text(unsigned char* rgb, int width, int height, int x, int y,
          const std::string& text) {
  static const Font font;
  int top = 0;
  int left = font.glyphs[static_cast<unsigned char>(text[0])].left;
  for (unsigned char c : text) {
    top = std::max(top, font.glyphs[c].top);
  }
  x -= left;
  for (unsigned char c : text) {
    const auto& g = font.glyphs[c];
    for (int gy = 0; gy < g.height; ++gy) {
      for (int gx = 0; gx < g.width; ++gx) {
        int px = x + g.left + gx;
        int py = y + top - g.top + gy;
        if (px < 0 || py < 0 || px >= width || py >= height) {
          continue;
        }
        int alpha = g.alpha[gy * g.width + gx];
        for (int channel = 0; channel < 3; ++channel) {
          auto& value = rgb[(py * width + px) * 3 + channel];
          value =
              static_cast<unsigned char>((value * (255 - alpha) + 127) / 255);
        }
      }
    }
    x += g.advance;
  }
}
}  // namespace

void Render(const Model& model, int width, int height, std::uint8_t* rgb) {
  std::fill_n(rgb, width * height * 3, 255);
  double range = 0;
  for (const auto& e : model.entities) {
    if (e.alive) {
      for (double x : e.pos) {
        range = std::max(range, std::abs(x));
      }
    }
  }
  if (range == 0) {
    range = 1;
  }
  double scale = 0.9 * model.original_range / range;
  const int center_x = width / 2;
  const int center_y = height / 2;
  int line = 0;
  for (const auto& e : model.entities) {
    if (!e.alive) {
      continue;
    }
    int x = static_cast<int>(std::floor(e.pos[0] / range * width / 2) * 0.9 +
                             center_x);
    int y = static_cast<int>(std::floor(-e.pos[1] / range * height / 2) * 0.9 +
                             center_y);
    int radius =
        static_cast<int>(e.size * 350 * (model.kOptions.rescaling ? scale : 1));
    Circle(rgb, width, height, x, y, radius, e.color);
    if (e.silent) {
      continue;
    }
    std::string message = e.name + " sends ";
    if (std::all_of(e.comm.begin(), e.comm.end(),
                    [](double c) { return c == 0; })) {
      message += '_';
    } else if (model.kOptions.continuous) {
      std::ostringstream stream;
      stream << '[' << std::fixed << std::setprecision(2);
      for (std::size_t i = 0; i < e.comm.size(); ++i) {
        if (i != 0u) {
          stream << ',';
        }
        stream << e.comm[i];
      }
      stream << ']';
      message += stream.str();
    } else {
      message += static_cast<char>(
          'A' + std::distance(e.comm.begin(),
                              std::max_element(e.comm.begin(), e.comm.end())));
    }
    message += "   ";
    Text(rgb, width, height, static_cast<int>(width * 0.05),
         static_cast<int>(height * 0.95 - height * 0.05 * line), message);
    ++line;
  }
}
}  // namespace mpe2
