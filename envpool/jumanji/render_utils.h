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

#ifndef ENVPOOL_JUMANJI_RENDER_UTILS_H_
#define ENVPOOL_JUMANJI_RENDER_UTILS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace jumanji::render {

struct Color {
  unsigned char r;
  unsigned char g;
  unsigned char b;
};

[[maybe_unused]] constexpr Color kWhite{255, 255, 255};
[[maybe_unused]] constexpr Color kPanel{247, 247, 247};
[[maybe_unused]] constexpr Color kGrid{192, 192, 192};
[[maybe_unused]] constexpr Color kDarkGrid{70, 70, 70};
[[maybe_unused]] constexpr Color kBlack{0, 0, 0};

inline std::size_t PixelOffset(int width, int x, int y) {
  return (static_cast<std::size_t>(y) * width + x) * 3;
}

inline void PutPixel(int width, int height, int x, int y, Color color,
                     unsigned char* rgb) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }
  const std::size_t index = PixelOffset(width, x, y);
  rgb[index] = color.r;
  rgb[index + 1] = color.g;
  rgb[index + 2] = color.b;
}

inline void Clear(int width, int height, Color color, unsigned char* rgb) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      PutPixel(width, height, x, y, color, rgb);
    }
  }
}

inline void FillRect(int width, int height, int left, int top, int right,
                     int bottom, Color color, unsigned char* rgb) {
  left = std::clamp(left, 0, width);
  right = std::clamp(right, 0, width);
  top = std::clamp(top, 0, height);
  bottom = std::clamp(bottom, 0, height);
  for (int y = top; y < bottom; ++y) {
    for (int x = left; x < right; ++x) {
      PutPixel(width, height, x, y, color, rgb);
    }
  }
}

inline void StrokeRect(int width, int height, int left, int top, int right,
                       int bottom, Color color, unsigned char* rgb,
                       int thickness = 1) {
  for (int i = 0; i < thickness; ++i) {
    for (int x = left + i; x < right - i; ++x) {
      PutPixel(width, height, x, top + i, color, rgb);
      PutPixel(width, height, x, bottom - 1 - i, color, rgb);
    }
    for (int y = top + i; y < bottom - i; ++y) {
      PutPixel(width, height, left + i, y, color, rgb);
      PutPixel(width, height, right - 1 - i, y, color, rgb);
    }
  }
}

inline void DrawLine(int width, int height, int x0, int y0, int x1, int y1,
                     Color color, unsigned char* rgb, int thickness = 1) {
  const int dx = std::abs(x1 - x0);
  const int dy = -std::abs(y1 - y0);
  const int sx = x0 < x1 ? 1 : -1;
  const int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  while (true) {
    const int radius = thickness / 2;
    FillRect(width, height, x0 - radius, y0 - radius, x0 + radius + 1,
             y0 + radius + 1, color, rgb);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

inline void FillCircle(int width, int height, int cx, int cy, int radius,
                       Color color, unsigned char* rgb) {
  const int r2 = radius * radius;
  for (int y = cy - radius; y <= cy + radius; ++y) {
    for (int x = cx - radius; x <= cx + radius; ++x) {
      const int dx = x - cx;
      const int dy = y - cy;
      if (dx * dx + dy * dy <= r2) {
        PutPixel(width, height, x, y, color, rgb);
      }
    }
  }
}

inline void StrokeCircle(int width, int height, int cx, int cy, int radius,
                         Color color, unsigned char* rgb) {
  const int outer = radius * radius;
  const int inner = std::max(0, radius - 2) * std::max(0, radius - 2);
  for (int y = cy - radius; y <= cy + radius; ++y) {
    for (int x = cx - radius; x <= cx + radius; ++x) {
      const int dx = x - cx;
      const int dy = y - cy;
      const int d2 = dx * dx + dy * dy;
      if (inner <= d2 && d2 <= outer) {
        PutPixel(width, height, x, y, color, rgb);
      }
    }
  }
}

inline void FillCell(int width, int height, int rows, int cols, int row,
                     int col, Color color, unsigned char* rgb, int pad = 0) {
  const int left = col * width / cols;
  const int right = (col + 1) * width / cols;
  const int top = row * height / rows;
  const int bottom = (row + 1) * height / rows;
  FillRect(width, height, left + pad, top + pad, right - pad, bottom - pad,
           color, rgb);
}

inline void DrawGrid(int width, int height, int rows, int cols, Color color,
                     unsigned char* rgb, int thickness = 1) {
  for (int row = 0; row <= rows; ++row) {
    const int y = row * height / rows;
    FillRect(width, height, 0, y - thickness / 2, width,
             y + (thickness + 1) / 2, color, rgb);
  }
  for (int col = 0; col <= cols; ++col) {
    const int x = col * width / cols;
    FillRect(width, height, x - thickness / 2, 0, x + (thickness + 1) / 2,
             height, color, rgb);
  }
}

inline std::pair<int, int> CellCenter(int width, int height, int rows, int cols,
                                      int row, int col) {
  const int x0 = col * width / cols;
  const int x1 = (col + 1) * width / cols;
  const int y0 = row * height / rows;
  const int y1 = (row + 1) * height / rows;
  return {(x0 + x1) / 2, (y0 + y1) / 2};
}

inline Color Palette(int index) {
  constexpr std::array<Color, 20> colors = {{
      {31, 119, 180},  {255, 127, 14},  {44, 160, 44},   {214, 39, 40},
      {148, 103, 189}, {140, 86, 75},   {227, 119, 194}, {127, 127, 127},
      {188, 189, 34},  {23, 190, 207},  {174, 199, 232}, {255, 187, 120},
      {152, 223, 138}, {255, 152, 150}, {197, 176, 213}, {196, 156, 148},
      {247, 182, 210}, {199, 199, 199}, {219, 219, 141}, {158, 218, 229},
  }};
  return colors[static_cast<std::size_t>(
      (index % colors.size() + colors.size()) % colors.size())];
}

inline Color Blend(Color a, Color b, float t) {
  t = std::clamp(t, 0.0f, 1.0f);
  return {
      static_cast<unsigned char>(std::lround(a.r * (1.0f - t) + b.r * t)),
      static_cast<unsigned char>(std::lround(a.g * (1.0f - t) + b.g * t)),
      static_cast<unsigned char>(std::lround(a.b * (1.0f - t) + b.b * t)),
  };
}

inline void DrawSegment(int width, int height, int left, int top, int right,
                        int bottom, Color color, unsigned char* rgb) {
  FillRect(width, height, left, top, right, bottom, color, rgb);
}

inline void DrawDigit(int width, int height, int digit, int left, int top,
                      int right, int bottom, Color color, unsigned char* rgb) {
  if (digit < 0 || digit > 9 || right - left < 4 || bottom - top < 6) {
    return;
  }
  constexpr std::array<std::uint8_t, 10> segments = {
      0b1111110, 0b0110000, 0b1101101, 0b1111001, 0b0110011,
      0b1011011, 0b1011111, 0b1110000, 0b1111111, 0b1111011,
  };
  const int w = right - left;
  const int h = bottom - top;
  const int t = std::max(1, std::min(w, h) / 7);
  const int mid = top + h / 2;
  const std::uint8_t s = segments[static_cast<std::size_t>(digit)];
  if ((s & 0b1000000) != 0) {
    DrawSegment(width, height, left + t, top, right - t, top + t, color, rgb);
  }
  if ((s & 0b0100000) != 0) {
    DrawSegment(width, height, right - t, top + t, right, mid, color, rgb);
  }
  if ((s & 0b0010000) != 0) {
    DrawSegment(width, height, right - t, mid, right, bottom - t, color, rgb);
  }
  if ((s & 0b0001000) != 0) {
    DrawSegment(width, height, left + t, bottom - t, right - t, bottom, color,
                rgb);
  }
  if ((s & 0b0000100) != 0) {
    DrawSegment(width, height, left, mid, left + t, bottom - t, color, rgb);
  }
  if ((s & 0b0000010) != 0) {
    DrawSegment(width, height, left, top + t, left + t, mid, color, rgb);
  }
  if ((s & 0b0000001) != 0) {
    DrawSegment(width, height, left + t, mid - t / 2, right - t,
                mid + (t + 1) / 2, color, rgb);
  }
}

inline void DrawNumber(int width, int height, int value, int left, int top,
                       int right, int bottom, Color color, unsigned char* rgb) {
  if (value < 0) {
    return;
  }
  if (value < 10) {
    DrawDigit(width, height, value, left, top, right, bottom, color, rgb);
    return;
  }
  const int mid = (left + right) / 2;
  DrawDigit(width, height, value / 10, left, top, mid - 1, bottom, color, rgb);
  DrawDigit(width, height, value % 10, mid + 1, top, right, bottom, color, rgb);
}

}  // namespace jumanji::render

#endif  // ENVPOOL_JUMANJI_RENDER_UTILS_H_
