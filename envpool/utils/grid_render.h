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

#ifndef ENVPOOL_UTILS_GRID_RENDER_H_
#define ENVPOOL_UTILS_GRID_RENDER_H_

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "opencv2/imgproc.hpp"

namespace rendering {

// Cells are numbered in row-major order, with the origin at the top left.
inline void RenderBoard(int rows, int cols, std::string_view walls,
                        std::initializer_list<std::pair<int, char>> markers,
                        int width, int height, unsigned char* rgb) {
  cv::Mat frame(height, width, CV_8UC3, rgb);
  frame.setTo(cv::Scalar::all(255));
  for (int i = 0; i < static_cast<int>(walls.size()); ++i) {
    if (walls[i] != '#') {
      continue;
    }
    const int col = i % cols;
    const int row = i / cols;
    cv::rectangle(frame,
                  cv::Rect(cv::Point(col * width / cols, row * height / rows),
                           cv::Point((col + 1) * width / cols,
                                     (row + 1) * height / rows)),
                  cv::Scalar::all(0), cv::FILLED);
  }
  const int cell = std::min(width / cols, height / rows);
  for (auto [i, ch] : markers) {
    const int x = (2 * (i % cols) + 1) * width / (2 * cols);
    const int y = (2 * (i / cols) + 1) * height / (2 * rows);
    cv::putText(frame, std::string(1, ch), {x - cell / 4, y + cell / 4},
                cv::FONT_HERSHEY_SIMPLEX, cell / 42.0, cv::Scalar::all(0),
                std::max(1, cell / 18), cv::LINE_AA);
  }
}

}  // namespace rendering

#endif  // ENVPOOL_UTILS_GRID_RENDER_H_
