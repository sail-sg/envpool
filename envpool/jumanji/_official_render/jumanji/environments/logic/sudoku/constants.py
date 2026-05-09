# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

BOARD_WIDTH = 9

# BOX_IDX is a 9x9 array of indices of the 9 3x3 boxes in a sudoku board.
# The indices are used to extract the 3x3 boxes from the board.
BOX_IDX = np.array(
    [
        [0, 1, 2, 9, 10, 11, 18, 19, 20],
        [27, 28, 29, 36, 37, 38, 45, 46, 47],
        [54, 55, 56, 63, 64, 65, 72, 73, 74],
        [3, 4, 5, 12, 13, 14, 21, 22, 23],
        [30, 31, 32, 39, 40, 41, 48, 49, 50],
        [57, 58, 59, 66, 67, 68, 75, 76, 77],
        [6, 7, 8, 15, 16, 17, 24, 25, 26],
        [33, 34, 35, 42, 43, 44, 51, 52, 53],
        [60, 61, 62, 69, 70, 71, 78, 79, 80],
    ]
)


# Sample board with solution for debugging and test purposes
INITIAL_BOARD_SAMPLE = np.array(
    [
        [0, 0, 0, 8, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 3],
        [5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 2, 0, 0, 3, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 7, 5],
        [0, 0, 3, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 6, 0, 0],
    ]
)

SOLVED_BOARD_SAMPLE = np.array(
    [
        [2, 3, 7, 8, 4, 1, 5, 6, 9],
        [1, 8, 6, 7, 9, 5, 2, 4, 3],
        [5, 9, 4, 3, 2, 6, 7, 1, 8],
        [3, 1, 5, 6, 7, 4, 8, 9, 2],
        [4, 6, 9, 5, 8, 2, 1, 3, 7],
        [7, 2, 8, 1, 3, 9, 4, 5, 6],
        [6, 4, 2, 9, 1, 8, 3, 7, 5],
        [8, 5, 3, 4, 6, 7, 9, 2, 1],
        [9, 7, 1, 2, 5, 3, 6, 8, 4],
    ]
)
