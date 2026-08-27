// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_CRAFTAX_RENDERER_H_
#define ENVPOOL_CRAFTAX_RENDERER_H_

#include <cstddef>
#include <vector>

#include "envpool/craftax/game.h"

namespace craftax {

struct EncodedTexture {
  const char* name;
  const unsigned char* bytes;
  std::size_t size;
};
extern const EncodedTexture* const kTextures;
extern const std::size_t kTextureCount;

// Float RGB on the official 0..255 scale. Pixel observations divide by 255;
// public RGB rendering converts to uint8 only after the final composite.
std::vector<float> Pixels(const Game& game, int tile_size);

}  // namespace craftax

#endif  // ENVPOOL_CRAFTAX_RENDERER_H_
