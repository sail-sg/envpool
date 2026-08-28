/*
 * Copyright 2026 Garena Online Private Limited
 * Copyright 2019-2021 The dm_control Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "envpool/mujoco/locomotion/scene.h"
#include "third_party/dmc_locomotion/metadata.h"

namespace mujoco_locomotion {
namespace {

struct Rectangle {
  int x, y, width, height;
};

// The same greedy odd-sized covering as arenas/covering.py. This is also
// needed for floor textures: MuJoCo repeats textures per rectangle, not cell.
std::vector<Rectangle> Cover(const std::string& grid, char token, int size) {
  std::vector<bool> covered(grid.size(), false);
  std::vector<Rectangle> rectangles;
  auto index = [size](int x, int y) { return y * (size + 1) + x; };
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      if (grid[index(x, y)] != token || covered[index(x, y)]) {
        continue;
      }
      int width = size - x;
      Rectangle best{x, y, 1, 1};
      for (int end_y = y; end_y < size; ++end_y) {
        int end_x = x;
        while (end_x < x + width && grid[index(end_x, end_y)] == token &&
               !covered[index(end_x, end_y)]) {
          ++end_x;
        }
        width = end_x - x;
        if (width == 0) {
          break;
        }
        if (width % 2 == 0) {
          --width;
        }
        const int height = end_y - y + 1;
        if (((height % 2) != 0) && width * height > best.width * best.height) {
          best.width = width;
          best.height = height;
        }
      }
      for (int dy = 0; dy < best.height; ++dy) {
        for (int dx = 0; dx < best.width; ++dx) {
          covered[index(x + dx, y + dy)] = true;
        }
      }
      rectangles.push_back(best);
    }
  }
  return rectangles;
}

std::string FixedMaze(RandomState* random) {
  std::string grid;
  std::vector<int> empty;
  for (int y = 0; y < 7; ++y) {
    for (int x = 0; x < 7; ++x) {
      const bool wall = x == 0 || x == 6 || y == 0 || y == 6;
      if (!wall && (x != 3 || (y != 2 && y != 4))) {
        empty.push_back(grid.size());
      }
      grid += wall ? '*' : ' ';
    }
    grid += '\n';
  }
  grid[2 * 8 + 3] = 'P';
  grid[4 * 8 + 3] = 'G';
  // NumPy choice(..., replace=False) permutes the whole candidate array.
  random->Shuffle(&empty);
  for (int i = 0; i < 5; ++i) {
    grid[empty[i]] = 'G';
  }
  return grid;
}

}  // namespace

void Scene::MazeArena(const TaskConfig& task, RandomState* random) {
  const bool rodent = task.walker == Walker::kRodent;
  const bool fixed = task.task == Task::kHeterogeneous;
  const int size = fixed ? 7 : 11;
  const double scale = rodent ? .5 : 3;
  const double height = rodent ? .3 : 2;
  const double offset = (size - 1) / 2.0;
  if (fixed) {
    // The upstream constructor also draws the first layout before reset.
    if (!maze_initialized_) {
      FixedMaze(random);
    }
    maze_entities = FixedMaze(random);
    maze_variations.clear();
    for (int y = 0; y < size; ++y) {
      maze_variations += std::string(size, '.') + '\n';
    }
  } else {
    if (!maze_) {
      maze_ = std::make_unique<deepmind::labmaze::RandomMaze>(
          11, 11, 4, 4, 5, 1000, 0, 26, false, true, 1, "P", 3, "G",
          random->Int(2147483648U));
    }
    maze_->Regenerate();
    maze_entities = maze_->EntityLayer();
    maze_variations = maze_->VariationsLayer();
  }
  maze_initialized_ = true;
  Set(Child(Child(Root(), "default"), "geom"), "rgba", {1, 1, 1, 1});
  auto add_textures = [&](const std::string& prefix, const auto& textures,
                          const std::string& kind) {
    pugi::xml_document document;
    auto model = document.append_child("mujoco");
    for (auto name : textures) {
      auto texture = Child(model, "asset").append_child("texture");
      Set(texture, "name", std::string(name));
      Set(texture, "type", "2d");
      Set(texture, "file",
          labmaze_asset_path_ + "/style_01/" + kind + "_" + std::string(name) +
              "_d.png");
    }
    Attach(model, prefix, RootJoint::kFixed);
  };
  if (!rodent) {
    pugi::xml_document document;
    auto model = document.append_child("mujoco");
    auto sky = Child(model, "asset").append_child("texture");
    Set(sky, "name", "texture");
    Set(sky, "type", "skybox");
    const std::array faces = {"left", "right", "up", "down", "front", "back"};
    const std::array files = {"lf", "rt", "up", "dn", "ft", "bk"};
    for (int i = 0; i < 6; ++i) {
      Set(sky, (std::string("file") + faces[i]).c_str(),
          labmaze_asset_path_ + "/sky_03/" + files[i] + ".png");
    }
    Attach(model, "labmaze_sky_03/", RootJoint::kFixed);
  } else {
    auto sky = Asset().append_child("texture");
    Set(sky, "name", "aesthetic_skybox");
    Set(sky, "type", "skybox");
    Set(sky, "file",
        asset_path_ +
            "/locomotion/arenas/assets/outdoor_natural/OutdoorSkybox2048.png");
    Set(sky, "gridsize", "3 4");
    Set(sky, "gridlayout", ".U..LFRB.D..");
  }
  add_textures("labmaze_style_01/", kWallTextures, "wall");
  std::vector<std::string> floors;
  if (rodent) {
    for (const char* name : {"aesthetic_texture_main", "aesthetic_texture"}) {
      auto texture = Asset().append_child("texture");
      Set(texture, "name", name);
      Set(texture, "type", "2d");
      Set(texture, "file",
          asset_path_ +
              "/locomotion/arenas/assets/outdoor_natural/"
              "OutdoorGrassFloorD.png");
      floors.emplace_back(name);
    }
  } else {
    add_textures("labmaze_style_01_1/", kFloorTextures, "floor");
    for (auto name : kFloorTextures) {
      floors.push_back("labmaze_style_01_1/" + std::string(name));
    }
  }
  auto ground = World().append_child("geom");
  Set(ground, "name", "ground");
  Set(ground, "type", "plane");
  Set(ground, "pos", {0, 0, 0});
  Set(ground, "size", {size * scale / 2, size * scale / 2, 1});
  Set(ground, "rgba", {0, 0, 0, 0});
  ground_geoms.emplace_back("ground");
  auto walls = World().append_child("body");
  Set(walls, "name", "maze_body");
  Set(Child(Child(Root(), "visual"), "map"), "znear", .0005);
  auto camera = World().append_child("camera");
  Set(camera, "name", "top_camera");
  Set(camera, "pos", {0, 0, 100});
  Set(camera, "zaxis", {0, 0, 1});
  Set(camera, "fovy",
      360 / std::acos(-1) * std::atan2(1.1 * size * scale / 2, 100));
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const char token = maze_entities[y * (size + 1) + x];
      std::array<double, 3> position{(x - offset) * scale,
                                     -(y - offset) * scale, 0};
      if (token == 'G') {
        target_positions.push_back(position);
      }
      if (token == 'P') {
        spawn_positions.push_back(position);
      }
    }
  }
  auto planes = Root().append_child("envpool_maze_world");
  const std::string wall_texture =
      "labmaze_style_01/" +
      std::string(kWallTextures[random->Int(kWallTextures.size())]);
  int wall_id = 0;
  const std::array<std::array<double, 6>, 6> axes{{{0, -1, 0, 0, 0, 1},
                                                   {0, 1, 0, 0, 0, 1},
                                                   {1, 0, 0, 0, 0, 1},
                                                   {-1, 0, 0, 0, 0, 1},
                                                   {-1, 0, 0, 0, 1, 0},
                                                   {1, 0, 0, 0, 1, 0}}};
  for (const auto& wall : Cover(maze_entities, '*', size)) {
    const std::string name = "wall*_" + std::to_string(wall_id++);
    std::array<double, 3> pos{
        (wall.x + (wall.width - 1) / 2.0 - offset) * scale,
        -(wall.y + (wall.height - 1) / 2.0 - offset) * scale, height / 2};
    std::array<double, 3> half{wall.width * scale / 2, wall.height * scale / 2,
                               height / 2};
    auto geom = walls.append_child("geom");
    Set(geom, "name", name);
    Set(geom, "type", "box");
    Set(geom, "pos", Numbers(pos.data(), 3));
    Set(geom, "size", Numbers(half.data(), 3));
    Set(geom, "group", 3);
    for (int axis = 0; axis < 3; ++axis) {
      const int a = axis == 0 ? 1 : 0;
      const int b = axis == 2 ? 1 : 2;
      const auto material_name = name + "_" + "xyz"[axis];
      auto material = Asset().append_child("material");
      Set(material, "name", material_name);
      Set(material, "texture", wall_texture);
      Set(material, "texrepeat", {2 * half[a] / scale, 2 * half[b] / scale});
      for (int side = 0; side < 2; ++side) {
        if (axis == 2 && side == 0) {
          continue;
        }
        std::array<double, 3> face;
        std::copy_n(pos.data(), 3, face.data());
        face[axis] += ((side != 0) ? 1 : -1) * half[axis];
        auto plane = planes.append_child("geom");
        Set(plane, "name",
            name + "_texturing_" + ((side != 0) ? "pos_" : "neg_") +
                "xyz"[axis]);
        Set(plane, "type", "plane");
        Set(plane, "pos", Numbers(face.data(), 3));
        Set(plane, "size", {half[a], half[b], scale});
        Set(plane, "xyaxes", Numbers(axes[axis * 2 + side].data(), 6));
        Set(plane, "material", material_name);
        Set(plane, "contype", 0);
        Set(plane, "conaffinity", 0);
      }
    }
  }
  const auto main_floor = random->Int(floors.size());
  for (char variation : std::string(".ABCDEFGHIJKLMNOPQRSTUVWXYZ")) {
    if (maze_variations.find(variation) == std::string::npos) {
      break;
    }
    auto texture = main_floor;
    if (variation != '.') {
      do {
        texture = random->Int(floors.size());
      } while (texture == main_floor);
    }
    int tile_id = 0;
    for (const auto& tile : Cover(maze_variations, variation, size)) {
      const auto name =
          std::string("floor_") +
          (variation == '.' ? "" : std::string(1, variation) + "_") +
          std::to_string(tile_id++);
      auto material = Asset().append_child("material");
      Set(material, "name", name);
      Set(material, "texture", floors[texture]);
      Set(material, "texrepeat",
          {static_cast<double>(tile.width), static_cast<double>(tile.height)});
      auto plane = planes.append_child("geom");
      Set(plane, "name", name);
      Set(plane, "type", "plane");
      Set(plane, "material", name);
      Set(plane, "pos",
          {(tile.x + (tile.width - 1) / 2.0 - offset) * scale,
           -(tile.y + (tile.height - 1) / 2.0 - offset) * scale, 0});
      Set(plane, "size",
          {tile.width * scale / 2, tile.height * scale / 2, scale});
      Set(plane, "contype", 0);
      Set(plane, "conaffinity", 0);
    }
  }
}

void Scene::AddTargets(const TaskConfig& task, RandomState* random) {
  // Composer adds these planes at episode initialization, after the walker.
  for (auto plane : Root().child("envpool_maze_world").children()) {
    World().append_copy(plane);
  }
  Root().remove_child("envpool_maze_world");
  auto positions = target_positions;
  random->Shuffle(&positions);
  const bool heterogeneous = task.task == Task::kHeterogeneous;
  if (heterogeneous && random->Int(2) == 0) {
    std::swap(target_colors_[0], target_colors_[1]);
  }
  const double radius = task.walker == Walker::kRodent ? .05 : .4;
  for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
    const int type =
        heterogeneous && i >= static_cast<int>(positions.size() / 2) ? 1 : 0;
    const int color = heterogeneous ? target_colors_[type] : 2;
    const auto name =
        "target_" + std::to_string(type) + "_" +
        std::to_string(i - ((type != 0) ? positions.size() / 2 : 0)) + "/";
    TargetSphere(name, radius, task.walker == Walker::kRodent ? .125 : 1,
                 color);
    Set(World().last_child(), "pos", Numbers(positions[i].data(), 3));
    targets.push_back(name);
    target_types.push_back(type);
  }
}

void Scene::TargetSphere(const std::string& name, double radius, double height,
                         int color, bool two_touch) {
  pugi::xml_document document;
  auto model = document.append_child("mujoco");
  const std::string material_name =
      two_touch ? "target_sphere_init" : "target_sphere";
  const std::array<std::string, 3> texture_names{
      material_name, "target_sphere_inter", "target_sphere_final"};
  for (int i = 0; i < (two_touch ? 3 : 1); ++i) {
    auto texture = Child(model, "asset").append_child("texture");
    Set(texture, "name", texture_names[i]);
    Set(texture, "type", "cube");
    Set(texture, "builtin", "checker");
    Set(texture, "rgb1",
        {color == 1 ? .4 : 0, color == 0 ? .4 : 0, color == 2 ? .4 : 0});
    Set(texture, "rgb2",
        {color == 1 ? .7 : 0, color == 0 ? .7 : 0, color == 2 ? .7 : 0});
    if (i == 1) {
      Set(texture, "rgb1", {1, 1, .4});
      Set(texture, "rgb2", {.7, .7, 0});
    } else if (i == 2) {
      Set(texture, "rgb1", {.4, .7, 1});
      Set(texture, "rgb2", {0, .4, .7});
    }
    Set(texture, "width", 100);
    Set(texture, "height", 100);
  }
  auto material = Child(model, "asset").append_child("material");
  Set(material, "name", material_name);
  Set(material, "texture", material_name);
  auto geom = Child(model, "worldbody").append_child("geom");
  Set(geom, "name", "geom");
  Set(geom, "type", "sphere");
  Set(geom, "margin", -2 * radius);
  Set(geom, "gap", 2 * radius);
  Set(geom, "pos", {0, 0, height});
  Set(geom, "size", {radius});
  Set(geom, "material", material_name);
  Attach(model, name, RootJoint::kFixed);
}

void Scene::TwoTouchTarget() {
  TargetSphere("target_0_0/", .025, .14, 0, true);
  Set(Find(World(), "geom", "target_0_0/geom"), "pos", {1, 1, .14});
  targets.emplace_back("target_0_0/");
  target_types.push_back(0);
}

}  // namespace mujoco_locomotion
