/*
 * Copyright 2021 Garena Online Private Limited
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
// https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

#include "car_racing.h"

namespace box2d {

b2PolygonShape generatePolygon(float* array, int size) {
  b2Vec2* vertices = (b2Vec2*) malloc(size * sizeof(b2Vec2));
  for (int i = 0; i < 2 * size; i += 2) {
    vertices[i].Set(array[i], array[i + 1]);
  }
  b2PolygonShape polygon;
  polygon.Set(vertices, size);
  free(vertices);
  return polygon;
}


FrictionDetector::FrictionDetector(CarRacingEnv* _env)
        : env(_env){}
void FrictionDetector::BeginContact (b2Contact *contact) {
      _Contact(contact, true);
    }
void FrictionDetector::EndContact (b2Contact *contact) {
      _Contact(contact, false);
    }
void FrictionDetector::_Contact(b2Contact *contact, bool begin) {
      UserData* tile = nullptr;
      UserData* obj = nullptr;
      void* u1 = (void*) contact->GetFixtureA()->GetBody()->GetUserData().pointer;
      void* u2 = (void*) contact->GetFixtureB()->GetBody()->GetUserData().pointer;

      if (u1 && static_cast<UserData*>(u1)->isTile) {
        tile = static_cast<UserData*>(u1);
        obj = static_cast<UserData*>(u2);
      }
      if (u2 && static_cast<UserData*>(u2)->isTile) {
        tile = static_cast<UserData*>(u2);
        obj = static_cast<UserData*>(u1);
      }
      if (tile == nullptr) return;
  
      tile->tileColor[0] = kRoadColor[0];
      tile->tileColor[1] = kRoadColor[1];
      tile->tileColor[2] = kRoadColor[2];

      // if not obj or "tiles" not in obj.__dict__:
      //     return
      if (obj == nullptr) return;
      if (begin) {
        obj->objTiles.insert(tile);
        if (!tile->tileRoadVisited) {
          tile->tileRoadVisited = true;
          env->reward += 1000.0 / env->track.size();
          env->tile_visited_count += 1;
        }
      } else {
        obj->objTiles.erase(tile);
      }
    }
};


Car:: Car(b2World* _world, float init_angle, float init_x, float init_y)
      : world(_world), fuel_spent(0.0) {
        b2BodyDef bd;
        bd.position.Set(init_x, init_y);
        bd.angle = init_angle;
        bd.type = b2_dynamicBody;
        hull = world->CreateBody(&bd);

        b2PolygonShape polygon1 = generatePolygon(kHullPoly1, 4);
        fixtures.emplace_back(hull->CreateFixture(&polygon1, 1.f));

        b2PolygonShape polygon2 = generatePolygon(kHullPoly2, 4);
        fixtures.emplace_back(hull->CreateFixture(&polygon2, 1.f));

        b2PolygonShape polygon3 = generatePolygon(kHullPoly3, 8);
        fixtures.emplace_back(hull->CreateFixture(&polygon3, 1.f));

        b2PolygonShape polygon4 = generatePolygon(kHullPoly4, 4);
        fixtures.emplace_back(hull->CreateFixture(&polygon4, 1.f));

    }

}  // namespace box2d
