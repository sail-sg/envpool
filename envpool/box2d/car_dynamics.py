# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Top-down car dynamics simulation.

Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
This simulation is a bit more detailed, with wheels rotation.

Created by Oleg Klimov
"""

import math

import Box2D
import numpy as np
from gym.error import DependencyNotInstalled

try:
  from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
except ImportError:
  raise DependencyNotInstalled(
    "box2D is not installed, run `pip install gym[box2d]`"
  )

SIZE = 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
FRICTION_LIMIT = (
  1000000 * SIZE * SIZE
)  # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [
  (+25, +20),
  (+50, -10),
  (+50, -40),
  (+20, -90),
  (-20, -90),
  (-50, -40),
  (-50, -10),
  (-25, +20),
]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]
WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)


class Car:

  def __init__(self, world, init_angle, init_x, init_y):
    print("Car constructor", init_angle, init_x, init_y)
    self.world: Box2D.b2World = world
    self.hull: Box2D.b2Body = self.world.CreateDynamicBody(
      position=(init_x, init_y),
      angle=init_angle,
      fixtures=[
        fixtureDef(
          shape=polygonShape(
            vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY1]
          ),
          density=1.0,
        ),
        fixtureDef(
          shape=polygonShape(
            vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY2]
          ),
          density=1.0,
        ),
        fixtureDef(
          shape=polygonShape(
            vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY3]
          ),
          density=1.0,
        ),
        fixtureDef(
          shape=polygonShape(
            vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY4]
          ),
          density=1.0,
        ),
      ],
    )
    self.wheels = []
    self.fuel_spent = 0.0
    WHEEL_POLY = [
      (-WHEEL_W, +WHEEL_R),
      (+WHEEL_W, +WHEEL_R),
      (+WHEEL_W, -WHEEL_R),
      (-WHEEL_W, -WHEEL_R),
    ]
    for wx, wy in WHEELPOS:
      front_k = 1.0 if wy > 0 else 1.0
      w = self.world.CreateDynamicBody(
        position=(init_x + wx * SIZE, init_y + wy * SIZE),
        angle=init_angle,
        fixtures=fixtureDef(
          shape=polygonShape(
            vertices=[
              (x * front_k * SIZE, y * front_k * SIZE) for x, y in WHEEL_POLY
            ]
          ),
          density=0.1,
          categoryBits=0x0020,
          maskBits=0x001,
          restitution=0.0,
        ),
      )
      w.wheel_rad = front_k * WHEEL_R * SIZE
      w.color = WHEEL_COLOR
      w.gas = 0.0
      w.brake = 0.0
      w.steer = 0.0
      w.phase = 0.0  # wheel angle
      w.omega = 0.0  # angular velocity
      rjd = revoluteJointDef(
        bodyA=self.hull,
        bodyB=w,
        localAnchorA=(wx * SIZE, wy * SIZE),
        localAnchorB=(0, 0),
        enableMotor=True,
        enableLimit=True,
        maxMotorTorque=180 * 900 * SIZE * SIZE,
        motorSpeed=0,
        lowerAngle=-0.4,
        upperAngle=+0.4,
      )
      # todo check joint
      w.joint = self.world.CreateJoint(rjd)
      w.tiles = set()
      w.userData = w
      self.wheels.append(w)
    self.drawlist = self.wheels + [self.hull]

  def gas(self, gas):
    """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
    gas = np.clip(gas, 0, 1)
    for w in self.wheels[2:4]:
      diff = gas - w.gas
      if diff > 0.1:
        diff = 0.1  # gradually increase, but stop immediately
      w.gas += diff

  def brake(self, b):
    """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
    for w in self.wheels:
      w.brake = b

  def steer(self, s):
    """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
    self.wheels[0].steer = s
    self.wheels[1].steer = s

  def step(self, dt):
    print("car step!!!!")
    for w in self.wheels:
      # Steer each wheel
      dir = np.sign(w.steer - w.joint.angle)
      val = abs(w.steer - w.joint.angle)
      w.joint.motorSpeed = dir * min(50.0 * val, 3.0)

      # Position => friction_limit
      friction_limit = FRICTION_LIMIT * 0.6  # Grass friction if no tile
      for tile in w.tiles:
        friction_limit = max(
          friction_limit, FRICTION_LIMIT * tile.road_friction
        )
        grass = False

      # Force
      forw = w.GetWorldVector((0, 1))
      side = w.GetWorldVector((1, 0))
      v = w.linearVelocity
      vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
      vs = side[0] * v[0] + side[1] * v[1]  # side speed

      # add small coef not to divide by zero
      w.omega += (
        dt * ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA /
        (abs(w.omega) + 5.0)
      )
      self.fuel_spent += dt * ENGINE_POWER * w.gas

      if w.brake >= 0.9:
        w.omega = 0
      elif w.brake > 0:
        BRAKE_FORCE = 15  # radians per second
        dir = -np.sign(w.omega)
        val = BRAKE_FORCE * w.brake
        if abs(val) > abs(w.omega):
          val = abs(w.omega)  # low speed => same as = 0
        w.omega += dir * val
      w.phase += w.omega * dt

      vr = w.omega * w.wheel_rad  # rotating wheel speed
      f_force = -vf + vr  # force direction is direction of speed difference
      p_force = -vs

      # Physically correct is to always apply friction_limit until speed is equal.
      # But dt is finite, that will lead to oscillations if difference is already near zero.

      # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
      f_force *= 205000 * SIZE * SIZE
      p_force *= 205000 * SIZE * SIZE
      force = np.sqrt(np.square(f_force) + np.square(p_force))

      if abs(force) > friction_limit:
        f_force /= force
        p_force /= force
        force = friction_limit  # Correct physics here
        f_force *= force
        p_force *= force

      w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

      w.ApplyForceToCenter(
        (
          p_force * side[0] + f_force * forw[0],
          p_force * side[1] + f_force * forw[1],
        ),
        True,
      )

  def draw(self, surface, zoom, translation, angle, draw_particles=True):
    import pygame.draw

    for obj in self.drawlist:
      for f in obj.fixtures:
        trans = f.body.transform
        path = [trans * v for v in f.shape.vertices]
        path = [(coords[0], coords[1]) for coords in path]
        path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
        path = [
          (
            coords[0] * zoom + translation[0],
            coords[1] * zoom + translation[1],
          ) for coords in path
        ]
        color = [int(c * 255) for c in obj.color]

        pygame.draw.polygon(surface, color=color, points=path)

        if "phase" not in obj.__dict__:
          continue
        a1 = obj.phase
        a2 = obj.phase + 1.2  # radians
        s1 = math.sin(a1)
        s2 = math.sin(a2)
        c1 = math.cos(a1)
        c2 = math.cos(a2)
        if s1 > 0 and s2 > 0:
          continue
        if s1 > 0:
          c1 = np.sign(c1)
        if s2 > 0:
          c2 = np.sign(c2)
        white_poly = [
          (-WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
          (+WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
          (+WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
          (-WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
        ]
        white_poly = [trans * v for v in white_poly]

        white_poly = [(coords[0], coords[1]) for coords in white_poly]
        white_poly = [
          pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly
        ]
        white_poly = [
          (
            coords[0] * zoom + translation[0],
            coords[1] * zoom + translation[1],
          ) for coords in white_poly
        ]
        pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)

  def destroy(self):
    self.world.DestroyBody(self.hull)
    self.hull = None
    for w in self.wheels:
      self.world.DestroyBody(w)
    self.wheels = []
