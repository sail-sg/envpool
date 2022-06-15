# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Util."""
import collections
import logging

import dm_env
import numpy as np
import tree

import wandb

logging.getLogger().setLevel(logging.INFO)

# Can represent either a single transition, a trajectory, or a batch of
# trajectories.
Transition = collections.namedtuple(
  'Transition', ['timestep', 'agent_out', 'agent_state']
)


def _preprocess_none(t) -> np.ndarray:
  if t is None:
    return np.array(0., dtype=np.float32)
  else:
    return np.asarray(t)


def preprocess_step(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  if timestep.discount is None:
    timestep = timestep._replace(discount=1.)
  return tree.map_structure(_preprocess_none, timestep)


def preprocess_step_env_pool(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  timestep = timestep._replace(
    observation=timestep.observation.obs.astype(np.float32)
  )
  return timestep


class NullLogger:
  """Logger that does nothing."""

  def write(self, _):
    pass

  def close(self):
    pass


class AbslLogger:
  """Writes to logging.info."""

  def write(self, d):
    logging.info(d)

  def close(self):
    pass


class WBLogger:
  """Writes to logging.info and W&B."""

  def __init__(
    self, wb_project, wb_entity, run_name, config, use_wb=False
  ) -> None:
    self._use_wb = use_wb
    if use_wb:
      wandb.init(
        project=wb_project,
        entity=wb_entity,
        name=run_name,
        config=config,
      )

  def write(self, d):
    logging.info(d)
    if self._use_wb:
      wandb.log(d)

  def close(self):
    if self._use_wb:
      wandb.finish()
