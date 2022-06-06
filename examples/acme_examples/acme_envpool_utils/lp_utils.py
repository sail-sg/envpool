# Copyright 2021 Garena Online Private Limited
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
"""Helpers for launching distributed acme experiments."""

import os
import socket
from typing import Optional

import launchpad as lp
from acme.jax import experiments
from acme.jax.experiments import config
from launchpad.nodes.python.local_multi_processing import PythonProcess

ActorId = int


class CPUManager:

  def __init__(self) -> None:
    self.total_cpu = os.cpu_count() or 0
    self.given_count = 0

  def get(self, count: int) -> dict:
    assert self.given_count + count <= self.total_cpu, \
      "no enough CPU resources!"
    cpu_id = [
      str(x) for x in range(self.given_count, self.given_count + count)
    ]
    cpu_id = ",".join(cpu_id)
    self.given_count += count
    return {"cpu_list": cpu_id}


class TPUManager:

  def __init__(self, split: int) -> None:
    assert split in [1, 2, 4], "split must be 1, 2 or 4"
    if split == 1:
      self.chips_per_host_bounds = "2,2,1"
      self.tpu_ids = ["0,1,2,3"]
    elif split == 2:
      self.chips_per_host_bounds = "1,2,1"
      self.tpu_ids = ["0,1", "2,3"]
    else:
      self.chips_per_host_bounds = "1,1,1"
      self.tpu_ids = ["0", "1", "2", "3"]

    self.split = split
    self.given_count = 0
    host_name = socket.gethostname()
    try:
      self.tpu_task_id = int(host_name[-1])
    except ValueError:
      self.tpu_task_id = 0

  def _template(self, port: int, tpu_id: str) -> dict:
    return {
      'TPU_CHIPS_PER_HOST_BOUNDS': self.chips_per_host_bounds,
      'TPU_HOST_BOUNDS': '1,1,1',
      'TPU_VISIBLE_DEVICES': tpu_id,
      'TPU_MESH_CONTROLLER_ADDRESS': f'localhost:{port}',
      'TPU_MESH_CONTROLLER_PORT': f'{port}',
      'CLOUD_TPU_TASK_ID': self.tpu_task_id,
      'TF_CPP_MIN_LOG_LEVEL': '0',
    }

  def get(self) -> dict:
    assert self.given_count < self.split, "no enough TPU resources!"
    flag_dict = self._template(
      8476 + self.given_count, self.tpu_ids[self.given_count]
    )
    self.given_count += 1
    return flag_dict


class TPUResourceManager:

  def __init__(self, tpu_split: int = 4) -> None:
    self.cpu_manager = CPUManager()
    self.tpu_manager = TPUManager(split=tpu_split)
    self.stats = [
      f"TPU split = {tpu_split} | CPU total = {self.cpu_manager.total_cpu}"
    ]

  def get(
    self, cpu_count: int, need_tpu: bool = False, remark: str = ""
  ) -> dict:
    assert cpu_count > 0, "please ask for at least 1 cpu"
    flags = {"TPU_VISIBLE_DEVICES": ""}
    cpu_flags = self.cpu_manager.get(cpu_count)
    flags.update(cpu_flags)
    if need_tpu:
      tpu_flags = self.tpu_manager.get()
      flags.update(tpu_flags)
    if remark:
      cpu = cpu_flags.get("cpu_list", "")
      stats = f"{remark}: [cpu] {cpu}"
      if need_tpu:
        tpu = tpu_flags.get("TPU_VISIBLE_DEVICES", "")
        stats += f", [tpu] {tpu}"
      self.stats.append(stats)
    return flags

  def describe(self) -> str:
    return "===Resources Allocation===\n" + "\n".join(self.stats)


def run_distributed_experiment(
  program: lp.Program,
  experiment: config.Config,
  num_actors: int,
  resource_config: Optional[dict] = None
):
  if program is None:
    program = experiments.make_distributed_experiment(
      experiment=experiment, num_actors=num_actors
    )
  vm_resource = TPUResourceManager(1)
  if resource_config is not None:
    resources = {}
    for label, cfg in resource_config.items():
      resources[label] = PythonProcess(
        env=vm_resource.get(cfg[0], cfg[1], remark=label)
      )
  else:  # Default resource config.
    actor_resource = vm_resource.get(2, remark="actor")
    replay_resource = vm_resource.get(2, remark="replay")
    learner_resource = vm_resource.get(4, True, remark="learner")
    resources = {
      "actor": PythonProcess(env=actor_resource),
      "replay": PythonProcess(env=replay_resource),
      "learner": PythonProcess(env=learner_resource),
    }
  print(vm_resource.describe())
  lp.launch(
    program,
    terminal="tmux_session",
    launch_type="local_mp",
    local_resources=resources
  )
