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
"""Initialize CPU affinity for distributed launching."""

import os

from absl import logging

cpu_list = os.environ.get("cpu_list", None)
if cpu_list is not None:
  from absl import flags
  try:
    task_id = flags.FLAGS.lp_task_id
  except Exception:
    task_id = 0
  cpu_list = cpu_list.split(',')
  logging.info(f"{cpu_list}, {task_id}")
  cpu_mask = [int(i) + int(task_id) for i in cpu_list]
  os.sched_setaffinity(0, cpu_mask)
