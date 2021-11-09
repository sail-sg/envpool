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
"""Atari Network from tianshou repo."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union, no_type_check

import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
  """Reference: Human-level control through deep reinforcement learning."""

  def __init__(
    self,
    c: int,
    h: int,
    w: int,
    action_shape: Sequence[int],
    device: Union[str, int, torch.device] = "cpu",
    features_only: bool = False,
  ) -> None:
    """Constructor of DQN."""
    super().__init__()
    self.device = device
    self.net = nn.Sequential(
      nn.Conv2d(c, 32, kernel_size=8, stride=4),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU(inplace=True),
      nn.Flatten(),
    )
    with torch.no_grad():
      self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
    if not features_only:
      self.net = nn.Sequential(
        self.net,
        nn.Linear(self.output_dim, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, np.prod(action_shape)),
      )
      self.output_dim = np.prod(action_shape)

  @no_type_check
  def forward(
    self,
    x: Union[np.ndarray, torch.Tensor],
    state: Optional[Any] = None,
    info: Optional[Dict[str, Any]] = None,
  ) -> Tuple[torch.Tensor, Any]:
    r"""Mapping: x -> Q(x, \*)."""
    x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
    return self.net(x), state


class C51(DQN):
  """Reference: A distributional perspective on reinforcement learning."""

  def __init__(
    self,
    c: int,
    h: int,
    w: int,
    action_shape: Sequence[int],
    num_atoms: int = 51,
    device: Union[str, int, torch.device] = "cpu",
  ) -> None:
    """Constructor of C51."""
    self.action_num = np.prod(action_shape)
    super().__init__(c, h, w, [self.action_num * num_atoms], device)
    self.num_atoms = num_atoms

  def forward(
    self,
    x: Union[np.ndarray, torch.Tensor],
    state: Optional[Any] = None,
    info: Optional[Dict[str, Any]] = None,
  ) -> Tuple[torch.Tensor, Any]:
    r"""Mapping: x -> Z(x, \*)."""
    x, state = super().forward(x)
    x = x.view(-1, self.num_atoms).softmax(dim=-1)
    x = x.view(-1, self.action_num, self.num_atoms)
    return x, state


class QRDQN(DQN):
  """Reference: Distributional Reinforcement Learning with Quantile \
  Regression."""

  def __init__(
    self,
    c: int,
    h: int,
    w: int,
    action_shape: Sequence[int],
    num_quantiles: int = 200,
    device: Union[str, int, torch.device] = "cpu",
  ) -> None:
    """Constructor of QRDQN."""
    self.action_num = np.prod(action_shape)
    super().__init__(c, h, w, [self.action_num * num_quantiles], device)
    self.num_quantiles = num_quantiles

  def forward(
    self,
    x: Union[np.ndarray, torch.Tensor],
    state: Optional[Any] = None,
    info: Optional[Dict[str, Any]] = None,
  ) -> Tuple[torch.Tensor, Any]:
    r"""Mapping: x -> Z(x, \*)."""
    x, state = super().forward(x)
    x = x.view(-1, self.action_num, self.num_quantiles)
    return x, state
