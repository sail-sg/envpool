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

import argparse
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from gae import compute_gae
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import envpool


class CnnActorCritic(nn.Module):

  def __init__(self, action_size: int):
    super().__init__()
    layers = [
      nn.Conv2d(4, 32, kernel_size=8, stride=4),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU(inplace=True),
      nn.Flatten(),
      nn.Linear(3136, 512),
      nn.ReLU(inplace=True),
    ]
    self.net = nn.Sequential(*layers)
    self.actor = nn.Linear(512, action_size)
    self.critic = nn.Linear(512, 1)
    # orthogonal initialization
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    feature = self.net(x / 255.0)
    return F.softmax(self.actor(feature), dim=-1), self.critic(feature)


class MlpActorCritic(nn.Module):

  def __init__(self, state_size: int, action_size: int):
    super().__init__()
    layers = [
      nn.Linear(state_size, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 64),
      nn.ReLU(inplace=True),
    ]
    self.net = nn.Sequential(*layers)
    self.actor = nn.Linear(64, action_size)
    self.critic = nn.Linear(64, 1)
    # orthogonal initialization
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    feature = self.net(x)
    return F.softmax(self.actor(feature), dim=-1), self.critic(feature)


class DiscretePPO:

  def __init__(
    self,
    actor_critic: nn.Module,
    optim: torch.optim.Optimizer,
    dist_fn: Type[torch.distributions.Distribution],
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: argparse.Namespace,
  ):
    self.actor_critic = actor_critic
    self.optim = optim
    self.dist_fn = dist_fn
    self.config = config
    self.training = True
    self.numenv = config.numenv
    self.lr_scheduler = lr_scheduler

  def predictor(self, obs: torch.Tensor) -> torch.Tensor:
    logits, value = self.actor_critic(obs)
    if not self.training:
      action = logits.argmax(-1)
      dist = None
      log_prob = None
    else:
      dist = self.dist_fn(logits)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    return action, log_prob, value, dist

  def learner(
    self,
    obs: torch.Tensor,
    act: torch.Tensor,
    rew: np.ndarray,
    done: np.ndarray,
    env_id: np.ndarray,
    log_prob: torch.Tensor,
    value: torch.Tensor,
  ) -> Dict[str, float]:
    # compute GAE
    T, B = rew.shape
    N = T * B
    returns, advantage, mask = compute_gae(
      self.config.gamma,
      self.config.gae_lambda,
      value.cpu().numpy().reshape(T, B),
      rew,
      done,
      env_id,
      self.numenv,
    )
    index = np.arange(N)[mask.reshape(N) > 0]
    returns = torch.from_numpy(returns.reshape(N)).to(value.device)
    advantage = torch.from_numpy(advantage.reshape(N)).to(value.device)
    losses, clip_losses, vf_losses, ent_losses = [], [], [], []
    # split traj
    for _ in range(self.config.repeat_per_collect):
      np.random.shuffle(index)
      for start_index in range(0, len(index), self.config.batch_size):
        i = index[start_index:start_index + self.config.batch_size]
        b_adv = advantage[i]
        if self.config.norm_adv:
          mean, std = b_adv.mean(), b_adv.std()
          b_adv = (b_adv - mean) / std
        _, b_log_prob, b_value, b_dist = self.predictor(obs[i])
        ratio = (b_dist.log_prob(act[i]) - log_prob[i]).exp().float()
        ratio = ratio.reshape(ratio.shape[0], -1).transpose(0, 1)
        surr1 = ratio * b_adv
        surr2 = ratio.clamp(
          1.0 - self.config.eps_clip, 1.0 + self.config.eps_clip
        ) * b_adv
        clip_loss = -torch.min(surr1, surr2).mean()
        vf_loss = (returns[i] - b_value.flatten()).pow(2).mean()
        ent_loss = b_dist.entropy().mean()
        loss = clip_loss + self.config.vf_coef * vf_loss - self.config.ent_coef * ent_loss
        # update param
        self.optim.zero_grad()
        loss.backward()
        if self.config.max_grad_norm:
          nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), max_norm=self.config.max_grad_norm
          )
        self.optim.step()
        clip_losses.append(clip_loss.item())
        vf_losses.append(vf_loss.item())
        ent_losses.append(ent_loss.item())
        losses.append(loss.item())
    self.lr_scheduler.step()
    # return loss
    return {
      "loss": np.mean(losses),
      "loss/clip": np.mean(clip_losses),
      "loss/vf": np.mean(vf_losses),
      "loss/ent": np.mean(ent_losses),
    }


class MovAvg:

  def __init__(self, size: int = 100):
    self.size = size
    self.cache = []

  def add_bulk(self, x: np.ndarray) -> float:
    self.cache += x.tolist()
    if len(self.cache) > self.size:
      self.cache = self.cache[-self.size:]
    return np.mean(self.cache)


class Actor:

  def __init__(
    self,
    policy: DiscretePPO,
    train_envs: Any,
    test_envs: Any,
    writer: SummaryWriter,
    config: argparse.Namespace,
  ):
    self.policy = policy
    self.train_envs = train_envs
    self.test_envs = test_envs
    self.writer = writer
    self.config = config
    self.obs_batch = []
    self.act_batch = []
    self.rew_batch = []
    self.done_batch = []
    self.envid_batch = []
    self.value_batch = []
    self.logprob_batch = []
    self.reward_stat = np.zeros(len(train_envs))
    train_envs.async_reset()
    test_envs.async_reset()

  def run(self) -> None:
    env_step = 0
    stat = MovAvg()
    episodic_reward = 0
    for epoch in range(1, 1 + self.config.epoch):
      with tqdm.trange(
        self.config.step_per_epoch, desc=f'Epoch #{epoch}'
      ) as t:
        while t.n < self.config.step_per_epoch:
          # collect
          for i in range(self.config.step_per_collect // self.config.waitnum):
            obs, rew, done, info = self.train_envs.recv()
            env_id = info["env_id"]
            obs = torch.tensor(obs, device="cuda")
            self.obs_batch.append(obs)
            with torch.no_grad():
              act, log_prob, value, _ = self.policy.predictor(obs)
            self.act_batch.append(act)
            self.logprob_batch.append(log_prob)
            self.value_batch.append(value)
            self.train_envs.send(act.cpu().numpy(), env_id)
            self.rew_batch.append(rew)
            self.done_batch.append(done)
            self.envid_batch.append(env_id)
            t.update(self.config.waitnum)
            env_step += self.config.waitnum
            self.reward_stat[env_id] += info["reward"]
            done = done & (info["lives"] == 0)
            if np.any(done):
              done_id = env_id[done]
              episodic_reward = self.reward_stat[done_id]
              self.reward_stat[done_id] = 0
              self.writer.add_scalar(
                "train/reward",
                stat.add_bulk(episodic_reward),
                global_step=env_step,
              )
          # learn
          result = self.policy.learner(
            torch.cat(self.obs_batch),
            torch.cat(self.act_batch),
            np.stack(self.rew_batch),
            np.stack(self.done_batch),
            np.stack(self.envid_batch),
            torch.cat(self.logprob_batch),
            torch.cat(self.value_batch),
          )
          result["reward"] = np.mean(episodic_reward)
          self.obs_batch = []
          self.act_batch = []
          self.rew_batch = []
          self.done_batch = []
          self.envid_batch = []
          self.value_batch = []
          self.logprob_batch = []
          t.set_postfix(**result)
          for k, v in result.items():
            self.writer.add_scalar(f"train/{k}", v, global_step=env_step)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, default='Pong-v5')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--lr', type=float, default=2.5e-4)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--epoch', type=int, default=100)
  parser.add_argument('--step-per-epoch', type=int, default=100000)
  parser.add_argument('--step-per-collect', type=int, default=1024)
  parser.add_argument('--repeat-per-collect', type=int, default=4)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--numenv', type=int, default=16)
  parser.add_argument('--waitnum', type=int, default=8)
  parser.add_argument('--test-num', type=int, default=10)
  parser.add_argument('--logdir', type=str, default='log')
  parser.add_argument(
    '--watch',
    default=False,
    action='store_true',
    help='watch the play of pre-trained policy only'
  )
  # ppo special
  parser.add_argument('--vf-coef', type=float, default=0.25)
  parser.add_argument('--ent-coef', type=float, default=0.01)
  parser.add_argument('--gae-lambda', type=float, default=0.95)
  parser.add_argument('--eps-clip', type=float, default=0.2)
  parser.add_argument('--max-grad-norm', type=float, default=0.5)
  parser.add_argument('--rew-norm', type=int, default=0)
  parser.add_argument('--norm-adv', type=int, default=1)
  parser.add_argument('--recompute-adv', type=int, default=0)
  parser.add_argument('--dual-clip', type=float, default=None)
  parser.add_argument('--value-clip', type=int, default=0)
  parser.add_argument('--lr-decay', type=int, default=True)
  args = parser.parse_args()

  train_envs = envpool.make(
    args.task,
    env_type="gym",
    num_envs=args.numenv,
    batch_size=args.waitnum,
    episodic_life=True,
    reward_clip=True,
    # thread_affinity=False,
  )
  test_envs = envpool.make(
    args.task,
    env_type="gym",
    num_envs=args.test_num,
    episodic_life=False,
    reward_clip=False,
    # thread_affinity=False,
  )
  state_n = np.prod(train_envs.observation_space.shape)
  action_n = train_envs.action_space.n
  actor_critic = nn.DataParallel(CnnActorCritic(action_n).cuda())
  # actor_critic = nn.DataParallel(MlpActorCritic(state_n, action_n).cuda())
  optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
  # decay learning rate to 0 linearly
  max_update_num = np.ceil(
    args.step_per_epoch / args.step_per_collect
  ) * args.epoch

  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
  )
  dist = torch.distributions.Categorical
  policy = DiscretePPO(
    actor_critic=actor_critic,
    optim=optim,
    dist_fn=dist,
    lr_scheduler=lr_scheduler,
    config=args,
  )
  writer = SummaryWriter(args.logdir)
  writer.add_text("args", str(args))
  Actor(policy, train_envs, test_envs, writer, args).run()
