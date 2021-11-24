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
import os
import pprint

import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

import envpool


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, default="Pendulum-v0")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--buffer-size", type=int, default=20000)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--gamma", type=float, default=0.95)
  parser.add_argument("--epoch", type=int, default=5)
  parser.add_argument("--step-per-epoch", type=int, default=150000)
  parser.add_argument("--episode-per-collect", type=int, default=16)
  parser.add_argument("--repeat-per-collect", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
  parser.add_argument("--training-num", type=int, default=16)
  parser.add_argument("--test-num", type=int, default=100)
  parser.add_argument("--logdir", type=str, default="log")
  parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu"
  )
  # ppo special
  parser.add_argument("--vf-coef", type=float, default=0.25)
  parser.add_argument("--ent-coef", type=float, default=0.0)
  parser.add_argument("--eps-clip", type=float, default=0.2)
  parser.add_argument("--max-grad-norm", type=float, default=0.5)
  parser.add_argument("--gae-lambda", type=float, default=0.95)
  parser.add_argument("--rew-norm", type=int, default=1)
  parser.add_argument("--dual-clip", type=float, default=None)
  parser.add_argument("--value-clip", type=int, default=1)
  parser.add_argument("--norm-adv", type=int, default=1)
  parser.add_argument("--recompute-adv", type=int, default=0)
  parser.add_argument("--watch", action="store_true")
  return parser.parse_args()


def run_ppo(args):
  env = gym.make(args.task)
  if args.task == "Pendulum-v0":
    env.spec.reward_threshold = -250
  args.state_shape = env.observation_space.shape or env.observation_space.n
  args.action_shape = env.action_space.shape or env.action_space.n
  args.max_action = env.action_space.high[0]

  train_envs = envpool.make_gym(args.task, num_envs=args.training_num)
  test_envs = envpool.make_gym(args.task, num_envs=args.test_num)
  args.state_shape = env.observation_space.shape or env.observation_space.n
  args.action_shape = env.action_space.shape or env.action_space.n
  # ss_ = train_envs.observation_space.shape or train_envs.observation_space.n
  # assert ss_ == args.state_shape
  # as_ = train_envs.action_space.shape or train_envs.action_space.n
  # assert as_ == args.action_shape

  # train_envs = DummyVectorEnv(
  #   [lambda: gym.make(args.task) for _ in range(args.training_num)])
  # train_envs.seed(args.seed)
  # seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  # model
  net = Net(
    args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device
  )
  actor = ActorProb(
    net, args.action_shape, max_action=args.max_action, device=args.device
  ).to(args.device)
  critic = Critic(
    Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
    device=args.device
  ).to(args.device)
  actor_critic = ActorCritic(actor, critic)
  # orthogonal initialization
  for m in actor_critic.modules():
    if isinstance(m, torch.nn.Linear):
      torch.nn.init.orthogonal_(m.weight)
      torch.nn.init.zeros_(m.bias)
  optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

  # replace DiagGuassian with Independent(Normal) which is equivalent
  # pass *logits to be consistent with policy.forward
  def dist(*logits):
    return Independent(Normal(*logits), 1)  # pylint: disable=E1120

  policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist,
    discount_factor=args.gamma,
    max_grad_norm=args.max_grad_norm,
    eps_clip=args.eps_clip,
    vf_coef=args.vf_coef,
    ent_coef=args.ent_coef,
    reward_normalization=args.rew_norm,
    advantage_normalization=args.norm_adv,
    recompute_advantage=args.recompute_adv,
    # dual_clip=args.dual_clip,
    # dual clip cause monotonically increasing log_std :)
    value_clip=args.value_clip,
    gae_lambda=args.gae_lambda,
    action_space=env.action_space,
  )
  # collector
  train_collector = Collector(
    policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
  )
  test_collector = Collector(policy, test_envs)
  # log
  log_path = os.path.join(args.logdir, args.task, "ppo")
  writer = SummaryWriter(log_path)
  logger = TensorboardLogger(writer)

  def save_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

  def stop_fn(mean_rewards):
    return mean_rewards >= env.spec.reward_threshold

  def watch():
    # Let"s watch its performance!
    env = DummyVectorEnv(
      [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    env.seed(args.seed)
    policy.eval()
    collector = Collector(policy, env)
    result = collector.collect(n_episode=args.test_num)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
    if args.task == "Pendulum-v0":
      assert stop_fn(rews.mean() + 50)
    return rews.mean()

  if args.watch:
    return watch()

  # trainer
  result = onpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    args.epoch,
    args.step_per_epoch,
    args.repeat_per_collect,
    args.test_num,
    args.batch_size,
    episode_per_collect=args.episode_per_collect,
    stop_fn=stop_fn,
    save_fn=save_fn,
    logger=logger,
  )
  pprint.pprint(result)
  assert stop_fn(result["best_reward"])
  watch()


if __name__ == "__main__":
  run_ppo(get_args())
