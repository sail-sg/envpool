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
"""
https://github.com/vwxyzjn/cleanrl/pull/100
CleanRL: Solve Pong in about 5 mins by PPO:

python3 ppo_atari_envpool.py --clip-coef=0.2 --num-envs=16 \
  --num-minibatches=8 --num-steps=128 --update-epochs=3
"""

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import envpool


def parse_args():
  # fmt: off
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--exp-name",
    type=str,
    default=os.path.basename(__file__).rstrip(".py"),
    help="the name of this experiment"
  )
  parser.add_argument(
    "--gym-id",
    type=str,
    default="Pong-v5",
    help="the id of the gym environment"
  )
  parser.add_argument(
    "--learning-rate",
    type=float,
    default=2.5e-4,
    help="the learning rate of the optimizer"
  )
  parser.add_argument(
    "--seed", type=int, default=1, help="seed of the experiment"
  )
  parser.add_argument(
    "--total-timesteps",
    type=int,
    default=10000000,
    help="total timesteps of the experiments"
  )
  parser.add_argument(
    "--torch-deterministic",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="if toggled, `torch.backends.cudnn.deterministic=False`"
  )
  parser.add_argument(
    "--cuda",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="if toggled, cuda will be enabled by default"
  )
  parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases"
  )
  parser.add_argument(
    "--wandb-project-name",
    type=str,
    default="ppo-implementation-details",
    help="the wandb's project name"
  )
  parser.add_argument(
    "--wandb-entity",
    type=str,
    default=None,
    help="the entity (team) of wandb's project"
  )

  # Algorithm specific arguments
  parser.add_argument(
    "--num-envs",
    type=int,
    default=8,
    help="the number of parallel game environments"
  )
  parser.add_argument(
    "--num-steps",
    type=int,
    default=128,
    help="the number of steps to run in each environment per policy rollout"
  )
  parser.add_argument(
    "--anneal-lr",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="toggle learning rate annealing for policy and value networks"
  )
  parser.add_argument(
    "--gae",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="use GAE for advantage computation"
  )
  parser.add_argument(
    "--gamma", type=float, default=0.99, help="the discount factor gamma"
  )
  parser.add_argument(
    "--gae-lambda",
    type=float,
    default=0.95,
    help="the lambda for the general advantage estimation"
  )
  parser.add_argument(
    "--num-minibatches",
    type=int,
    default=4,
    help="the number of mini-batches"
  )
  parser.add_argument(
    "--update-epochs",
    type=int,
    default=4,
    help="the K epochs to update the policy"
  )
  parser.add_argument(
    "--norm-adv",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="toggle advantages normalization"
  )
  parser.add_argument(
    "--clip-coef",
    type=float,
    default=0.1,
    help="the surrogate clipping coefficient"
  )
  parser.add_argument(
    "--clip-vloss",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="toggle whether or not to use a clipped loss "
    "for the value function, as per the paper."
  )
  parser.add_argument(
    "--ent-coef", type=float, default=0.01, help="coefficient of the entropy"
  )
  parser.add_argument(
    "--vf-coef",
    type=float,
    default=0.5,
    help="coefficient of the value function"
  )
  parser.add_argument(
    "--max-grad-norm",
    type=float,
    default=0.5,
    help="the maximum norm for the gradient clipping"
  )
  parser.add_argument(
    "--target-kl",
    type=float,
    default=None,
    help="the target KL divergence threshold"
  )
  args = parser.parse_args()
  args.batch_size = int(args.num_envs * args.num_steps)
  args.minibatch_size = int(args.batch_size // args.num_minibatches)
  # fmt: on
  return args


class RecordEpisodeStatistics(gym.Wrapper):

  def __init__(self, env, deque_size=100):
    super(RecordEpisodeStatistics, self).__init__(env)
    self.num_envs = getattr(env, "num_envs", 1)
    self.episode_returns = None
    self.episode_lengths = None
    # get if the env has lives
    self.has_lives = False
    env.reset()
    info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
    if info["lives"].sum() > 0:
      self.has_lives = True
      print("env has lives")

  def reset(self, **kwargs):
    observations = super(RecordEpisodeStatistics, self).reset(**kwargs)
    self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    self.lives = np.zeros(self.num_envs, dtype=np.int32)
    self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    return observations

  def step(self, action):
    observations, rewards, dones, infos = super(RecordEpisodeStatistics,
                                                self).step(action)
    self.episode_returns += infos["reward"]
    self.episode_lengths += 1
    self.returned_episode_returns[:] = self.episode_returns
    self.returned_episode_lengths[:] = self.episode_lengths
    all_lives_exhausted = infos["lives"] == 0
    if self.has_lives:
      self.episode_returns *= 1 - all_lives_exhausted
      self.episode_lengths *= 1 - all_lives_exhausted
    else:
      self.episode_returns *= 1 - dones
      self.episode_lengths *= 1 - dones
    infos["r"] = self.returned_episode_returns
    infos["l"] = self.returned_episode_lengths
    return (
      observations,
      rewards,
      dones,
      infos,
    )


def layer_init(layer, std=2**0.5, bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class Agent(nn.Module):

  def __init__(self, envs):
    super(Agent, self).__init__()
    self.network = nn.Sequential(
      layer_init(nn.Conv2d(4, 32, 8, stride=4)),
      nn.ReLU(),
      layer_init(nn.Conv2d(32, 64, 4, stride=2)),
      nn.ReLU(),
      layer_init(nn.Conv2d(64, 64, 3, stride=1)),
      nn.ReLU(),
      nn.Flatten(),
      layer_init(nn.Linear(64 * 7 * 7, 512)),
      nn.ReLU(),
    )
    self.actor = layer_init(
      nn.Linear(512, envs.single_action_space.n), std=0.01
    )
    self.critic = layer_init(nn.Linear(512, 1), std=1)

  def get_value(self, x):
    return self.critic(self.network(x / 255.0))

  def get_action_and_value(self, x, action=None):
    hidden = self.network(x / 255.0)
    logits = self.actor(hidden)
    probs = Categorical(logits=logits)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
  args = parse_args()
  run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
  if args.track:
    import wandb

    wandb.init(
      project=args.wandb_project_name,
      entity=args.wandb_entity,
      sync_tensorboard=True,
      config=vars(args),
      name=run_name,
      monitor_gym=True,
      save_code=True,
    )
  writer = SummaryWriter(f"runs/{run_name}")
  writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" %
    ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )

  # TRY NOT TO MODIFY: seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic

  device = torch.device(
    "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
  )

  # env setup
  envs = envpool.make(
    args.gym_id,
    env_type="gym",
    num_envs=args.num_envs,
    episodic_life=True,
    reward_clip=True,
  )
  envs.num_envs = args.num_envs
  envs.single_action_space = envs.action_space
  envs.single_observation_space = envs.observation_space
  envs = RecordEpisodeStatistics(envs)
  assert isinstance(
    envs.action_space, gym.spaces.Discrete
  ), "only discrete action space is supported"

  agent = Agent(envs).to(device)
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

  # ALGO Logic: Storage setup
  obs = torch.zeros(
    (args.num_steps, args.num_envs) + envs.single_observation_space.shape
  ).to(device)
  actions = torch.zeros(
    (args.num_steps, args.num_envs) + envs.single_action_space.shape
  ).to(device)
  logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
  rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
  dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
  values = torch.zeros((args.num_steps, args.num_envs)).to(device)
  avg_returns = deque(maxlen=20)

  # TRY NOT TO MODIFY: start the game
  global_step = 0
  start_time = time.time()
  next_obs = torch.Tensor(envs.reset()).to(device)
  next_done = torch.zeros(args.num_envs).to(device)
  num_updates = args.total_timesteps // args.batch_size

  for update in range(1, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
      frac = 1.0 - (update - 1.0) / num_updates
      lrnow = frac * args.learning_rate
      optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
      global_step += 1 * args.num_envs
      obs[step] = next_obs
      dones[step] = next_done

      # ALGO LOGIC: action logic
      with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        values[step] = value.flatten()
      actions[step] = action
      logprobs[step] = logprob

      # TRY NOT TO MODIFY: execute the game and log data.
      next_obs, reward, done, info = envs.step(action.cpu().numpy())
      rewards[step] = torch.tensor(reward).to(device).view(-1)
      next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
        done
      ).to(device)

      for idx, d in enumerate(done):
        if d and info["lives"][idx] == 0:
          print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
          avg_returns.append(info["r"][idx])
          writer.add_scalar(
            "charts/avg_episodic_return", np.average(avg_returns), global_step
          )
          writer.add_scalar(
            "charts/episodic_return", info["r"][idx], global_step
          )
          writer.add_scalar(
            "charts/episodic_length", info["l"][idx], global_step
          )

    # bootstrap value if not done
    with torch.no_grad():
      next_value = agent.get_value(next_obs).reshape(1, -1)
      if args.gae:
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
          if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
          else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
          delta = rewards[
            t] + args.gamma * nextvalues * nextnonterminal - values[t]
          advantages[t] = lastgaelam = \
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
      else:
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(args.num_steps)):
          if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            next_return = next_value
          else:
            nextnonterminal = 1.0 - dones[t + 1]
            next_return = returns[t + 1]
          returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
        advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for _epoch in range(args.update_epochs):
      np.random.shuffle(b_inds)
      for start in range(0, args.batch_size, args.minibatch_size):
        end = start + args.minibatch_size
        mb_inds = b_inds[start:end]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
          b_obs[mb_inds],
          b_actions.long()[mb_inds]
        )
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
          # calculate approx_kl http://joschu.net/blog/kl-approx.html
          old_approx_kl = (-logratio).mean()
          approx_kl = ((ratio - 1) - logratio).mean()
          clipfracs += [
            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
          ]

        mb_advantages = b_advantages[mb_inds]
        if args.norm_adv:
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
          )

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(
          ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
          v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
          v_clipped = b_values[mb_inds] + torch.clamp(
            newvalue - b_values[mb_inds],
            -args.clip_coef,
            args.clip_coef,
          )
          v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
          v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
          v_loss = 0.5 * v_loss_max.mean()
        else:
          v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

      if args.target_kl is not None:
        if approx_kl > args.target_kl:
          break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(
      y_true - y_pred
    ) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar(
      "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar(
      "losses/old_approx_kl", old_approx_kl.item(), global_step
    )
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar(
      "charts/SPS", int(global_step / (time.time() - start_time)), global_step
    )

  envs.close()
  writer.close()
