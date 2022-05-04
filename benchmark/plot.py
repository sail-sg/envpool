#!/usr/bin/env python3
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

import argparse

import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

data = {}


def reset_data() -> None:
  global data
  data = {
    "Num. Workers": [],
    "FPS": [],
    "Env": [],
    "System": [],
    "Method": [],
  }


def parse_table(env: str, system: str, suffix: str) -> None:
  private_copy = {
    "Num. Workers": [],
    "FPS": [],
    "Env": [],
    "System": [],
    "Method": [],
  }
  sep = f"<!-- {env} - {system} -->"
  raw = open("README.md").read().split(sep)[1].strip().splitlines()
  worker_num = list(map(int, raw[0].split("|")[2:-1]))
  for line in raw[2:]:
    line = line.split("|")[1:-1]
    method = line.pop(0).strip()
    for w, f in zip(worker_num, line):
      for d in [data, private_copy]:
        d["Num. Workers"].append(w)
        d["FPS"].append(None if f.strip() == "/" else float(f))
        d["Env"].append(env)
        d["System"].append(system)
        d["Method"].append(method)
  d = pd.DataFrame(private_copy)
  plot = sns.lineplot(
    x="Num. Workers", y="FPS", hue="Method", data=d, marker="o"
  )
  plot.xaxis.set_major_formatter(ticker.EngFormatter())
  plot.yaxis.set_major_formatter(ticker.EngFormatter())
  plot.legend(fontsize=9)
  plot.set_title(f"{env} throughput, {system}")
  plot.set_xlabel("Num. Workers")
  frame_skip = {"Atari": 4, "Mujoco": 5}[env]
  plot.set_ylabel(f"FPS, frameskip = {frame_skip}")
  plot.get_figure().savefig(f"{env}_{system}.{suffix}")
  plot.get_figure().clf()


def benchmark(suffix: str) -> None:
  global data
  reset_data()
  for env in ["Atari", "Mujoco"]:
    for system in ["Laptop", "Workstation", "TPU-VM", "DGX-A100"]:
      parse_table(env, system, suffix)
  data = pd.DataFrame(data)
  print(data.groupby(["Env", "Method", "System"]).max())

  def mapping(x, y, **kwargs):
    plot = sns.lineplot(x=x, y=y, **kwargs)
    plot.xaxis.set_major_formatter(ticker.EngFormatter())
    plot.yaxis.set_major_formatter(ticker.EngFormatter())

  g = sns.FacetGrid(
    data,
    row="Env",
    col="System",
    hue="Method",
    height=3,
    aspect=1.6,
    sharex=False,
    sharey=False,
  )
  g.map(mapping, "Num. Workers", "FPS", marker="o")
  g.add_legend(bbox_to_anchor=(0.52, 1.02), ncol=6)
  axes = g.axes.flatten()
  alphabet = "abcdefgh"
  for ax, i in zip(axes, alphabet):
    env, system = ax.get_title().split("|")
    env = env.split("=")[-1].strip()
    system = system.split("=")[-1].strip()
    ax.set_title(f"({i}) {env} throughput, {system}")
    if ax.get_ylabel():
      frame_skip = {"Atari": 4, "Mujoco": 5}[env]
      ax.set_ylabel(f"FPS, frameskip = {frame_skip}")
  g.savefig(f"throughput.{suffix}")


if __name__ == "__main__":
  pd.options.display.float_format = '{:,.0f}'.format
  parser = argparse.ArgumentParser()
  parser.add_argument("--suffix", type=str, default="png")
  args = parser.parse_args()
  benchmark(args.suffix)
