#!/usr/bin/env python3

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


def parse_table(env: str, system: str) -> None:
  sep = f"<!-- {env} - {system} -->"
  raw = open("README.md").read().split(sep)[1].strip().splitlines()
  worker_num = list(map(int, raw[0].split("|")[2:-1]))
  for line in raw[2:]:
    line = line.split("|")[1:-1]
    method = line.pop(0).strip()
    for w, f in zip(worker_num, line):
      data["Num. Workers"].append(w)
      data["FPS"].append(None if f.strip() == "/" else float(f))
      data["Env"].append(env)
      data["System"].append(system)
      data["Method"].append(method)


def benchmark(filename: str) -> None:
  global data
  reset_data()
  for env in ["Atari", "Mujoco"]:
    for system in ["Laptop", "Workstation", "TPU-VM", "DGX-A100"]:
      parse_table(env, system)
  data = pd.DataFrame(data)
  print(data.groupby(["Env", "Method", "System"]).max())

  def mapping(x, y, **kwargs):
    plot = sns.lineplot(x=x, y=y, **kwargs)
    plot.xaxis.set_major_formatter(ticker.EngFormatter())
    plot.yaxis.set_major_formatter(ticker.EngFormatter())

  g = sns.FacetGrid(data, row="Env", col="System", hue="Method",
                    height=3, aspect=1.6, sharex=False, sharey=False)
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
  g.savefig(filename)


if __name__ == "__main__":
  pd.options.display.float_format = '{:,.0f}'.format
  benchmark("throughput.png")
