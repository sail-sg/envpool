import shlex
import subprocess
import time

WB = True
NUM_FRAME = 10_000_000
GAMES = [
  "Walker2d-v4", "Hopper-v4", "InvertedDoublePendulum-v4",
  "InvertedPendulum-v4", "Swimmer-v4", "Ant-v4"
]
GAMES = ["HalfCheetah-v3"]

for game in GAMES:
  for num_env in [4, 8, 32]:
    for batch_method in ["--use-envpool", "--use-vec-env"]:
      if batch_method == "--use-vec-env" and num_env != 32:
        continue
      for seed in [11, 12, 13, 14, 15, 16]:
        cmd = (
          f"python ppo_continuous.py --env-name {game} "
          f"--seed {seed} --num-envs {num_env} "
          f"--num-steps {NUM_FRAME} "
        )
        cmd += f"{batch_method} "
        if WB:
          cmd += "--use-wb --wb-entity openrlbenchmark "
        args = shlex.split(cmd)
        print(args)
        p = subprocess.Popen(args)
        while p.poll() is None:
          time.sleep(5)

# python ppo_continuous.py --num-envs 16 --use-envpool --use-wb --wb-entity openrlbenchmark --num-steps 10000000
