# Benchmark

The following results are generated from four types of machine:

1. Personal laptop: 12 core ``Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz``
2. Personal workstation: 32 core AMD 5950X
3. TPU-VM: 96 core ``Intel(R) Xeon(R) CPU @ 2.00GHz``
4. DGX-A100: 256 core ``AMD EPYC 7742 64-Core Processor``

We use `PongNoFrameskip-v4` and `Ant-v3` for Atari/Mujoco environment benchmark test. The package version is in `requirements.txt`:

```bash
$ pip install -r requirements.txt
```

To align with other baseline result, FPS is multiplied with `frame_skip`.

## Testing Method and Command

When increasing the number of envs, we also increased the total number of steps to make each test run for about one minute.

### For-loop

Command to run:

```bash
# atari
python3 test_gym.py --env atari --num-envs 12 --total-step 6000
# mujoco
python3 test_gym.py --env mujoco --num-envs 12 --total-step 12000
```

### Subprocess (gym.vector_env)

Command to run:

```bash
# atari
python3 test_gym.py --env atari --async_ --num-envs 10 --total-step 20000
# mujoco
python3 test_gym.py --env mujoco --async_ --num-envs 10 --total-step 50000
```

### Sample Factory

To run with Ant-v3 in SF, add one line in `sample_factory/envs/mujoco/mujoco_utils.py`:

```diff
 MUJOCO_ENVS = [ 
+    MujocoSpec('mujoco_ant', 'Ant-v3'),
     MujocoSpec('mujoco_hopper', 'Hopper-v2'),
     MujocoSpec('mujoco_halfcheetah', 'HalfCheetah-v2'),
     MujocoSpec('mujoco_humanoid', 'Humanoid-v2'),
 ]
```

and finally use FPS \* 5 as the result.

Command to run:

```bash
# atari
python3 -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --env=atari_pong --env_frameskip=4 --num_workers=12 --num_envs_per_worker=1 --sample_env_frames=1600000 --experiment=test
# mujoco
python3 -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --env=mujoco_ant --env_frameskip=1 --num_workers=12 --num_envs_per_worker=1 --sample_env_frames=1000000 --experiment=test
```

We found that `num_envs_per_worker == 1` is best for all scenarios.

### EnvPool

#### sync



#### async



#### numa+async



### BRAX and Isaac-gym (Mujoco only)





## Result

### Atari

<!-- Atari - Laptop -->

| Atari - Laptop  | 1       | 2       | 3        | 4        | 6        | 8        | 10       | 12       |
| --------------- | ------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| For-loop        | 4745.54 | 4796.03 | 4694.94  | 4776.76  | 4811.98  | 4892.70  | 4795.49  | 4830.31  |
| Subprocess      | 4006.04 | 7274.79 | 10028.28 | 11251.66 | 12235.83 | 13280.10 | 15863.42 | 15658.02 |
| Sample-Factory  | 5844.7  | 11148.0 | 15567.5  | 18236.7  | 25879.3  | 26695.2  | 28216.4  | 28034.7  |
| EnvPool (sync)  |         |         |          |          |          |          |          |          |
| EnvPool (async) |         |         |          |          |          |          |          |          |

<!-- Atari - Laptop -->

<!-- Atari - Workstation -->

| Atari - Workstation  | 1    | 2    | 4    | 8    | 12   | 16   | 20   | 24   | 28   | 32   |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |

<!-- Atari - Workstation -->

<!-- Atari - TPU-VM -->

| Atari - TPU-VM       | 1    | 2    | 4    | 8    | 16   | 24   | 32   | 48   | 64   | 80   | 96   |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |      |

<!-- Atari - TPU-VM -->

<!-- Atari - DGX-A100 -->

| Atari - DGX-A100     | 1       | 2       | 4        | 8        | 16       | 32       | 64       | 96       | 128      | 160     | 192     | 224     | 256     |
| -------------------- | ------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ------- | ------- | ------- | ------- |
| For-loop             | 4449.38 | 4587.37 | 4620.44  | 4635.26  | 4617.21  | 4639.16  | 4618.30  | 4594.96  | 4629.90  | 4616.15 | 4640.20 | 4596.57 | 4620.50 |
| Subprocess           | 4176.38 | 7873.72 | 12326.86 | 18362.00 | 30468.18 | 34045.35 | 45986.38 | 48130.98 | 37746.73 |         |         |         |         |
| Sample-Factory       |         |         |          |          |          |          |          |          |          |         |         |         |         |
| EnvPool (sync)       |         |         |          |          |          |          |          |          |          |         |         |         |         |
| EnvPool (async)      |         |         |          |          |          |          |          |          |          |         |         |         |         |
| EnvPool (numa+async) |         |         |          |          |          |          |          |          |          |         |         |         |         |

<!-- Atari - DGX-A100 -->

### Mujoco

<!-- Mujoco - Laptop -->

| Mujoco - Laptop | 1        | 2        | 3        | 4        | 6        | 8        | 10       | 12       |
| --------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| For-loop        | 12325.95 | 12453.54 | 12861.30 | 12517.09 | 12467.92 | 12447.57 | 12631.33 | 12576.39 |
| Subprocess      | 8377.65  | 14851.20 | 18479.33 | 23137.12 | 26667.67 | 29260.77 | 36586.01 | 31952.74 |
| Sample-Factory  | 13270.0  | 25452.0  | 34882.0  | 41666.5  | 58892.0  | 60657.5  | 62509.5  | 59489.0  |
| EnvPool (sync)  |          |          |          |          |          |          |          |          |
| EnvPool (async) |          |          |          |          |          |          |          |          |

<!-- Mujoco - Laptop -->

<!-- Mujoco - Workstation -->

| Mujoco - Workstation | 1    | 2    | 4    | 8    | 12   | 16   | 20   | 24   | 28   | 32   |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |

<!-- Mujoco - Workstation -->

<!-- Mujoco - TPU-VM -->

| Mujoco - TPU-VM      | 1    | 2    | 4    | 8    | 16   | 24   | 32   | 48   | 64   | 80   | 96   |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |      |

<!-- Mujoco - TPU-VM -->

<!-- Mujoco - DGX-A100 -->

| Mujoco - DGX-A100    | 1        | 2        | 4        | 8        | 16       | 32       | 64       | 96       | 128      | 160      | 192      | 224      | 256        |
| -------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ---------- |
| For-loop             | 11018.57 | 11269.45 | 11059.39 | 11250.06 | 11505.15 | 11328.79 | 11568.72 | 11485.74 | 11245.55 | 11478.49 | 11430.16 | 11151.71 | 11199.28   |
| Subprocess           |          |          |          |          |          |          |          |          |          | 82451.39 |          |          |            |
| Sample-Factory       |          |          |          |          |          |          |          |          |          |          |          |          |            |
| EnvPool (sync)       |          |          |          |          |          |          |          |          |          |          |          |          |            |
| EnvPool (async)      |          |          |          |          |          |          |          |          |          |          |          |          | 2331272.82 |
| EnvPool (numa+async) |          |          |          |          |          |          |          |          |          |          |          |          |            |

<!-- Mujoco - DGX-A100 -->
