# Benchmark

The following results are generated from four types of machine:

1. Personal laptop: 12 core ``Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz``
2. Personal workstation: 32 core ``AMD Ryzen 9 5950X 16-Core Processor``
3. TPU-VM: 96 core ``Intel(R) Xeon(R) CPU @ 2.00GHz``
4. DGX-A100: 256 core ``AMD EPYC 7742 64-Core Processor``

We use `PongNoFrameskip-v4` and `Ant-v3` for Atari/Mujoco environment benchmark test. The package version is in `requirements.txt`:

```bash
$ pip install -r requirements.txt
```

To align with other baseline results, FPS is multiplied with `frame_skip` (4 for `PongNoFrameskip-v4` and 5 for `Ant-v3`).

## Testing Method and Command

When increasing the number of envs, we also adjust the total number of steps to make each test run for about one minute.

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

We found that `num_envs_per_worker == 1` is best for all scenarios. Here's our Python script:

```python
def run_sf(w, fac=312500, frame_skip=1, task="atari_pong"):
    p = subprocess.check_output(shlex.split(f"python3 -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --env={task} --env_frameskip={frame_skip} --num_workers={w} --num_envs_per_worker=1 --sample_env_frames={fac * w} --experiment=test"), stderr=subprocess.STDOUT)
    return float([i for i in p.decode().splitlines() if "avg FPS" in i][0].split("FPS: ")[-1].split("\x1b")[0])

for i in num_workers:
    print(i, run_sf(i, frame_skip=4, task="atari_pong", fac=fac))
for i in num_workers:
    print(i, run_sf(i, frame_skip=1, task="mujoco_ant", fac=fac) * 5)
```

### EnvPool

#### sync

```bash
# atari
python3 test_envpool.py --env atari --num-envs 12 --batch-size 12
# mujoco
python3 test_envpool.py --env mujoco --num-envs 12 --batch-size 12 --total-step 200000
```

#### async

```bash
# atari
python3 test_envpool.py --env atari --num-envs 36 --batch-size 12
# mujoco
python3 test_envpool.py --env mujoco --num-envs 36 --batch-size 12 --total-step 200000
```

#### numa+async



### BRAX and Isaac-gym (Mujoco only)





## Result

### Atari

<!-- Atari - Laptop -->

| Atari - Laptop  | 1        | 2        | 3        | 4        | 6        | 8        | 10       | 12       |
| --------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| For-loop        | 4745.54  | 4796.03  | 4694.94  | 4776.76  | 4811.98  | 4892.70  | 4795.49  | 4830.31  |
| Subprocess      | 4006.04  | 7274.79  | 10028.28 | 11251.66 | 12235.83 | 13280.10 | 15863.42 | 15658.02 |
| Sample-Factory  | 5844.7   | 11148.0  | 15567.5  | 18236.7  | 25879.3  | 26695.2  | 28216.4  | 28034.7  |
| EnvPool (sync)  | 7887.51  | 14605.92 | 20288.29 | 26427.86 | 33587.28 | 28602.50 | 34311.75 | 37395.68 |
| EnvPool (async) | 10213.75 | 18880.65 | 26599.45 | 36375.89 | 48390.40 | 46921.23 | 47184.54 | 49438.56 |

<!-- Atari - Laptop -->

<!-- Atari - Workstation -->

| Atari - Workstation | 1        | 2        | 4        | 8         | 12        | 16        | 20        | 24        | 28        | 32        |
| ------------------- | -------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| For-loop            | 7739.15  | 7900.56  | 7853.82  | 7865.10   | 7914.04   | 7855.68   | 7587.67   | 7857.92   | 7635.10   | 7868.14   |
| Subprocess          | 7126.57  | 13086.18 | 23402.05 | 33733.84  | 39766.60  | 42567.05  | 30384.52  | 37224.14  | 46132.40  | 47699.40  |
| Sample-Factory      | 9259.5   | 18429.2  | 36776.8  | 71435.0   | 101555.5  | 106382.5  | 127522.5  | 131653.0  | 136605.7  | 138847.2  |
| EnvPool (sync)      | 12623.93 | 23416.68 | 44527.99 | 78612.10  | 105459.54 | 126382.48 | 106088.13 | 117524.07 | 127986.00 | 133824.37 |
| EnvPool (async)     | 14577.17 | 28383.39 | 55106.44 | 106992.10 | 153258.47 | 188554.16 | 192034.45 | 196540.73 | 200427.90 | 199684.50 |

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

| Atari - DGX-A100     | 1       | 2       | 4       | 8       | 16      | 32       | 64       | 96       | 128      | 160      | 192      | 224      | 256      |
| -------------------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| For-loop             | 4449.38 | 4587.37 | 4620.44 | 4635.26 | 4617.21 | 4639.16  | 4618.30  | 4594.96  | 4629.90  | 4616.15  | 4640.20  | 4596.57  | 4620.50  |
| Subprocess           |         |         |         |         |         |          |          |          |          |          |          |          |          |
| Sample-Factory       | 5563.2  | 11003.0 | 21976.3 | 43891.1 | 87702.0 | 175408.8 | 350855.5 | 476048.4 | 505494.8 | 616958.7 | 651428.8 | 679186.5 | 707494.3 |
| EnvPool (sync)       |         |         |         |         |         |          |          |          |          |          |          |          |          |
| EnvPool (async)      |         |         |         |         |         |          |          |          |          |          |          |          |          |
| EnvPool (numa+async) |         |         |         |         |         |          |          |          |          |          |          |          |          |

<!-- Atari - DGX-A100 -->

### Mujoco

<!-- Mujoco - Laptop -->

| Mujoco - Laptop | 1        | 2        | 3        | 4        | 6        | 8         | 10        | 12        |
| --------------- | -------- | -------- | -------- | -------- | -------- | --------- | --------- | --------- |
| For-loop        | 12325.95 | 12453.54 | 12861.30 | 12517.09 | 12467.92 | 12447.57  | 12631.33  | 12576.39  |
| Subprocess      | 8377.65  | 14851.20 | 18479.33 | 23137.12 | 26667.67 | 29260.77  | 36586.01  | 31952.74  |
| Sample-Factory  | 13270.0  | 25452.0  | 34882.0  | 41666.5  | 58892.0  | 60657.5   | 62509.5   | 60847.0   |
| EnvPool (sync)  | 15641.44 | 30409.65 | 40063.78 | 43126.54 | 58395.28 | 53269.71  | 63424.83  | 66622.24  |
| EnvPool (async) | 20922.70 | 41279.93 | 57362.56 | 73119.43 | 95542.45 | 105126.36 | 100771.24 | 101603.31 |

<!-- Mujoco - Laptop -->

<!-- Mujoco - Workstation -->

| Mujoco - Workstation | 1        | 2        | 4         | 8         | 12        | 16        | 20        | 24        | 28        | 32        |
| -------------------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| For-loop             | 19472.04 | 19251.41 | 19902.03  | 20076.99  | 19959.82  | 19513.40  | 19460.23  | 19724.42  | 20297.76  | 19797.03  |
| Subprocess           | 14428.85 | 26943.13 | 48700.27  | 71303.02  | 89901.77  | 102833.40 | 93676.48  | 97473.05  | 105432.15 | 102533.10 |
| Sample-Factory       | 20854.0  | 40113.5  | 78408.5   | 156563.0  | 225075.0  | 268005.5  | 284237.5  | 296082.5  | 305235.0  | 309264.5  |
| EnvPool (sync)       | 25725.25 | 50531.72 | 90808.85  | 180372.40 | 212389.98 | 309341.24 | 282954.27 | 326454.83 | 357376.48 | 380950.25 |
| EnvPool (async)      | 34500.65 | 68382.03 | 133496.84 | 265710.65 | 383015.28 | 478845.88 | 511142.63 | 538558.16 | 566014.54 | 582445.50 |

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

| Mujoco - DGX-A100    | 1        | 2        | 4        | 8        | 16       | 32       | 64       | 96       | 128       | 160       | 192       | 224       | 256        |
| -------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- | --------- | --------- | --------- | ---------- |
| For-loop             | 11018.57 | 11269.45 | 11059.39 | 11250.06 | 11505.15 | 11328.79 | 11568.72 | 11485.74 | 11245.55  | 11478.49  | 11430.16  | 11151.71  | 11199.28   |
| Subprocess           |          |          |          |          |          |          |          |          |           |           |           |           |            |
| Sample-Factory       | 11870.0  | 24602.0  | 48577.0  | 96826.5  | 193800.5 | 381208.5 | 761752.0 | 985909.0 | 1249369.5 | 1332128.5 | 1397427.5 | 1318249.0 | 1573262.0  |
| EnvPool (sync)       |          |          |          |          |          |          |          |          |           |           |           |           |            |
| EnvPool (async)      |          |          |          |          |          |          |          |          |           |           |           |           | 2331272.82 |
| EnvPool (numa+async) |          |          |          |          |          |          |          |          |           |           |           |           |            |

<!-- Mujoco - DGX-A100 -->
