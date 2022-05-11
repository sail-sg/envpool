# Benchmark

The following results are generated from four types of machine:

1. Personal laptop: 12 core ``Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz``, GTX1060
2. Personal workstation: 32 core ``AMD Ryzen 9 5950X 16-Core Processor``, 2x RTX3090
3. TPU-VM: 96 core ``Intel(R) Xeon(R) CPU @ 2.00GHz``, 2 NUMA core, TPU v3-8
4. DGX-A100: 256 core ``AMD EPYC 7742 64-Core Processor``, 8 NUMA core, 8x A100

We use `PongNoFrameskip-v4` (with environment wrappers from [OpenAI baselines](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)) and `Ant-v3` for Atari/Mujoco environment benchmark test with `envpool==0.5.3.post1`. Other packages' versions are all in `requirements.txt`:

```bash
$ pip install -r requirements.txt
```

To align with other baseline results, FPS is multiplied with `frame_skip` (4 for `PongNoFrameskip-v4` and 5 for `Ant-v3`).

## Highest FPS Overview


| Atari Highest FPS    | Laptop (12) | Workstation (32) | TPU-VM (96) | DGX-A100 (256) |
| -------------------- | ----------- | ---------------- | ----------- | -------------- |
| For-loop             | 4,893       | 7,914            | 3,993       | 4,640          |
| Subprocess           | 15,863      | 47,699           | 46,910      | 71,943         |
| Sample-Factory       | 28,216      | 138,847          | 222,327     | 707,494        |
| EnvPool (sync)       | 37,396      | 133,824          | 170,380     | 427,851        |
| EnvPool (async)      | **49,439**  | **200,428**      | 359,559     | 891,286        |
| EnvPool (numa+async) | /           | /                | **373,169** | **1,069,922**  |

| Mujoco Highest FPS   | Laptop (12) | Workstation (32) | TPU-VM (96) | DGX-A100 (256) |
| -------------------- | ----------- | ---------------- | ----------- | -------------- |
| For-loop             | 12,861      | 20,298           | 10,474      | 11,569         |
| Subprocess           | 36,586      | 105,432          | 87,403      | 163,656        |
| Sample-Factory       | 62,510      | 309,264          | 461,515     | 1,573,262      |
| EnvPool (sync)       | 66,622      | 380,950          | 296,681     | 949,787        |
| EnvPool (async)      | **105,126** | **582,446**      | 887,540     | 2,363,864      |
| EnvPool (numa+async) | /           | /                | **896,830** | **3,134,287**  |

![](../_static/images/throughput/throughput.png)

## Testing Method and Command

All of the scripts are under [benchmark/](https://github.com/sail-sg/envpool/tree/master/benchmark) folder. When increasing the number of envs, we also adjust the total number of steps to make each test run for about one minute.

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

To run with Ant-v3 in Sample Factory, add one line in `sample_factory/envs/mujoco/mujoco_utils.py`:

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
python3 -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --env=atari_pong --env_frameskip=4 --num_workers=12 --num_envs_per_worker=1 --sample_env_frames=1600000
# mujoco
python3 -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --env=mujoco_ant --env_frameskip=1 --num_workers=12 --num_envs_per_worker=1 --sample_env_frames=1000000
```

We found that `num_envs_per_worker == 1` is best for all scenarios.

<!--

```python
def run_sf(w, fac=312500, frame_skip=1, task="atari_pong"):
    cmd = f"python3 -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --env={task} --env_frameskip={frame_skip} --num_workers={w} --num_envs_per_worker=1 --sample_env_frames={fac * w}"
    p = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
    return float([i for i in p.decode().splitlines() if "avg FPS" in i][0].split("FPS: ")[-1].split("\x1b")[0])

for i in num_workers:
    print(i, run_sf(i, frame_skip=4, task="atari_pong", fac=fac))
for i in num_workers:
    print(i, run_sf(i, frame_skip=1, task="mujoco_ant", fac=fac) * 5)
```

-->

### EnvPool

<!--

```bash
for i in num_workers:
    for j in [1, 2.5, 2.6, 3, 4]:
        print(i, j)
        os.system(f"python3 test_envpool.py --env mujoco --num-envs {int(i * j)} --batch-size {int(i)} 2>/dev/null > tmp")
        os.system("grep FPS tmp")

numa_cnt = 8
for i in num_workers:
    x = i // numa_cnt
    if x == 0:
        continue
    for j in [2.5, 3, 4]:
        os.system(f"./numa_test.sh {numa_cnt} python3 test_envpool.py --env mujoco --num-envs {int(x * j)} --batch-size {x} --thread-affinity-offset -1")
        print(i, x, int(x * j), f'{sum([float([i for i in open(f"log{i}").read().splitlines() if "EnvPool FPS" in i][0].split("=")[-1]) for i in range(numa_cnt)]):.2f}')
```

-->

#### sync

```bash
# atari
python3 test_envpool.py --env atari --num-envs 12 --batch-size 12
# mujoco
python3 test_envpool.py --env mujoco --num-envs 12 --batch-size 12
```

#### async

```bash
# atari
python3 test_envpool.py --env atari --num-envs 36 --batch-size 12
# mujoco
python3 test_envpool.py --env mujoco --num-envs 36 --batch-size 12
```

#### numa+async

Use `numactl -s` to determine the number of NUMA cores.

```bash
# atari
./numa_test.sh 8 python3 test_envpool.py --env atari --num-envs 100 --batch-size 32 --thread-affinity-offset -1
# mujoco
./numa_test.sh 8 python3 test_envpool.py --env mujoco --num-envs 100 --batch-size 32 --thread-affinity-offset -1
```

### Brax and Isaac-gym (Mujoco only)

TODO

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

![](../_static/images/throughput/Atari_Laptop.png)

<!-- Atari - Workstation -->

| Atari - Workstation | 1        | 2        | 4        | 8         | 12        | 16        | 20        | 24        | 28        | 32        |
| ------------------- | -------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| For-loop            | 7739.15  | 7900.56  | 7853.82  | 7865.10   | 7914.04   | 7855.68   | 7587.67   | 7857.92   | 7635.10   | 7868.14   |
| Subprocess          | 7126.57  | 13086.18 | 23402.05 | 33733.84  | 39766.60  | 42567.05  | 30384.52  | 37224.14  | 46132.40  | 47699.40  |
| Sample-Factory      | 9259.5   | 18429.2  | 36776.8  | 71435.0   | 101555.5  | 106382.5  | 127522.5  | 131653.0  | 136605.7  | 138847.2  |
| EnvPool (sync)      | 12623.93 | 23416.68 | 44527.99 | 78612.10  | 105459.54 | 126382.48 | 106088.13 | 117524.07 | 127986.00 | 133824.37 |
| EnvPool (async)     | 14577.17 | 28383.39 | 55106.44 | 106992.10 | 153258.47 | 188554.16 | 192034.45 | 196540.73 | 200427.90 | 199684.50 |

<!-- Atari - Workstation -->

![](../_static/images/throughput/Atari_Workstation.png)

<!-- Atari - TPU-VM -->

| Atari - TPU-VM       | 1       | 2        | 4        | 8        | 16        | 24        | 32        | 48        | 64        | 80        | 96        |
| -------------------- | ------- | -------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| For-loop             | 3830.19 | 3942.33  | 3993.01  | 3987.62  | 3967.83   | 3990.12   | 3976.47   | 3986.15   | 3946.44   | 3964.18   | 3973.26   |
| Subprocess           | 3361.86 | 6586.32  | 12341.66 | 21547.19 | 34152.83  | 34864.23  | 38675.01  | 45471.75  | 41927.33  | 45893.35  | 46910.45  |
| Sample-Factory       | 4906.3  | 9751.2   | 19450.3  | 38828.2  | 76206.7   | 108471.7  | 137571.6  | 203113.6  | 210596.9  | 217512.9  | 222327.4  |
| EnvPool (sync)       | 7213.41 | 13827.95 | 27057.69 | 47143.35 | 71660.49  | 98892.99  | 123136.03 | 148110.55 | 141873.23 | 159635.70 | 170380.26 |
| EnvPool (async)      | 8836.44 | 17815.91 | 35524.72 | 69888.53 | 127106.74 | 184798.27 | 246497.85 | 352195.40 | 354203.40 | 356793.59 | 359558.61 |
| EnvPool (numa+async) | /       | 17976.26 | 35761.01 | 71967.27 | 136663.09 | 196424.25 | 253789.56 | 368680.81 | 371798.47 | 373169.33 | 362744.14 |

<!-- Atari - TPU-VM -->

![](../_static/images/throughput/Atari_TPU-VM.png)

<!-- Atari - DGX-A100 -->

| Atari - DGX-A100     | 1       | 2        | 4        | 8        | 16        | 32        | 64        | 96        | 128       | 160       | 192       | 224        | 256        |
| -------------------- | ------- | -------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------- | ---------- |
| For-loop             | 4449.38 | 4587.37  | 4620.44  | 4635.26  | 4617.21   | 4639.16   | 4618.30   | 4594.96   | 4629.90   | 4616.15   | 4640.20   | 4596.57    | 4620.50    |
| Subprocess           | 4052.06 | 7832.98  | 12460.71 | 18306.28 | 24754.34  | 33336.38  | 43208.56  | 52435.64  | 42449.85  | 32958.90  | 45312.39  | 45767.11   | 71942.74   |
| Sample-Factory       | 5563.2  | 11003.0  | 21976.3  | 43891.1  | 87702.0   | 175408.8  | 350855.5  | 476048.4  | 505494.8  | 616958.7  | 651428.8  | 679186.5   | 707494.3   |
| EnvPool (sync)       | 7723.96 | 14865.81 | 28499.79 | 52681.02 | 91970.45  | 155386.07 | 243231.45 | 304423.24 | 358549.95 | 367559.69 | 388419.70 | 427851.27  | 427395.89  |
| EnvPool (async)      | 8790.69 | 17866.75 | 36089.43 | 70749.63 | 139540.29 | 278186.45 | 451858.26 | 677504.68 | 817738.45 | 838174.97 | 881210.42 | 891286.00  | 874802.04  |
| EnvPool (numa+async) | /       | /        | /        | 70629.88 | 140528.93 | 279113.15 | 555426.41 | 762417.99 | 936443.47 | 955620.20 | 998668.02 | 1032953.80 | 1069921.98 |

<!-- Atari - DGX-A100 -->

![](../_static/images/throughput/Atari_DGX-A100.png)

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

![](../_static/images/throughput/Mujoco_Laptop.png)

<!-- Mujoco - Workstation -->

| Mujoco - Workstation | 1        | 2        | 4         | 8         | 12        | 16        | 20        | 24        | 28        | 32        |
| -------------------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| For-loop             | 19472.04 | 19251.41 | 19902.03  | 20076.99  | 19959.82  | 19513.40  | 19460.23  | 19724.42  | 20297.76  | 19797.03  |
| Subprocess           | 14428.85 | 26943.13 | 48700.27  | 71303.02  | 89901.77  | 102833.40 | 93676.48  | 97473.05  | 105432.15 | 102533.10 |
| Sample-Factory       | 20854.0  | 40113.5  | 78408.5   | 156563.0  | 225075.0  | 268005.5  | 284237.5  | 296082.5  | 305235.0  | 309264.5  |
| EnvPool (sync)       | 25725.25 | 50531.72 | 90808.85  | 180372.40 | 212389.98 | 309341.24 | 282954.27 | 326454.83 | 357376.48 | 380950.25 |
| EnvPool (async)      | 34500.65 | 68382.03 | 133496.84 | 265710.65 | 383015.28 | 478845.88 | 511142.63 | 538558.16 | 566014.54 | 582445.50 |

<!-- Mujoco - Workstation -->

![](../_static/images/throughput/Mujoco_Workstation.png)

<!-- Mujoco - TPU-VM -->

| Mujoco - TPU-VM      | 1        | 2        | 4        | 8         | 16        | 24        | 32        | 48        | 64        | 80        | 96        |
| -------------------- | -------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| For-loop             | 9960.98  | 10239.58 | 10186.08 | 10473.73  | 10201.70  | 10370.85  | 10454.78  | 10460.48  | 10455.71  | 10360.71  | 10386.68  |
| Subprocess           | 7236.32  | 13788.93 | 25054.73 | 40668.40  | 64148.06  | 60409.58  | 70747.21  | 78947.79  | 87403.16  | 79734.62  | 81964.35  |
| Sample-Factory       | 11008.0  | 21368.0  | 42730.0  | 83475.5   | 153976.0  | 222311.5  | 280664.5  | 406916.5  | 432212.0  | 449143.0  | 461515.0  |
| EnvPool (sync)       | 13706.61 | 26587.92 | 49074.86 | 92444.28  | 155288.26 | 181397.00 | 231293.39 | 283748.86 | 250586.54 | 268296.99 | 296680.68 |
| EnvPool (async)      | 18195.81 | 37359.25 | 78337.13 | 148284.57 | 259915.75 | 386448.09 | 512987.78 | 745083.58 | 801768.88 | 857586.18 | 887539.80 |
| EnvPool (numa+async) | /        | 35804.57 | 75467.72 | 147281.29 | 284323.79 | 412165.16 | 516120.17 | 755509.66 | 816405.50 | 868455.12 | 896830.21 |

<!-- Mujoco - TPU-VM -->

![](../_static/images/throughput/Mujoco_TPU-VM.png)

<!-- Mujoco - DGX-A100 -->

| Mujoco - DGX-A100    | 1        | 2        | 4        | 8         | 16        | 32        | 64         | 96         | 128        | 160        | 192        | 224        | 256        |
| -------------------- | -------- | -------- | -------- | --------- | --------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| For-loop             | 11018.57 | 11269.45 | 11059.39 | 11250.06  | 11505.15  | 11328.79  | 11568.72   | 11485.74   | 11245.55   | 11478.49   | 11430.16   | 11151.71   | 11199.28   |
| Subprocess           | 8814.10  | 17201.64 | 27106.27 | 44383.63  | 62785.60  | 83054.19  | 151352.88  | 158797.86  | 148815.92  | 116200.41  | 163656.36  | 147653.41  | 161599.97  |
| Sample-Factory       | 11870.0  | 24602.0  | 48577.0  | 96826.5   | 193800.5  | 381208.5  | 761752.0   | 985909.0   | 1249369.5  | 1332128.5  | 1397427.5  | 1318249.0  | 1573262.0  |
| EnvPool (sync)       | 16024.43 | 31899.44 | 61605.04 | 114488.28 | 228492.88 | 388624.94 | 656277.80  | 832101.96  | 949787.15  | 858298.85  | 945808.57  | 813799.36  | 849410.96  |
| EnvPool (async)      | 21177.71 | 44025.65 | 92312.35 | 176135.82 | 354006.02 | 700052.08 | 1167838.03 | 1678787.71 | 1730102.62 | 2052844.58 | 2185146.77 | 2355604.96 | 2363863.67 |
| EnvPool (numa+async) | /        | /        | /        | 170348.47 | 340269.34 | 693793.45 | 1388410.00 | 1920762.84 | 2341562.20 | 2569997.03 | 2776143.15 | 2964886.91 | 3134286.77 |

<!-- Mujoco - DGX-A100 -->

![](../_static/images/throughput/Mujoco_DGX-A100.png)
