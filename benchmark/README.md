# Benchmark

The following results are generated from four types of machine:

1. Personal laptop: 12 core ``Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz``
2. TPU-VM: 96 core ``Intel(R) Xeon(R) CPU @ 2.00GHz``
3. [AS4124](https://www.supermicro.org.cn/en/Aplus/system/4U/4124/AS-4124GS-TNR.cfm): 192 core ``AMD EPYC 7642 48-Core Processor``
4. DGX-A100: 256 core ``AMD EPYC 7742 64-Core Processor``

We use `Pong-v5` and `Ant-v3` for Atari/Mujoco environment benchmark test, with `ale-py==0.7.5`, `mujoco==2.1.5`, `gym==0.23.1`, `envpool==0.5.3`.

To align with other baseline result, FPS is multiplied with `frame_skip`.

## Testing Method and Command

### For-loop



### Subprocess (gym.vector_env)

Version to test: `gym==0.23.1`

### Sample Factory

Version to test: `sample_factory==1.123.0`

```bash
python3 -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --env=atari_pong --env_frameskip=4 --num_workers=384 --num_envs_per_worker=1 --sample_env_frames=64000000 --experiment=test
```

### EnvPool

#### sync



#### async



#### numa+async



### BRAX and Isaac-gym (Mujoco only)





## Result

### Atari

<!-- Atari - Laptop -->

| Atari - Laptop  | 1    | 2    | 3    | 4    | 6    | 8    | 10   | 12   |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop        |      |      |      |      |      |      |      |      |
| Subprocess      |      |      |      |      |      |      |      |      |
| Sample-Factory  |      |      |      |      |      |      |      |      |
| EnvPool (sync)  |      |      |      |      |      |      |      |      |
| EnvPool (async) |      |      |      |      |      |      |      |      |

<!-- Atari - Laptop -->

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

<!-- Atari - AS4124 -->

| Atari - AS4124       | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 96   | 128  | 160  | 192  |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |      |

<!-- Atari - AS4124 -->

<!-- Atari - DGX-A100 -->

| Atari - DGX-A100     | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 96   | 128  | 160  | 192  | 224  | 256  |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |      |      |      |

<!-- Atari - DGX-A100 -->

### Mujoco

<!-- Mujoco - Laptop -->

| Mujoco - Laptop      | 1    | 2    | 3    | 4    | 6    | 8    | 10   | 12   |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |

<!-- Mujoco - Laptop -->

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

<!-- Mujoco - AS4124 -->

| Mujoco - AS4124      | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 96   | 128  | 160  | 192  |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |      |

<!-- Mujoco - AS4124 -->

<!-- Mujoco - DGX-A100 -->

| Mujoco - DGX-A100    | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 96   | 128  | 160  | 192  | 224  | 256  |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |      |      |      |

<!-- Mujoco - DGX-A100 -->
