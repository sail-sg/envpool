# Benchmark

The following results are generated from four types of machine:

1. Personal laptop: 12 core ``Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz``
2. TPU-VM: 96 core ``Intel(R) Xeon(R) CPU @ 2.00GHz``
3. ???: 192 core ``AMD EPYC 7642 48-Core Processor``
4. DGX-A100: 256 core ``AMD EPYC 7742 64-Core Processor``

We use `Pong-v5` and `Ant-v3` for Atari/Mujoco environment benchmark test.

To align with other baseline result, FPS is multiplied with `frame_skip`.

## Testing Method and Command

### For-loop

Version to test: `gym==0.23.1`

### Subprocess (gym.vector_env)

Version to test: `gym==0.23.1`

### Sample Factory

Version to test: `sample_factory==1.123.0`

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

<!-- Atari - ??? -->

| Atari - ???          | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 96   | 128  | 160  | 192  |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |      |

<!-- Atari - ??? -->

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

<!-- Mujoco - ??? -->

| Mujoco - ???         | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 96   | 128  | 160  | 192  |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| For-loop             |      |      |      |      |      |      |      |      |      |      |      |
| Subprocess           |      |      |      |      |      |      |      |      |      |      |      |
| Sample-Factory       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (sync)       |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (async)      |      |      |      |      |      |      |      |      |      |      |      |
| EnvPool (numa+async) |      |      |      |      |      |      |      |      |      |      |      |

<!-- Mujoco - ??? -->

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
