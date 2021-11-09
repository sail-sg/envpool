Benchmark Results on Atari
==========================

The following results are generated from four types of machine:

1. Personal laptop: 12 core ``Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz``
2. TPU-VM: 96 core ``Intel(R) Xeon(R) CPU @ 2.00GHz``
3. Apollo: 96 core ``AMD EPYC 7352 24-Core Processor``
4. DGX-A100: 256 core ``AMD EPYC 7742 64-Core Processor``

FPS is based on ALE frame (Pong). Using ``envpool==0.4.0``

+----------------------+----------------------+-------------+-------------+----------------+
| Highest FPS          | i7-8750H laptop (12) | TPU-VM (96) | Apollo (96) | DGX-A100 (256) |
+======================+======================+=============+=============+================+
| For-loop             | 4876                 | 3817        | 4053        | 4336           |
+----------------------+----------------------+-------------+-------------+----------------+
| Subprocess           | 18249                | 42885       | 19560       | 79509          |
+----------------------+----------------------+-------------+-------------+----------------+
| Sample Factory       | 27035                | 192074      | 262963      | 639389         |
+----------------------+----------------------+-------------+-------------+----------------+
| EnvPool (sync)       | 40791                | 175938      | 159191      | 470170         |
+----------------------+----------------------+-------------+-------------+----------------+
| EnvPool (async)      | 50513                | 352243      | 410941      | 845537         |
+----------------------+----------------------+-------------+-------------+----------------+
| EnvPool (numa+async) | /                    | 367799      | 458414      | 1060371        |
+----------------------+----------------------+-------------+-------------+----------------+

.. image:: ../_static/images/atari_throughput_tpu.png
    :align: center

.. image:: ../_static/images/atari_throughput_apollo.png
    :align: center

.. image:: ../_static/images/atari_throughput_dgx.png
    :align: center
