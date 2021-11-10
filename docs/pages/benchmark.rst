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
| For-loop             | 4,876                | 3,817       | 4,053       | 4,336          |
+----------------------+----------------------+-------------+-------------+----------------+
| Subprocess           | 18,249               | 42,885      | 19,560      | 79,509         |
+----------------------+----------------------+-------------+-------------+----------------+
| Sample Factory       | 27,035               | 192,074     | 262,963     | 639,389        |
+----------------------+----------------------+-------------+-------------+----------------+
| EnvPool (sync)       | 40,791               | 175,938     | 159,191     | 470,170        |
+----------------------+----------------------+-------------+-------------+----------------+
| EnvPool (async)      | 50,513               | 352,243     | 410,941     | 845,537        |
+----------------------+----------------------+-------------+-------------+----------------+
| EnvPool (numa+async) | /                    | 367,799     | 458,414     | 1,060,371      |
+----------------------+----------------------+-------------+-------------+----------------+

.. image:: ../_static/images/atari_throughput_tpu.png
    :align: center

.. image:: ../_static/images/atari_throughput_apollo.png
    :align: center

.. image:: ../_static/images/atari_throughput_dgx.png
    :align: center
