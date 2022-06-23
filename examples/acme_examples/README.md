# [Acme](https://github.com/deepmind/acme)

Acme has only released v0.4.0 on PyPI for now (22/06/03), which is far behind
the master codes, where APIs for constructing experiments were added.

We are using a specific master version (84e0923), so please make sure you
install acme using [method 4](https://github.com/deepmind/acme#installation).

To setup the environment: `pip install -r requirements.txt`

To run the experiments:

```bash
python examples/acme_examples/ppo_continuous.py --use-envpool
```

The benchmark training curve: see [openrlbenchmark](https://wandb.ai/openrlbenchmark/acme).
