# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test Vizdoom env by well-trained RL agents."""

import gc
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import torch
from absl import logging
from absl.testing import absltest
from tianshou.data import Batch
from tianshou.policy import C51Policy

import envpool.vizdoom.registration  # noqa: F401
from envpool.atari.atari_network import C51
from envpool.registration import make_gym

# try:
#   import cv2
# except ImportError:
#   cv2 = None


_PACKAGE_DIR = os.path.dirname(__file__)


def _get_map_path(path: str) -> str:
    return os.path.join(_PACKAGE_DIR, "maps", path)


def _get_package_path(path: str) -> str:
    return os.path.join(_PACKAGE_DIR, path)


def _cleanup_runtime_dir() -> None:
    if os.path.isdir("_vizdoom"):
        shutil.rmtree("_vizdoom")
    elif os.path.exists("_vizdoom"):
        os.remove("_vizdoom")


@contextmanager
def _temporary_workdir() -> Iterator[None]:
    prev_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="vizdoom-runtime-") as tempdir:
        os.chdir(tempdir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


def _eval_c51_impl(
    task: str,
    resume_path: str,
    num_envs: int = 10,
    seed: int = 0,
    cfg_path: str | None = None,
    reward_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    task_id = "".join([g.capitalize() for g in task.split("_")]) + "-v1"
    if cfg_path is None:
        cfg_path = _get_map_path(task + ".cfg")
    else:
        cfg_path = os.path.abspath(cfg_path)
    resume_path = os.path.abspath(resume_path)
    kwargs = {
        "num_envs": num_envs,
        "seed": seed,
        "wad_path": _get_map_path(task + ".wad"),
        "cfg_path": cfg_path,
        "use_combined_action": True,
    }
    if reward_config is not None:
        kwargs.update(reward_config=reward_config)
    with _temporary_workdir():
        _cleanup_runtime_dir()
        env = make_gym(task_id, **kwargs)
        try:
            state_shape = env.observation_space.shape
            action_shape = env.action_space.n
            device = "cuda" if torch.cuda.is_available() else "cpu"
            np.random.seed(seed)
            torch.manual_seed(seed)
            logging.info(state_shape)
            net = C51(*state_shape, action_shape, 51, device)  # type: ignore
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)

            policy = C51Policy(
                net, optim, 0.99, 51, -10, 10, 3, target_update_freq=500
            ).to(device)
            policy.load_state_dict(torch.load(resume_path, map_location=device))
            policy.eval()
            ids = cast(Any, np.arange(num_envs))
            reward = np.zeros(num_envs)
            length = np.zeros(num_envs)
            obs, _ = env.reset()
            for _ in range(555):
                if np.random.rand() < 0.05:
                    act = np.random.randint(action_shape, size=len(ids))
                else:
                    act = policy(Batch(obs=obs, info={})).act
                obs, rew, terminated, truncated, info = env.step(act, ids)
                done = np.logical_or(terminated, truncated)
                ids = cast(Any, np.asarray(info["env_id"]))
                reward[ids] += rew
                length[ids] += 1
                obs = obs[~done]
                ids = ids[~done]
                if len(ids) == 0:
                    break

            logging.info(
                f"Mean reward of {task}: {reward.mean()} ± {reward.std()}"
            )
            logging.info(
                f"Mean length of {task}: {length.mean()} ± {length.std()}"
            )
            return reward, length
        finally:
            env.close()
            del env
            gc.collect()
            _cleanup_runtime_dir()


def _eval_c51_subprocess(
    result_queue: mp.Queue,
    task: str,
    resume_path: str,
    cfg_path: str | None,
    reward_config: dict | None,
) -> None:
    reward, length = _eval_c51_impl(
        task,
        resume_path,
        cfg_path=cfg_path,
        reward_config=reward_config,
    )
    result_queue.put((reward, length))


class _VizdoomPretrainTest(absltest.TestCase):
    def get_map_path(self, path: str) -> str:
        return _get_map_path(path)

    def get_package_path(self, path: str) -> str:
        return _get_package_path(path)

    def eval_c51(
        self,
        task: str,
        resume_path: str,
        num_envs: int = 10,
        seed: int = 0,
        cfg_path: str | None = None,
        reward_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _eval_c51_impl(
            task,
            resume_path,
            num_envs=num_envs,
            seed=seed,
            cfg_path=cfg_path,
            reward_config=reward_config,
        )

    def eval_c51_subprocess(
        self,
        task: str,
        resume_path: str,
        cfg_path: str | None = None,
        reward_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_eval_c51_subprocess,
            args=(result_queue, task, resume_path, cfg_path, reward_config),
        )
        proc.start()
        proc.join(timeout=360)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            self.fail(f"Timed out waiting for {task} subprocess result")
        self.assertEqual(proc.exitcode, 0)
        try:
            reward, length = result_queue.get_nowait()
        except queue.Empty:
            self.fail(f"{task} subprocess exited without producing a result")
        result_queue.close()
        result_queue.join_thread()
        return reward, length

    def test_d1(self) -> None:
        model_path = self.get_package_path("policy-d1.pth")
        self.assertTrue(os.path.exists(model_path))
        _, length = self.eval_c51_subprocess("D1_basic", model_path)
        self.assertGreaterEqual(length.mean(), 500)

    def test_d3(self) -> None:
        model_path = self.get_package_path("policy-d3.pth")
        self.assertTrue(os.path.exists(model_path))
        reward_config = {"KILLCOUNT": [1, 0]}
        baseline_reward, baseline_length = self.eval_c51_subprocess(
            "D3_battle",
            model_path,
            reward_config=reward_config,
        )
        # test with customized config
        with open(self.get_map_path("D3_battle.cfg")) as f:
            cfg = f.read()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cfg", prefix="d3-", delete=False
        ) as f:
            custom_cfg_path = f.name
            f.write("# custom cfg path smoke test\n" + cfg)
        try:
            reward, length = self.eval_c51_subprocess(
                "D3_battle",
                model_path,
                cfg_path=custom_cfg_path,
                reward_config=reward_config,
            )
        finally:
            if os.path.exists(custom_cfg_path):
                os.remove(custom_cfg_path)
        np.testing.assert_allclose(
            reward.mean(),
            baseline_reward.mean(),
            rtol=0.25,
            atol=2.0,
        )
        np.testing.assert_allclose(
            length.mean(),
            baseline_length.mean(),
            rtol=0.1,
            atol=30.0,
        )


if __name__ == "__main__":
    absltest.main()
