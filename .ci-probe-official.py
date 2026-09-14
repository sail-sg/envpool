"""Replay captured native inputs using only the pinned official MuJoCo renderer."""
import sys
from pathlib import Path
import mujoco
import numpy as np

root = Path(sys.argv[1])

def inputs(path):
    model = mujoco.MjModel.from_binary_path(str(path))
    data = mujoco.MjData(model)
    with np.load(path.with_suffix('.npz')) as state:
        for key in ('qpos', 'qvel', 'act', 'ctrl'):
            getattr(data, key)[:] = state[key]
        data.qacc_warmstart[:] = state['warmstart']
        data.time = state['time'].item()
    mujoco.mj_forward(model, data)
    return model, data

def stats(actual, expected):
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    return int(delta.max()), int(delta.sum()), int(np.any(delta, axis=-1).sum())

for path in sorted(root.rglob('*-left-*.mjb')):
    if 'run_1_of_3' not in str(path):
        continue
    for mode in ('default', 'no_msaa', 'no_shadows', 'no_dither'):
        renderers = []
        baseline = None
        repeated_max = np.zeros(3, dtype=int)
        contexts_max = np.zeros(3, dtype=int)
        for repeat in range(12):
            model, data = inputs(path)
            if mode == 'no_msaa':
                model.vis.quality.offsamples = 0
            renderer = mujoco.Renderer(model, height=64, width=64)
            renderers.append(renderer)
            option = mujoco.MjvOption()
            option.geomgroup[1] = 0
            option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
            renderer.update_scene(data, camera='walker/egocentric', scene_option=option)
            if mode == 'no_shadows':
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            if mode == 'no_dither':
                import ctypes
                renderer._gl_context.make_current()
                gl = ctypes.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')
                gl.glDisable(0x0BD0)  # GL_DITHER
            frames = [renderer.render().copy() for _ in range(12)]
            reference = frames[4]
            for frame in frames[5:]:
                repeated_max = np.maximum(repeated_max, stats(frame, reference))
            if baseline is None:
                baseline = reference
            else:
                contexts_max = np.maximum(contexts_max, stats(reference, baseline))
        for renderer in reversed(renderers):
            renderer.close()
        print(path.name, mode, 'within_context_peak_sum_pixels=', repeated_max.tolist(), 'between_contexts=', contexts_max.tolist(), flush=True)
