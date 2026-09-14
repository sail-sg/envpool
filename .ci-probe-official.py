"""Replay captured native inputs using only the pinned official MuJoCo renderer."""
import sys
import ctypes
from pathlib import Path
import mujoco
import numpy as np

root = Path(sys.argv[1])

# Use exactly the pixel format from envpool/mujoco/dmc/render_oracle.py.
# The official accelerated default cannot create a context on hosted macOS.
from mujoco.cgl import cgl
attrib = cgl.CGLPixelFormatAttribute
profile = cgl.CGLOpenGLProfile
values = (
    attrib.CGLPFAOpenGLProfile, profile.CGLOGLPVersion_Legacy,
    attrib.CGLPFAColorSize, 24, attrib.CGLPFAAlphaSize, 8,
    attrib.CGLPFADepthSize, 24, attrib.CGLPFAStencilSize, 8,
    attrib.CGLPFAAllowOfflineRenderers, 0, 0,
)
offline_attribs = (ctypes.c_int * len(values))(*values)
choose_pixel_format = cgl.CGLChoosePixelFormat
cgl.CGLChoosePixelFormat = lambda ignored, pix, count: choose_pixel_format(offline_attribs, pix, count)
gl = ctypes.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')
gl.glGetString.restype = ctypes.c_char_p
reported_driver = False

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

def draw(renderer, mode):
    renderer._gl_context.make_current()
    context = renderer._mjr_context
    rect = renderer._rect
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
    mujoco.mjr_render(rect, renderer.scene, context)
    mujoco.mjr_finish()
    pixels = np.empty((64, 64, 3), dtype=np.uint8)
    if mode == 'finish_resolve' and context.offSamples:
        gl.glBindFramebuffer(0x8CA8, context.offFBO)  # GL_READ_FRAMEBUFFER
        gl.glReadBuffer(0x8CE0)  # GL_COLOR_ATTACHMENT0
        gl.glBindFramebuffer(0x8CA9, context.offFBO_r)  # GL_DRAW_FRAMEBUFFER
        gl.glDrawBuffer(0x8CE0)
        gl.glBlitFramebuffer(0, 0, 64, 64, 0, 0, 64, 64, 0x4000, 0x2600)
        gl.glFinish()
        gl.glBindFramebuffer(0x8CA8, context.offFBO_r)
        gl.glReadBuffer(0x8CE0)
        gl.glReadPixels(0, 0, 64, 64, 0x1907, 0x1401, pixels.ctypes.data_as(ctypes.c_void_p))
        mujoco.mjr_restoreBuffer(context)
    else:
        mujoco.mjr_readPixels(pixels, None, rect, context)
    error = gl.glGetError()
    if error:
        raise RuntimeError(f'OpenGL error: {error:#x}')
    return pixels[::-1].copy()

for path in sorted(root.rglob('*-left-*.mjb')):
    if 'run_1_of_3' not in str(path):
        continue
    for mode in ('native_read', 'finish_resolve', 'disable_mp'):
        renderers = []
        baseline = None
        repeated_max = np.zeros(3, dtype=int)
        contexts_max = np.zeros(3, dtype=int)
        for repeat in range(8):
            model, data = inputs(path)
            renderer = mujoco.Renderer(model, height=64, width=64)
            renderers.append(renderer)
            if not reported_driver:
                print('OpenGL driver:', [gl.glGetString(v) for v in (0x1F00, 0x1F01, 0x1F02)], flush=True)
                reported_driver = True
            if mode == 'disable_mp':
                context = ctypes.cast(renderer._gl_context._context, ctypes.c_void_p)
                enabled = ctypes.c_int()
                gl.CGLIsEnabled(context, 313, ctypes.byref(enabled))
                if repeat == 0:
                    print('CGL multiprocessor engine enabled:', enabled.value, flush=True)
                result = gl.CGLDisable(context, 313)
                if result:
                    raise RuntimeError(f'CGLDisable: {result}')
            option = mujoco.MjvOption()
            option.geomgroup[1] = 0
            option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
            renderer.update_scene(data, camera='walker/egocentric', scene_option=option)
            frames = [draw(renderer, mode) for _ in range(10)]
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
