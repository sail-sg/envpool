"""Render fixed Cartpole inputs using the pinned official MuJoCo package."""

import ctypes
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from mujoco.cgl import cgl

root = Path(sys.argv[1])
mode = sys.argv[2]
attrib, profile = cgl.CGLPixelFormatAttribute, cgl.CGLOpenGLProfile
values = (
    attrib.CGLPFAOpenGLProfile,
    profile.CGLOGLPVersion_Legacy,
    attrib.CGLPFAColorSize,
    24,
    attrib.CGLPFAAlphaSize,
    8,
    attrib.CGLPFADepthSize,
    24,
    attrib.CGLPFAStencilSize,
    8,
    attrib.CGLPFAAllowOfflineRenderers,
    0,
    0,
)
offline = (ctypes.c_int * len(values))(*values)
choose = cgl.CGLChoosePixelFormat
cgl.CGLChoosePixelFormat = lambda ignored, pix, count: choose(
    offline, pix, count
)
gl = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
gl.glGetString.restype = ctypes.c_char_p
reported = False


def stats(a, b):
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return np.array(
        [delta.max(), delta.sum(), np.any(delta, axis=-1).sum()], dtype=int
    )


def inputs(path):
    model = mujoco.MjModel.from_binary_path(str(path))
    if mode == "no_msaa":
        model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    with np.load(path.with_suffix(".npz")) as state:
        viewer = json.loads(str(state["metadata"]))["viewer"]
        for key in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
            target = getattr(data, key)
            target[:] = state[key].reshape(target.shape)
        data.time = state["time"].item()
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        viewer["entity_name"] + "/" + viewer["body_name"],
    )
    assert camera.trackbodyid >= 0
    camera.fixedcamid = -1
    camera.distance, camera.elevation, camera.azimuth = (
        viewer["distance"],
        viewer["elevation"],
        viewer["azimuth"],
    )
    camera.lookat[:] = viewer["lookat"]
    option = mujoco.MjvOption()
    option.geomgroup[:] = viewer["geom_group"]
    option.sitegroup[:] = viewer["site_group"]
    return model, data, camera, option


for pattern in ("*Balance-0-left-1.mjb", "*Balance-32-left-1.mjb", "*Swingup-32-left-1.mjb", "*Swingup-97-left-1.mjb"):
  for path in sorted(root.rglob(pattern)):
    model, data, camera, option = inputs(path)
    baseline = None
    within = np.zeros(3, dtype=int)
    between = np.zeros(3, dtype=int)
    first_delta = np.zeros(3, dtype=int)
    for context in range(2):
        renderer = mujoco.Renderer(model, height=80, width=96)
        if not reported:
            print("OpenGL driver:", [gl.glGetString(v) for v in (0x1F00, 0x1F01, 0x1F02)], flush=True)
            print("model framebuffer:", renderer._mjr_context.offWidth, renderer._mjr_context.offHeight, flush=True)
            reported = True
        if mode != "model_size":
            mujoco.mjr_resizeOffscreen(96, 80, renderer._mjr_context)
        for token in (0x8D57, 0x80A8, 0x80A9):
            v=ctypes.c_int()
            gl.glGetIntegerv(token, ctypes.byref(v))
        frames=[]
        for draw in range(8):
            # Match native context bind/unbind and camera updates every draw.
            if draw:
                renderer._gl_context.make_current()
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderer._mjr_context)
            mujoco.mjv_updateCamera(model, data, camera, renderer.scene)
            renderer.update_scene(data, camera=camera, scene_option=option)
            if mode == "no_shadows":
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            mujoco.mjr_render(renderer._rect, renderer.scene, renderer._mjr_context)
            mujoco.mjr_finish()
            pixels=np.empty((80,96,3),np.uint8)
            mujoco.mjr_readPixels(pixels,None,renderer._rect,renderer._mjr_context)
            frames.append(pixels[::-1].copy())
            cgl.CGLUnlockContext(renderer._gl_context._context)
            cgl.CGLSetCurrentContext(None)
        reference=frames[4]
        for frame in frames[5:]:
            within=np.maximum(within,stats(frame,reference))
        first_delta=np.maximum(first_delta,stats(frames[0],reference))
        if baseline is None:
            baseline=reference
        else:
            between=np.maximum(between,stats(reference,baseline))
        renderer._gl_context.make_current()
        renderer.close()
    native_path=path.with_name(path.name.rsplit("-left-",1)[0]+"-frames.npz")
    with np.load(native_path) as native:
        native_delta=stats(baseline,native["actual"][0])
    print(path.name, mode, "within=",within.tolist(),"between=",between.tolist(),"first=",first_delta.tolist(),"native=",native_delta.tolist(),flush=True)
