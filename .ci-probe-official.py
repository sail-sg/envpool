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



for task in ("Mjlab-Cartpole-Balance", "Mjlab-Cartpole-Swingup"):
    parents=sorted({p.parent for p in root.rglob(task+"-0-left-0.mjb")})
    for folder in parents:
        renderers={}
        first_frames={}
        def render(side, slot, step):
            key=(side,slot)
            path=folder/f"{task}-{step}-left-{slot}.mjb"
            if key not in renderers:
                model,data,camera,option=inputs(path)
                if mode=="initial_size":
                    model.vis.global_.offwidth=96
                    model.vis.global_.offheight=80
                renderer=mujoco.Renderer(model,height=80,width=96)
                if not renderers:
                    print("OpenGL driver:",[gl.glGetString(v) for v in (0x1F00,0x1F01,0x1F02)],flush=True)
                for token in (0x8D57,0x80A8,0x80A9):
                    v=ctypes.c_int()
                    gl.glGetIntegerv(token,ctypes.byref(v))
                mujoco.mjr_resizeOffscreen(96,80,renderer._mjr_context)
                renderers[key]=(model,data,camera,option,renderer)
                draws=5
            else:
                model,data,camera,option,renderer=renderers[key]
                renderer._gl_context.make_current()
                draws=1
            with np.load(path.with_suffix(".npz")) as state:
                for name in ("qpos","qvel","ctrl","mocap_pos","mocap_quat"):
                    target=getattr(data,name)
                    target[:]=state[name].reshape(target.shape)
                data.time=state["time"].item()
            mujoco.mj_forward(model,data)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN,renderer._mjr_context)
            for _ in range(draws):
                mujoco.mjv_updateCamera(model,data,camera,renderer.scene)
                renderer.update_scene(data,camera=camera,scene_option=option)
                if mode=="no_shadows":renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
                mujoco.mjr_render(renderer._rect,renderer.scene,renderer._mjr_context)
                mujoco.mjr_finish()
                pixels=np.empty((80,96,3),np.uint8)
                mujoco.mjr_readPixels(pixels,None,renderer._rect,renderer._mjr_context)
            cgl.CGLUnlockContext(renderer._gl_context._context)
            cgl.CGLSetCurrentContext(None)
            return pixels[::-1].copy()
        for step in (0,1,32,97,194):
            left=np.stack([render("left",slot,step) for slot in (1,0)])
            right=np.stack([render("right",slot,step) for slot in (1,0)])
            replay=stats(left,right)
            selection=stats(left[0],render("left",1,step))
            repeat=stats(left,np.stack([render("left",slot,step) for slot in (1,0)]))
            print(task,step,mode,"worker_replay=",replay.tolist(),"selection=",selection.tolist(),"repeat=",repeat.tolist(),flush=True)
        for model,data,camera,option,renderer in renderers.values():
            renderer._gl_context.make_current()
            renderer.close()
