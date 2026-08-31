# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build-time CPU physics export; never imported by the EnvPool runtime."""

import dataclasses
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import mujoco_warp as mjw
import warp as wp
from warp._src.apic.capture import APICapture


@wp.kernel(enable_backward=False)
def _fill(dst: wp.array(dtype=Any), value: Any):
    dst[wp.tid()] = value


def arrays(value: Any, prefix: str) -> Iterator[tuple[str, Any]]:
    """Walk nonempty Warp arrays with their upstream dataclass field names."""
    if isinstance(value, wp.array):
        if value.size:
            yield prefix, value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from arrays(
                getattr(value, field.name), prefix + "." + field.name
            )


@contextmanager
def _export_adapter(scene_bvh: int) -> Iterator[None]:
    """Adapt APIC bookkeeping, leaving all official physics kernels intact.

    Warp 1.14 records closures by their unqualified name and does not record
    nonzero CPU array fills. Those omissions affect MJWarp's specialized solver
    kernels and loop counters. Record unique symbols and ordinary fill kernels.
    The resulting graph is for serialization, not live Python graph replay.

    Scene BVHs are external native resources. Mark their scalar kernel argument
    as a relocatable handle; the native loader supplies handle 1. Mesh/texture
    arrays are rebuilt separately. No export-process address is a runtime input.
    """
    build_info = APICapture.build_launch_info
    pack_record = APICapture._pack_param_record
    fill_array, zero_array, refit = wp.array.fill_, wp.array.zero_, wp.Bvh.refit

    class KernelIdentity:
        def __init__(self, kernel: Any, module: Any):
            self.original = kernel
            self.key = (
                module.module_hash.hex() + ":" + kernel.get_mangled_name()
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self.original, name)

    def build(self: Any, kernel: Any, module: Any, *args: Any, **kwargs: Any):
        return build_info(
            self, KernelIdentity(kernel, module), module, *args, **kwargs
        )

    def pack(
        self: Any,
        record: Any,
        arg: Any,
        value: Any,
        packed: Any,
        *args: Any,
        **kwargs: Any,
    ):
        if scene_bvh and arg.type is wp.uint64 and int(value) == scene_bvh:
            arg, value, packed = (
                SimpleNamespace(type=wp.handle),
                1,
                type(packed)(1),
            )
        return pack_record(self, record, arg, value, packed, *args, **kwargs)

    def fill(array: Any, value: Any):
        if not array.size:
            return
        wp.launch(
            _fill,
            dim=array.size,
            inputs=[array.reshape((-1,)), array.dtype(value)],
            device=array.device,
        )

    def zero(array: Any):
        if array.size:
            zero_array(array)

    APICapture.build_launch_info = build
    APICapture._pack_param_record = pack
    wp.array.fill_, wp.array.zero_ = fill, zero
    # Native refit runs between the bounds and sensing operations. It is not
    # recordable by CPU APIC and must not mutate the oracle during export.
    wp.Bvh.refit = lambda self: None
    try:
        yield
    finally:
        APICapture.build_launch_info = build_info
        APICapture._pack_param_record = pack_record
        wp.array.fill_, wp.array.zero_, wp.Bvh.refit = (
            fill_array,
            zero_array,
            refit,
        )


def export_physics(
    sim: Any, output: Path, scene_bounds: Any = None
) -> dict[str, Any]:
    """Export one official model with native operations and resource metadata."""
    if wp.config.version != "1.14.0":
        raise ValueError("the exporter requires pinned Warp 1.14.0")
    if sim.num_envs != 1 or sim.wp_model.nflex:
        raise ValueError("built-in native presets require one rigid-body world")
    output.parent.mkdir(parents=True, exist_ok=True)
    model, data, context = sim.wp_model, sim.wp_data, sim._sensor_context
    bindings = dict(arrays(model, "model")) | dict(arrays(data, "data"))
    resources: list[dict[str, Any]] = []
    opaque = []
    rc = context.render_context if context is not None else None
    if rc is not None:
        if scene_bounds is None:
            raise ValueError("preserve scene BVH bounds before the first reset")
        bindings["resource.scene.lower"], bindings["resource.scene.upper"] = (
            scene_bounds
        )
        bindings.update(arrays(rc, "camera"))
        resources.append({"kind": "bvh", "handle": 1, "count": rc.lower.size})
        for kind, registry, ids in (
            ("mesh", rc.mesh_registry, rc.mesh_bvh_id),
            ("hfield", rc.hfield_registry, rc.hfield_bvh_id),
        ):
            if not ids.size:
                continue
            opaque.append(ids)
            for index, handle in enumerate(ids.numpy()):
                if not handle:
                    continue
                mesh = registry[int(handle)]
                name = f"resource.{kind}.{index}"
                bindings[name + ".points"] = mesh.points
                bindings[name + ".indices"] = mesh.indices
                resources.append({
                    "kind": kind,
                    "index": index,
                    "binding": name,
                    "leaf_size": 2,
                })
        if rc.textures.size:
            opaque.append(rc.textures)
        for index, texture in enumerate(rc.textures_registry):
            if texture._dtype is not wp.float32 or texture._num_channels != 4:
                raise ValueError("unexpected upstream camera texture format")
            pixels = wp.empty(
                (texture._height, texture._width, 4),
                dtype=wp.float32,
                device="cpu",
            )
            texture.copy_to(pixels)
            name = f"resource.texture.{index}"
            bindings[name] = pixels
            resources.append({
                "kind": "texture",
                "index": index,
                "binding": name,
                "width": texture._width,
                "height": texture._height,
                "channels": 4,
                "dtype": texture._dtype_code,
                "filter": int(texture._filter_mode),
                "address": [
                    int(texture._address_mode_u),
                    int(texture._address_mode_v),
                    int(texture._address_mode_w),
                ],
                "normalized": bool(texture._normalized_coords),
            })
        if context._rgb_unpacked is not None:
            bindings["camera.rgb"] = context._rgb_unpacked
        for sensor in context.raycast_sensors:
            for name, value in vars(sensor).items():
                if isinstance(value, wp.array) and value.size:
                    bindings[f"ray.{sensor.cfg.name}.{name}"] = value

    operations = {
        "step": lambda: mjw.step(model, data),
        "forward": lambda: mjw.forward(model, data),
        "reset": lambda: mjw.reset_data(model, data),
        "set_const": lambda: mjw.set_const(model, data),
        "set_const_0": lambda: mjw.set_const_0(model, data),
        "set_const_fixed": lambda: mjw.set_const_fixed(model, data),
    }
    if rc is not None:
        operations["bounds"] = lambda: mjw.refit_bvh(model, data, rc)

        def sense():
            if context.has_cameras:
                mjw.render(model, data, rc)
                context.unpack_rgb()
            for sensor in context.raycast_sensors:
                sensor.raycast_kernel(rc=rc)

        operations["sense"] = sense
    flags = {
        name: wp.zeros(1, dtype=wp.int32, device="cpu") for name in operations
    }
    bindings.update({"op." + name: flag for name, flag in flags.items()})
    # Some low-level bookkeeping launches use the current device rather than
    # inferring it from model arrays. Keep capture on CPU on GPU-equipped hosts.
    with (
        wp.ScopedDevice("cpu"),
        _export_adapter(rc.bvh.id if rc is not None else 0),
    ):
        with wp.ScopedCapture(device="cpu", apic=True) as capture:
            for name, operation in operations.items():
                wp.capture_if(flags[name], on_true=operation)
    graph = capture.graph
    recorder = graph._apic_capture
    if recorder.collected_mesh_ids - {1}:
        raise ValueError("unexpected opaque resource in built-in physics graph")
    recorder.collected_mesh_ids.clear()
    for value in bindings.values():
        recorder.track_array(value)
    # These are filled by the native loader. Restore the oracle buffers after
    # saving so subsequent reference rollouts use the untouched official state.
    backups = [wp.clone(value) for value in opaque]
    try:
        for value in opaque:
            value.zero_()
        wp.capture_save(graph, str(output), inputs=bindings)
    finally:
        for value, backup in zip(opaque, backups, strict=True):
            wp.copy(value, backup)
    return {
        "bindings": bindings,
        "resources": resources,
        "kernels": recorder.collected_kernels,
        "modules": {
            name: {k: v for k, v in info.items() if k != "module_exec"}
            for name, info in recorder.collected_modules.items()
        },
        "operations": list(operations),
    }
