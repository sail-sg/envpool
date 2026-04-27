# MyoSuite Metadata

This directory documents the generated metadata derived from the pinned
upstream `MyoHub/myosuite` source tree.

The source of truth is `../generate_metadata.py`, and the checked-in JSON
snapshot lives at `env_ids.json`. Regenerate the JSON after changing the
upstream pin or when validating that the local metadata still matches the
pinned official registry surface.

The vendored arm reach runtime asset lives at
`../simhive/myo_sim/arm/myoarm_reach.xml`. Regenerate it with
`../generate_arm_reach_asset.py` if the upstream arm-edit XML changes.

The checked-in snapshot includes:

- the direct and expanded upstream Env ID surfaces
- per-task entrypoint, runtime kwargs, and variant definitions
- normalized model and reference asset paths
- suite helper lists used by registration and validation code
