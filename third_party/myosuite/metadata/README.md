# MyoSuite Metadata

This directory stores generated metadata derived from the pinned upstream
`MyoHub/myosuite` source tree.

The source of truth is `../generate_metadata.py`. Regenerate the JSON after
changing the upstream pin or when validating that the local metadata still
matches the pinned official registry surface.

The checked-in snapshot includes:

- the direct and expanded upstream Env ID surfaces
- per-task entrypoint, runtime kwargs, and variant definitions
- normalized model and reference asset paths
- suite helper lists used by registration and validation code
