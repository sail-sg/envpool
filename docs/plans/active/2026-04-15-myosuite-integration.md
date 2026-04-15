# MyoSuite Integration ExecPlan

Date: 2026-04-15
Owner: Codex
Status: In Progress

## Goal

Integrate the full official MyoSuite surface into EnvPool as a native C++
MuJoCo family, with:

- no Python runtime bridge for environment logic
- official upstream IDs covered end to end
- upstream version pinning
- generated registry metadata from upstream source
- deterministic, alignment, render, and registry coverage
- release-compatible runtime asset packaging

## Upstream Pin

Primary upstream:

- `MyoHub/myosuite` `v2.11.6`
- release commit: `05cb84678373f91271004f99602ebbf01e57d1a1`

Asset repositories referenced by the pinned checkout:

- `MyoHub/myo_sim` `33f3ded946f55adbdcf963c99999587aadaf975f`
- `vikashplus/furniture_sim` `c97995afb81c9e2d7325b0069f9abc9a2c74a2f0`
- `vikashplus/object_sim` `87cd8dd5a11518b94fca16bc22bb04f6836c6aa7`
- `vikashplus/MPL_sim` `58dd1abc6058e0dc06e62f13a61c36adb4916815`
- `vikashplus/YCB_sim` `46edd9c361061c5d81a82f2511d4fbf76fead569`

The `myosuite_init.py` `myo_model` fetch path is not part of the public task
surface currently referenced by the env XMLs, so it is out of scope unless a
runtime task proves it is required.

## Surface Inventory

The pinned upstream exposes:

- 35 direct `myobase` IDs
- 19 direct `myochallenge` IDs
- about 190 direct `myodm` IDs
- `myoedits` duplicates of the arm reach tasks, which should not create extra
  EnvPool public IDs
- `Sarc` / `Fati` variants for every `myo*` ID
- `Reaf` variants for every `myoHand*` ID

Generated unique public ID counts at the current pin:

- 244 direct IDs
- 358 expanded IDs after upstream variant expansion

These counts must come from generated metadata, not handwritten lists.

## Constraints

1. Preserve MyoSuite public task naming exactly.
2. Keep runtime logic native in C++.
3. Keep upstream registry coverage machine-generated from pinned source.
4. Preserve XML-relative asset resolution in runfiles so native MuJoCo loading
   can use upstream asset trees without rewriting all includes up front.
5. Do not register any public EnvPool MyoSuite IDs until the corresponding
   native runtime family is implemented and tested.

## Architecture

### 1. Third-Party Intake

Add Bazel `http_archive` pins for the MyoSuite repo plus the five asset repos.
Expose filegroups for:

- runtime env assets from `myosuite/envs/myo/assets/**`
- source files needed for registry generation
- runtime asset trees from the five submodule repos

Stage runtime files under `envpool/mujoco` so upstream XML relative includes
continue to resolve:

- `envpool/mujoco/myosuite/...`
- `envpool/mujoco/simhive/myo_sim/...`
- `envpool/mujoco/simhive/furniture_sim/...`
- `envpool/mujoco/simhive/object_sim/...`
- `envpool/mujoco/simhive/MPL_sim/...`
- `envpool/mujoco/simhive/YCB_sim/...`

### 2. Generated Metadata

Create a repository-local generator that parses the pinned upstream Python
registration files and emits canonical metadata:

- suite breakdown
- direct ID list
- expanded ID list
- object lists for `myodm`
- duplicate-overridden IDs from `myoedits`

This metadata becomes the source of truth for:

- registration coverage tests
- generated registration modules
- doc task lists

### 3. Native Runtime Decomposition

The native implementation should follow the upstream task-class shape rather
than one-class-per-ID:

- `BaseV0`-style musculoskeletal base
- `PoseEnvV0`
- `ReachEnvV0`
- `KeyTurn`
- `ObjHold`
- `PenTwirl`
- `ReorientSAR`
- `Torso`
- `Walk` / `Terrain`
- `RunTrack`
- `ChaseTag`
- `Soccer`
- `Relocate`
- `Reorient`
- `Baoding`
- `Bimanual`
- `TableTennis`
- `TrackEnv` for `myodm`

The first implementation pass should prioritize common shared machinery:

- normalized muscle actuator mapping
- fatigue / sarcopenia / reafferentation variants
- observation packing
- reward dictionary plumbing
- deterministic reset and target generation
- render camera parity
- asset path resolution

### 4. Validation

Required before public registration:

- registry completeness against generated upstream metadata
- deterministic rollout tests over every supported official ID
- official oracle alignment for every supported official ID
- render tests for render-capable tasks
- release asset path validation

## Milestones

### Milestone 1: Intake and Metadata

Deliverables:

- Bazel third-party pins
- staged runtime asset tree
- generated metadata checked into the repo
- exec plan and decision log

Proof:

- metadata generator reproduces the expected direct and expanded counts
- Bazel can build the staged MyoSuite runtime asset targets
- Bazel metadata regression test re-generates the checked-in snapshot from the
  vendored upstream source tree
- Bazel MuJoCo smoke test loads every top-level staged MyoSuite model XML that
  is intended to be consumed directly

### Milestone 2: Shared Native Base

Deliverables:

- `envpool/mujoco/myosuite/` family skeleton
- native base env and spec helpers
- generated registration scaffolding wired to metadata but not yet public

Proof:

- family compiles
- a metadata-driven smoke test can instantiate specs for implemented tasks

### Milestone 3: MyoBase

Deliverables:

- full native `myobase` coverage, including variant IDs

Proof:

- deterministic + alignment + render coverage over all `myobase` tasks

### Milestone 4: MyoChallenge

Deliverables:

- full native `myochallenge` coverage, including variant IDs where upstream
  generates them

Proof:

- deterministic + alignment + render coverage over all `myochallenge` tasks

### Milestone 5: MyoDM

Deliverables:

- full native `myodm` surface

Proof:

- completeness test over all object and reference-motion tasks
- oracle-aligned stepping and asset path validation

### Milestone 6: Docs and Release

Deliverables:

- `docs/env/myosuite.rst`
- README support matrix entry
- release asset validation

Proof:

- docs build
- release-target smoke path loads staged MyoSuite assets

## Progress

- 2026-04-15: Read EnvPool new-env and contributing docs.
- 2026-04-15: Surveyed the existing MuJoCo, MetaWorld, and
  Gymnasium-Robotics integrations for structure and test patterns.
- 2026-04-15: Pinned MyoSuite to `v2.11.6` and cloned the upstream release for
  local analysis.
- 2026-04-15: Measured the public surface and identified the five external
  asset repositories needed for runtime XML resolution.
- 2026-04-15: Started landing third-party intake and metadata generation
  infrastructure in this repo.
- 2026-04-15: Added Bazel-pinned third-party intake for `myosuite` and the
  five required asset repositories.
- 2026-04-15: Added staged `myosuite_assets` runfiles layout under
  `envpool/mujoco` and verified representative XML include resolution.
- 2026-04-15: Checked in generated MyoSuite ID metadata and added Python
  helpers to load the vendored asset root and metadata in Bazel runfiles.
- 2026-04-15: Added `myosuite_metadata_test`, which re-generates the metadata
  snapshot from the vendored upstream registry sources and compares it
  byte-for-byte with the checked-in JSON.
- 2026-04-15: Added `myosuite_asset_smoke_test`, which loads every top-level
  staged MyoSuite model XML directly through MuJoCo to catch packaging and
  include-path regressions early.

## Surprises and Discoveries

- The MyoSuite public surface is much larger than a typical MuJoCo family.
  The `myodm` suite alone contributes about 190 direct IDs.
- The upstream XMLs are designed around relative includes into a `simhive`
  tree. Preserving that structure inside runfiles is simpler and less risky
  than rewriting every XML include up front.
- Several XMLs under `envs/myo/assets/` are not direct top-level models:
  `myohand_object.xml` is a template that requires object-name substitution,
  while files such as `myosuite_track.xml`, `paddle.xml`, and nested soccer
  fragments are included by higher-level models and are not meant to be loaded
  standalone. The smoke coverage therefore targets the top-level env model
  XMLs rather than every XML file in the tree.
- `myoedits` does not add new public tasks; it overrides the arm reach tasks.

## Decision Log

- Decision: preserve upstream XML include structure in staged assets.
  Reason: this minimizes early patching and keeps the runtime path close to the
  pinned official oracle.
- Decision: generate the MyoSuite registry metadata from upstream source before
  implementing public registration.
  Reason: hand-maintaining 358 expanded IDs would be brittle and violates the
  project guidance.

## Open Questions

- Whether any upstream task needs an additional asset repo beyond the five
  current submodules once the native runtime begins loading all task classes.
- How much of the staged asset tree can be compacted safely for release
  packaging after the family is green.
