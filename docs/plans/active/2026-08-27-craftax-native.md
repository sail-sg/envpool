# Add native Craftax environments

This is the living implementation and acceptance record for the Craftax port.

## Purpose / Big Picture

Run all four Craftax games (Classic/full game, symbolic/pixel observations)
through EnvPool's C++ thread pool without importing Craftax or JAX at runtime.
Preserve the pinned upstream rules, observations, rewards, termination, exposed
information, and renderer. Include the upstream AutoReset names, with their
reset timing explicitly implemented and tested, and EnvPool aliases.

## Progress

- [x] 2026-08-27: Read the EnvPool contribution and new-environment guides.
- [x] 2026-08-27: Created `jiayi/craftax-native` in an isolated worktree based
  on main commit `35b3282`.
- [x] 2026-08-27: Pinned Craftax v1.6.1, commit
  `c3c2e0d038c4e641f9481320c158f457f30c28f3`; inspected the factory, state
  definitions, MIT license, and packaged PNG assets.
- [x] Audited all eight factory names, established the pinned oracle, and passed
  the existing dummy pool baseline.
- [x] Implemented native state, Threefry streams, Classic and nine-level world
  generation, and both games; broad acceptance remains in progress.
- [x] Implemented native symbolic/pixel observations, embedded PNG textures,
  RGB rendering, four bindings, and both reset modes.
- [x] Passed all eight public names' 800-step oracle trajectories, all-name
  initial-state/64-pixel render checks, god-mode boundary trajectories, and
  99 directed game trajectories on macOS arm64.
- [x] Cover the complete surface with deterministic, oracle, and render tests.
- [x] Finish registration, package data, documentation, and comparison images.
- [x] Pass local source, static, documentation, release build, and installed
  wheel checks on macOS arm64 with Python 3.12.
- [x] Open draft PR #433 at source commit `f95035b`.
- [x] Reproduce macOS full source CI: 102/102 uncached targets passed.
- [x] Reproduce the platform-specific JAX pixel FMA differences using native
  and official standalone renderers, then the identical blend operands.
- [ ] Pass all-platform source CI and release packaging CI using the existing
  macOS, Linux x86-64, native ARM64 VM, and Windows hosts. GitHub-hosted jobs
  cannot start because of the account billing lock; local runs are recorded
  separately and are not represented as successful hosted checks.

## Surprises & Discoveries

- Craftax has four primary game/observation names and four AutoReset aliases.
  AutoReset is a behavioral contract, not another game.
- The full game's default map is nine 48-by-48 levels; Classic uses a single
  64-by-64 map. Their rules and inventory representations differ.
- Random events occur during steps, so matching only the reset state is not
  enough. The native implementation must preserve the oracle's random stream
  schedule after the one permitted reset-time synchronization.

- Development probes matched complete Classic episodes of 175, 396, and 93
  steps and full-game episodes of 370, 147, and 375 steps, including every
  native state field, symbolic observations, and rewards. All six initial
  states matched the oracle without synchronization. These are development
  probes, not the complete acceptance suite.
- Full Craftax uses Euclidean distance for spawning, world proximity, and
  torch light, but Manhattan distance for several mob decisions. Classic
  uses Manhattan distance for spawning as well.
- Pillow NEAREST resizing advances a double coordinate incrementally. Integer
  center coordinates differ at ties (e.g. 16-to-7 maps the middle to 7, not 8).
- The pinned JAX compiler contracts color blends in a specific order and turns
  division of the night-intensity texture into multiplication by a reciprocal.
  With explicit native contractions, standalone float RGB agrees bitwise at
  7/10 and 16 pixel tiles for day, night, and sleep.
- JAX compilation boundaries affect final float32 bits. Full symbolic reset
  folds initial 9/10 to 0.9, while dynamic step uses 9*float32(0.1). AutoReset
  returns an unfused light state while its observation uses the fused result
  (at timestep 2: 0.81411064 versus 0.81411058). These observable distinctions
  are preserved explicitly in the native wrapper.
- Classic score needs the ordered float32 achievement reduction, reciprocal
  mean, and the pinned LLVM float32 exponential polynomial. Using `expf` or
  multiplying the achievement count by log(101) changes its last bits.
- On ARM64 (macOS and Linux), JAX changes the blue-channel final-blend FMA order
  in four-pixel blocks for Classic pixel reset/non-AutoReset step graphs.
  Reversing that operation for x < 60 reproduces both awake and sleeping
  observations exactly. Keep the native renderer independent of LLVM vector
  width: only map blue allows one ULP at reset/two at step; while sleeping,
  its grayscale propagates this residual into map red/green (two ULPs).
  Its scalar tail remains exact and is excluded from the exception.
- On x86-64 (Linux and Windows), the same FMA reversal affects Classic
  AutoReset steps' map red and full resets' map red/green in eight-pixel blocks
  (x < 104). The standalone official renderer still agrees bitwise with C++.
  Reconstructing those exact operations reproduced all 24 reset/step probe
  frames across Windows and Linux ARM64 with zero residual. Complete 21-case
  trajectory diagnostics then measured at most two ULPs for Classic steps
  (including sleeping RGB) and one ULP for full resets, with no state, reward,
  termination, information, or uint8-render tolerance. Scope the comparison
  to those regions rather than changing rendering to follow LLVM vector width.
  Sleep probes also reproduced the residual exactly after respecting the
  x86-64 green/blue scalar tail (x >= 56), which remains bitwise and is excluded.
- Broader weighted-draw probes exposed a real selection bug: a sequential
  cumulative sum differed on 83/10,000 normalized 4096-entry draws. The pinned
  XLA CPU graph recursively scans 16-entry tiles. JAX v0.11.1 pins XLA commit
  `dcf304bc5dca1932b99f740b911dbd73631a1a69`; its CPU compiler applies
  `ReduceWindowRewriter` with `xla_reduce_window_rewrite_base_length`.
  The native counterpart now
  matches dynamic float32 cumulative arrays exactly; new oracle tests protect
  both dense and masked probability distributions.
- Expensive immutable official texture caches are test-only Bazel build
  artifacts, shared across tests instead of regenerated for every shard.
- God mode can walk outside the map. Map views use the oracle's padded
  dynamic-slice clipping, while entity coordinates still follow the player.

## Decision Log

- Keep the family under `envpool/craftax/` and upstream source/asset wiring under
  `third_party/craftax/`. Do not add a Python runtime bridge or invoke an
  official Python environment from native code.
- Use Craftax v1.6.1 throughout. Pin JAX with the oracle requirements and record
  its random-number configuration. Do not silently compare different versions.
- Reuse the existing EnvPool rendering/binding/registration mechanisms and
  existing PNG dependencies. Package only the game textures and license needed
  for native execution, not upstream Python, screenshots, or training data.
- Treat representative smoke tests only as development probes. Completion
  requires all games and public names, including nontrivial terminal rollouts.
- Add one protected core step-counter reset hook for same-step autoreset.
  Craftax publishes the terminal transition before resetting that counter;
  other families retain their existing behavior. This avoids an unbounded
  base counter across internally reset episodes.

## Outcomes & Retrospective

The native implementation is in draft PR #433. Full acceptance remains in
progress while source and release checks are reproduced on all four supported
platforms with the established local CI hosts. No package has been published
for this port. GitHub-hosted checks remain blocked by the account billing lock.

Validation evidence so far:

- `//envpool/dummy:dummy_py_envpool_test`: passed.
- `//envpool/craftax:craftax_envpool` and the test-only diagnostic extension:
  built successfully on local macOS arm64.
- `//envpool/craftax:craftax_test`: passed all 20 parameterized tests, covering
  all eight names/aliases, 400-step three-env deterministic replays with
  different thread counts, render selection/order/resize, malformed initial
  states, dm_env, and time limits.
- `//envpool/craftax:craftax_align_test`: passed all 21 cases in
  `configured-family-test.log`, including all names' 800-step trajectories,
  first-reset-only state injection, terminal score with all achievements,
  64-pixel tile rendering, god mode outside the map, Classic 16/32 and
  full 64 maps with custom capacities, day lengths, and noise overrides.
- `//envpool/craftax:craftax_behavior_test`: passed all 103 cases (99 directed
  trajectories and four weighted-draw cases) in `configured-family-test.log`;
  native and oracle state, observations,
  rewards, termination, and float RGB agree without residual tolerances.
- Whole-repository Ruff, Python format, mypy (164 files), and cpplint passed.
- `//third_party/craftax:render_compare`: passed, generated exact RGB
  comparisons after 96 Classic/overworld actions and 48 cave actions. Visually
  inspected the image; native is on the left and official on the right.
- `make -o doc-install docstyle`: doc8 and full Sphinx `-W` build passed.
- All 144 Craftax cases, `//envpool:make_test`, and all six core test targets
  passed again after C++ static-check cleanup (`post-tidy-tests.log`).
- After release validation, restored the development requirements and reran
  all ten targets with `--config=test`: all passed (`final-source-tests.log`).
  Final Ruff, formatting, mypy, cpplint, Buildifier, whitespace, and Sphinx
  checks passed after the documentation update.
- Clang-Tidy 22.1.8 passed all native game translation units and both bindings
  with diagnostics scoped to the new family. The binding commands used the
  actual Bazel compile options; unrelated existing core-template diagnostics
  are outside this local check. Full CI's Clang-Tidy job remains required.
- Whole-repository Ruff, format, mypy, cpplint, license checks and pinned
  clang-format checks passed.
- The normal `make bazel-release` path built a macOS arm64 CPython 3.12 wheel.
  `scripts/optimize_wheel.py` and `scripts/check_wheel_size.py` passed; the
  resulting wheel is 44,170,195 bytes, below the 100 MB release limit.
  Craftax contributes only its native extension, three registration/API
  modules, and license. Oracle modules, diagnostic extensions, JAX, and
  unused upstream assets are absent from the wheel.
- Installed that optimized wheel in a clean Python 3.12 environment with no
  Craftax, JAX, JAXlib, or Flax. All 20 Craftax tests, all 20 global make tests,
  the installed-wheel asset/origin check, three Jumanji registry tests, both
  examples, and the Procgen release consistency/render checks passed.
  `uv pip check` passed for all 26 installed packages.
- The configured package index does not yet supply the three asset packages
  at the 0.4 versions already required by main. For local release validation,
  built matching 0.4.0 wheels from this checkout's generated assets using the
  clean companion `envpool-assets` repository at `fa8d8a7` and its standard
  collect/build scripts. These are local test artifacts, not published
  releases; ordinary installation from the configured index remains blocked.
- Main's [release run 33114021535](https://github.com/sail-sg/envpool/actions/runs/33114021535)
  failed before any of its 12 jobs started. Check run `98663906668` explicitly
  reports an account lock due to a billing issue. There are no execution logs
  to diagnose; this does not establish any source failure or success. Linux,
  Windows, other supported Python versions, and full CI Clang-Tidy still need
  their normal jobs once the account is unlocked.

Transient logs and numerical probes are under `/private/tmp/envpool-craftax`.

## Context and Orientation

`envpool/core/env.h` provides per-environment state allocation and reset/step
callbacks. `envpool/core/py_envpool.h` binds the native batch pool to Python.
`envpool/pgx/` and `envpool/minigrid/` provide discrete-game and render examples;
`envpool/jumanji/` provides pinned JAX oracle-test examples. Global registration
is in `envpool/entry.py`, source/release dependencies in `envpool/BUILD`, and
installed-wheel smoke coverage in `envpool/make_test.py`.

An oracle is the official implementation used only in tests and documentation
tooling. A reset-time sync copies state at most once immediately after reset;
it never repairs divergent states during a rollout.

## Plan of Work

First enumerate the upstream factory and configuration surface. Establish
native data structures, Threefry random primitives, and a diagnostic binding
for reset-time state exchange. Port world generation, Classic rules, and full
Craftax rules in small independently checked functions. Keep state transitions
in C++, including random spawning, combat, crafting, inventory, sleep, plants,
levels, enchantments, potions, attributes, and the boss.

Port symbolic observations and the official texture-based renderer. Bind
symbolic and pixel specs, expose official information, and register every
supported name. Use deterministic replay and step-level comparison to find
the first divergence instead of widening tolerances. Complete asset fetching,
BUILD data dependencies, declarative packaging, docs, and the comparison image.

## Concrete Steps

Run from `/Users/jiayi/code/envpool-craftax`:

    make BAZEL_TEST_TARGETS=//envpool/dummy:dummy_py_envpool_test bazel-test
    make BAZEL_TEST_TARGETS=//envpool/craftax/... bazel-test
    make ruff py-format mypy cpplint clang-format buildifier
    make BAZEL_TEST_TARGETS=//envpool:make_test bazel-test
    make docstyle
    make bazel-release
    git diff --check

Use an isolated oracle environment under `/private/tmp/envpool-craftax-oracle`.
Keep development logs and transient probes under `/private/tmp/envpool-craftax`.

## Validation and Acceptance

All names must instantiate with the correct spaces. Native rollouts must be
deterministic for equal seeds and external action sequences, including pixels,
episode boundaries, and batched env selection. Oracle tests compare each step's
observations, reward, termination/truncation mapping, exposed information, and
rendering after at most one reset synchronization. Include randomized actions
and directed trajectories for crafting, combat, death, time limits, and full
game progression; compare through episode end when feasible. Add focused
behavior tests only for transitions not covered by those rollouts.

The tests must run without skips on supported Linux, macOS, and Windows jobs.
The installed wheel must run without Craftax/JAX installed. Document any
unavoidable platform residual only after identifying its numerical cause, and
keep it local to the affected operation/platform. Record job links and actual
results here before declaring acceptance complete.

## Idempotence and Recovery

The original worktree is untouched. Build/test commands can be repeated. Do
not delete user files, replace existing branches, or publish/merge a PR without
the applicable authorization. Keep the full unaccepted scope visible here if
an external prerequisite prevents completion.

## Artifacts and Notes

Upstream checkout: `/private/tmp/envpool-craftax-upstream`, detached at the
pinned tag. The upstream license is MIT, copyright 2024 Michael Matthews.

## Interfaces and Dependencies

Public APIs are EnvPool Gymnasium and dm_env pools, with discrete actions,
official symbolic or pixel observations, and batched native RGB rendering.
The official Craftax package and JAX belong to oracle tests only. C++ runtime
dependencies are EnvPool core and the existing PNG decoder.
