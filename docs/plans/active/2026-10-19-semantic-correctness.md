# Prove native correctness and remove oracle-specific complexity

## Purpose

Implement all findings from the history audit at
`/private/tmp/envpool-history-audit-20261019/audit.md`, including both cleanup
passes, in one ordinary pull request. Preserve semantic correctness and native
determinism without reproducing incidental CPU or renderer rounding.
Do not merge the PR or change the CI cache/cancellation policy.

## Progress

- [x] Audit history and the current agent guide; reuse the existing checkout.
- [x] R1: Replace Jumanji future-output replay with independent native rollouts.
- [x] R2: Numerically compare all 398 MyoSuite tasks and remove the CPU pin.
- [x] R3/B1: Use native Jumanji rendering, including asynchronous APIs.
- [x] R4: Use independent official Playground task steps across all 29 tasks.
- [x] A1/A2/B4: Simplify Craftax comparisons, score, and light/observation math.
- [x] A3: Share Playground's CGL bootstrap.
- [x] A4: Consolidate derived locomotion numerical budgets.
- [x] B2: Replace MJLab backend-specific trigonometry with ordinary native math.
- [x] B3: Remove unnecessary CGL redraws using visual correctness evidence.
- [x] Review the complete diff and perform both additional cleanup passes.
- [x] Re-audit RNG/rounding compatibility after the user's follow-up review.
- [x] Clarify checked hidden-state transfer through native info in AGENTS.md.
- [x] Finish macOS validation, static/doc checks and all MyoSuite doc images.
- [x] Run the repository-wide Linux registration and creation check.
- [ ] Finish the remaining Linux family checks.
- [x] Open one ready-for-review PR, not a draft: #440.
- [ ] Close the Jumanji render-content validation gap found in self-review.
- [ ] Resolve the Linux MJLab render failures and clang-tidy findings.
- [ ] Complete supported-platform CI and installed-wheel release checks.
- [ ] Preserve useful logs and clean up the task-owned Linux devbox.

## Baseline and work sequence

The audit inspected 282 commit titles and traced 35 recent mainline changes,
starting from `01f9cd9edc9ebcbb26c262321bfa3b3c9ccd08de`. Work is on
`jiayi/semantic-correctness-cleanup`.

Jumanji previously exposed 49 future-transition replay fields across 11 tasks
and 10 render-replay fields. MyoSuite selected 25 traces and numerically compared
only four. Playground used short trajectories and several reference steps reused
native state or copied native stepping logic. Removing those shortcuts first
exposed actual behavior defects; repair those before simplifying arithmetic.

## Implementation

- **Jumanji:** remove future-output configuration and the duplicate Python
  viewer. Share the JAX key implementation with Craftax. Correct the exposed
  transition/RNG defects, including Minesweeper flood fill, Tetris collisions,
  route completion, native box geometry, and termination semantics. Compare all
  26 tasks through independent episodes; check rendering and async behavior.
- **MyoSuite:** compare every registered task through an episode, with reset-only
  synchronization into the actual `Robot.mj_data`. Correct observations, rewards,
  binary contact categories, TableTennis rallies and simultaneous termination/
  truncation. Implement native NPC policies with shared PCG64 and colored noise.
  Keep small upstream interpolation, goalkeeper-physics and enum-comparison
  patches under `third_party/myosuite` with behavioral regressions.
- **Playground:** replace 22 copied stepping functions with official task steps
  using the established shared MuJoCo engine adapter. Remove incorrect extra
  native forward calls; resolve controller action-scale precision at setup.
  Correct double sharding so all 29 tasks actually run.
- **Arithmetic and rendering:** use ordinary Craftax score/light math and MJLab
  trigonometry; remove MKL/SLEEF dependencies. Share CGL setup and remove redundant
  warmup/redraw calls while retaining context ownership and synchronization.
  Consolidate derived locomotion precision checks.
- **Maintenance:** share reset snapshot manifests, remove redundant constants-only
  testing and a duplicate docstring, and defer the doc renderer's helper import
  until its isolated EnvPool namespace exists. Preserve the established float64
  MuJoCo observation interface and verify safe widening from official float32.

## Validation and evidence

Logs and root-cause reports are kept in
`/private/tmp/envpool-correctness-cleanup/`. Current evidence:

| Area | Completed local validation |
| --- | --- |
| Craftax | macOS API, 21 alignment methods and 103 behavior methods |
| Jumanji | All 26 targets on macOS and Linux; regenerated comparison image |
| Playground | macOS native target and all 29 tasks across four oracle shards |
| MJLab | macOS native API, full terrain/curriculum checks and both 500-step motion-wrap cases |
| Locomotion | macOS native and complete alignment targets |
| MPE2 | All 14 methods on macOS and Linux after sharing PCG64 |
| MyoSuite | All 398 oracle tasks on macOS and Linux; native API/randomization, representative render and C++ rally targets on both |
| Repository-wide | Linux `//envpool:make_test` passed |
| Static checks | Ruff, Mypy (180 sources), C++ lint, license headers; formatting of 32 C++ and 14 Bazel files |
| Docs | doc8, Sphinx HTML and spelling pass; all 398 MyoSuite tasks regenerated and all 1,592 frame pairs pass existing render assertions |

The final supported-platform test/release matrix remains an independent gate;
local or historical results do not establish a green PR head.

Useful commands, with the repository's pinned Bazelisk toolchain:

```bash
make ruff mypy cpplint
make docstyle
USE_BAZEL_VERSION=9.2.0 bazelisk test --config=test --test_output=errors //envpool/mujoco:myosuite_oracle_align_test
USE_BAZEL_VERSION=9.2.0 bazelisk test --config=test --test_output=errors //envpool/jumanji/...
git diff --check
```

Reuse the existing Bazel cache and checkout. The task-owned Linux machine has
16 CPUs and 64 GiB RAM. Its latest full related run passed 30 of 31 targets;
the remaining numerical oracle then passed with the field-based checks.

## Decisions and discoveries

- Exact repeated-native rollouts and discrete events remain useful checks.
  Cross-implementation floating values instead use justified numerical accuracy.
- Matching only the observation copy of a physics state does not synchronize the
  integrator. Copy reset state once into the actual physics data, then let both
  implementations advance independently.
- The MyoSuite spectral-noise probe compared 12,060 samples across buffer sizes
  and refills within 4.27e-14. Its initial RNG state is shared; future noise is not.
- Preserve the upstream intentionally stationary Soccer P1 policy; use P2 to
  exercise moving-goalkeeper behavior.
- MuJoCo's six-digit XML output lost meaningful edited-model parameters. Export
  those assets at 17 significant digits rather than widening output checks to
  cover that loss.
- Sensitive contact/muscle trajectories use separate peak and per-component
  time-average bounds. Equal-control diagnostics isolate control-roundoff effects.
  Fatigue TableTennis P0 also amplifies tiny model recompilation residuals;
  its measured accuracy is explicitly documented. Additional model/control replay
  remains diagnostic only, outside the shipped oracle.
- A task may terminate at the same instant its time limit truncates. Reuse the
  explicit termination state instead of deriving it as `done & ~trunc`.
- Keep the existing Linux pip MuJoCo engine: replacing C++ MjSpec bindings across
  independently built libraries caused ABI failures. The established platform
  setup remains the reference strategy.

## Decision log

- 2026-10-19: The user authorized R1–R4, A1–A4 and B1–B4 in one PR, superseding
  the audit's suggestion to split the work.
- 2026-10-19: The user requested an ordinary PR, not a draft. Do not merge it.
- 2026-10-19: Prefer ordinary math and evidence-backed accuracy over backend
  emulation; retain genuine resource, lifecycle and deterministic-RNG fixes.
- 2026-10-19: The user clarified that oracle convenience does not justify a
  compatible PRNG. Expose necessary hidden state through native `info`, check
  its correctness, and copy it into the oracle before independent stepping.
  Reuse existing fields; do not introduce a shared-random-input framework.
  Per-step synchronization is allowed when justified, but does not itself
  verify copied behavior. AGENTS.md now states this boundary explicitly.
- 2026-10-19: Read the original implementations, not just recent history:
  `6c80fcf7` (2021 MountainCar state assignment), `8cacab6c` (2022 HalfCheetah
  standard C++ sampling plus `info:qpos0`/`info:qvel0` and oracle `set_state`),
  and `ad139502` (2022 DMC reset through `physics.reset_context`). The existing
  state-transfer design already solves reset RNG differences; the guide and
  new-environment documentation now point back to that design.

## Follow-up self-review

The review of `ca83ff4d` supersedes the earlier conclusion that no further
compatibility cleanup was useful. Detailed evidence and two separate suggestion
lists are in `/private/tmp/envpool-correctness-cleanup/self-review-440/audit.md`.

- Jumanji's render assertion accepts an unrelated barcode image: shape and
  nonblank pixels do not establish scene correctness. Extend meaningful scene
  checks without restoring the duplicate Matplotlib renderer.
- Both Linux MJLab motion-wrap cases fail exact frame equality after their
  state/reward checks pass. Saved comparisons localize the visible difference
  to shadows. Fixed-state investigation is still needed before choosing a
  visual budget or changing rendering.
- The first hosted clang-tidy job failed on braces, enum types and conversions;
  lint passed. Full test/release jobs were still running when inspected.
- Remaining compatibility candidates include shared NumPy PCG64/Ziggurat,
  JAX keys/sampling/XLA scan order, locomotion's NumPy RandomState clone and
  platform flag, and MJLab's Torch sampling/draw counter. These are new review
  suggestions, not implemented removals. Native sampling distributions and
  temporal correlations remain required behavior.
- Additional deletion candidates are unused SearchAndRescue observation
  configuration, duplicate distance helpers, a uniform-choice CDF, redundant
  inverse functions, and CGL queries whose results are discarded. Risks and
  validation needs are recorded separately in the review.
- Playground covers all 29 tasks, but its oracle horizon is bounded at 128
  steps or termination. This must not be described as full episodes for every
  task. Jumanji's documented search-distance tolerance also needs correction.

## Outcomes and remaining work

The original audit changes are in ordinary PR #440. Follow-up self-review found
the validation issues above and additional simplification candidates; the work
is not complete. Fix the actionable validation/CI failures and track the new
suggestions separately. Record full test and release results for the real PR
head before declaring completion and releasing the Linux devbox.
