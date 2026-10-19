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
- [x] Replace PCG64/Ziggurat with native distributions and checked semantic
  state transfer for MPE2 and MyoSuite; complete macOS checks pass.
- [x] Replace Jumanji sampling with native distributions and checked semantic
  choices; all 26 macOS targets pass. Remove XLA scan-order emulation.
- [ ] Settle the remaining Craftax keyed-RNG tradeoff; its compact sampler is
  localized and simplified, but its oracle sequence coupling remains.
- [x] Remove locomotion RandomState compatibility and ARM FMA flag; macOS
  native 30 and oracle 65 methods pass, all 13 documentation comparisons pass.
- [x] Remove MJLab Torch sampling and RNG draw-count coupling; native tests
  and all 61 oracle methods pass on macOS.
- [x] Apply the second cleanup pass: unused SearchAndRescue config, duplicated
  distance helper, uniform-choice CDF, redundant transforms and CGL queries.
- [x] Finish baseline macOS validation, static/doc checks and all MyoSuite doc
  images (before follow-up RNG changes).
- [x] Run the repository-wide Linux registration and creation check.
- [ ] Finish the remaining Linux family checks.
- [x] Open one ready-for-review PR, not a draft: #440.
- [x] Add Jumanji scene-content checks for all 26 IDs; local checks pass.
- [x] Resolve the initial 36 clang-tidy findings (hosted check passed).
- [x] Diagnose and fix the CGL startup and Linux Playground library-selection
  failures; native CGL validation passes.
- [ ] Finish Linux validation of the revised MJLab and Playground checks.
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
  viewer. Use standard native sampling and checked semantic choices for stochastic
  steps; keep Craftax's remaining keyed sampler local to that family. Correct the exposed
  transition/RNG defects, including Minesweeper flood fill, Tetris collisions,
  route completion, native box geometry, and termination semantics. Compare all
  26 tasks through independent episodes; check rendering and async behavior.
- **MyoSuite:** compare every transition across a complete native episode,
  copying pre-step integrator state into the actual `Robot.mj_data` and
  checking its independently computed successor. Correct observations, rewards,
  binary contact categories, TableTennis rallies and simultaneous termination/
  truncation. Implement native NPC policies with standard random distributions
  and colored noise, independently stepping the oracle from checked sampled
  native reference and NPC state.
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
`/private/tmp/envpool-correctness-cleanup/`. This table records the baseline
before the follow-up RNG cleanup; it does not certify the current changes:

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
16 CPUs and 64 GiB RAM. An earlier full related run passed 30 of 31 targets;
the remaining numerical oracle then passed with the field-based checks.

## Decisions and discoveries

- Exact repeated-native rollouts and discrete events remain useful checks.
  Cross-implementation floating values instead use justified numerical accuracy.
- Matching only the observation copy of a physics state does not synchronize the
  integrator. Copy into the actual physics data and compare the independently
  computed successor before any subsequent synchronization.
- The earlier MyoSuite spectral-noise probe established the spectral algorithm;
  its 12,060 sample-by-sample comparisons are historical evidence only. The
  cleanup removes NumPy stream emulation and transfers sampled semantic state.
- Preserve the upstream intentionally stationary Soccer P1 policy; use P2 to
  exercise moving-goalkeeper behavior.
- MuJoCo's six-digit XML output lost meaningful edited-model parameters. Export
  those assets at 17 significant digits rather than widening output checks to
  cover that loss.
- Sensitive muscle/contact trajectories amplified benign model residuals.
  The follow-up cleanup replaces the growing per-task trajectory budgets with
  uniformly tight, synchronized transition checks. Every successor is compared
  before copying the next starting state; native uninterrupted replays remain.
- A task may terminate at the same instant its time limit truncates. Reuse the
  explicit termination state instead of deriving it as `done & ~trunc`.
- Linux Playground's official Barkour constructor succeeds with the pip engine
  but crashes after substituting the source-built library at MjSpec compilation.
  Keep the Linux bindings paired with their matching pip library. A full
  29-task validation is running; the isolated constructor check passes.

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
  platform flag, and MJLab's Torch sampling/draw counter. The user authorized
  all four groups. The NumPy group is implemented; the remaining groups are
  tracked above. Native distributions and temporal correlations remain required.
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
is not complete. Implement the authorized follow-up cleanup and fix the
actionable validation/CI failures. Record full test and release results for the real PR
head before declaring completion and releasing the Linux devbox.

## Follow-up cleanup implementation

- MPE2/MyoSuite use standard native uniform, integer and normal distributions.
  Delete PCG64/Ziggurat downloads and compatibility code; retain spectral noise.
  MPE2's 14 methods and MyoSuite's complete 398-task oracle, native, rendering
  and C++ tests pass on macOS, using the common transition budget.
- Remove the unused SearchAndRescue field, duplicated distance helper, uniform
  tracking CDF, inverse-transform round trip and discarded CGL queries. All five
  tracking checks, including 36 clips and first-use rendering, pass on macOS.
- Render review found real missing content: fixed padding erased Sudoku digits
  at 128px, and the shared number renderer truncated values above 99. The new
  behavior tests fail on the old binary; proportional padding and arbitrary
  digit counts fix both causes. Both rebuilt environment tests pass. The full
  first-wave suite has passed all 33 targets on macOS.
- The Jumanji content-check extension passes all 26 tasks after reset and
  three steps and rejects an unrelated checkerboard for all 26. It checks
  semantic landmarks rather than exact drawing styles; the integrated test
  passes in 47.3 seconds, and its full Bazel target passed with the MJLab batch.
- MJLab uses native sampling, checked command/reference/push/control state and
  ordinary quaternion arithmetic. Independent probes isolated a stale oracle
  curriculum and contact amplification of float32 motor-target rounding.
  All 61 alignment methods and native tests pass on macOS.
- Locomotion uses native sampling and checked actual layouts, models and target
  events. Its RandomState clone and ARM FMA/strict flags are removed. The native
  target passed; fixes to the oracle clock, compiled target placement and a
  seed-dependent hand fixture pass the full 65-method rerun. All 13
  documentation comparisons were regenerated and checked.
- SearchAndRescue uses native Gaussian acceleration and checked target velocity
  in the oracle. Independent tests cover increment statistics, temporal
  independence, speed clipping and wrapped motion. Categorical sampling uses
  an ordinary cumulative sum instead of emulating XLA tiles. The mixed 32-target rebuilt suite passes. All five other Jumanji
  samplers have also been replaced; its full 26-target suite passes. Craftax's
  local sampler simplification passes all three targets. Replacing its remaining
  core and sequence-coupled harness is a separate pending maintenance decision.
- The Windows std::array iterator fix is pushed in `0686f7f0`.
- The official fixed-state macOS replay in run `39249567582` demonstrated large
  incorrect startup shadows (mean error up to 13.44/255), followed by 477 stable
  frames. `717c09f1` restores the established first-frame CGL warmup while
  retaining deletion of unused GL queries and the every-frame duplicate draw.
  This protects completed rendering; it does not tighten pixel tolerances.
- Current validation and actionable failures are tracked in
  `/private/tmp/envpool-correctness-cleanup/state.json`. Historical passes do not
  replace checks on the final PR head.

## Latest review and validation follow-up

- `7813612a` replaces the five remaining Jumanji samplers. Oracle checks validate
  sampled tiles, fruit, pieces and ghost actions; MMST checks all six priority
  orders. A Snake terminal-wall observation bug exposed by the changed sequences
  has a regression that fails on the old binary and passes after the fix.
- `a4b5761d` removes the unused shared JAX sampler target and alias layer, and
  simplifies Craftax integer, uniform and categorical arithmetic. This does not
  claim to remove Craftax's remaining Threefry/key sequence coupling.
- The render-content PR comment is answered and resolved, with all 26 scene
  checks passing and unrelated checkerboards rejected. No other review comments
  were present at this check.
- MJLab's 12 documentation scenes and locomotion's 13 comparisons are regenerated
  and checked. The Craftax regeneration command passes with a one-level RGB quantization
  budget instead of requiring bitwise pixels.
- A follow-up review covers the changed state-transfer boundaries: copied
  sampling/control inputs are checked separately, and independently computed
  successors are recorded before the next integrator-state copy. Full hosted
  checks remain pending; no historical result certifies the latest head.

- The final pass found another Snake reset boundary: a fruit-only override was
  ignored, while a head-only override could place fruit on the head. Reset now
  honors each fixed position independently and samples the unspecified free
  cell. Both old failures are reproduced and the rebuilt focused check passes;
  identical fixed positions are rejected. The full Jumanji/create-test rerun is
  tracked with the final-head validation.

- Linux's final-source run exposed an oracle validation error for random mazes:
  the number of rooms can vary, so total spawn/goal counts need not match an
  independently sampled map. Check the configured counts in each actual room,
  the room limit and global totals instead. An official 512-map probe contains
  valid two-, three- and four-room layouts; both previously failing Linux cases
  pass with the correction. Full macOS alignment and final-platform CI continue.
