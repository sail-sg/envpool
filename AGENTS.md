# EnvPool Agent Guide

This file is for agents adding a new environment family or upstream task family
to EnvPool.

## Goal

Ship a native EnvPool implementation that is:

- C++ at runtime
- fully registered
- documented
- deterministic
- pinned to a specific official oracle/version when one exists
- alignment-tested against the official oracle when one exists
- render-tested when rendering is supported
- green in full-platform CI and release packaging

Do not stop at a partial port.

## Hard Rules

- Do not add a Python runtime bridge for environment logic.
- Do not import the official Python environment inside C++ implementation code.
- If there is an upstream family with multiple registered IDs, support all of
  them before calling the port complete.
- Match the public EnvPool IDs to the upstream task family shape. If the
  upstream exposes multiple named scenarios / env IDs, register them
  individually instead of collapsing them into one generic task.
- Official Python packages may be used only in tests and doc-generation tooling
  as an oracle.
- Pick the exact upstream oracle/version before implementation and keep tests
  anchored to that target. Do not mix comparison baselines across upstream
  versions in one family.
- Keep changes localized to the new env family unless shared infrastructure is
  strictly required.
- Prefer following the closest already-green in-tree alignment/render harness
  before inventing a new comparison strategy. Reuse proven patterns such as the
  MuJoCo DMC-style oracle and render checks when the family shape matches.
- Prefer reusing existing `third_party/` libraries before adding new ones.
  Do not duplicate SDL-like dependencies just because a new env family needs
  them.
- Prefer fetching official upstream/library source in `third_party/` over
  copying source blobs into the repo. If code must be patched, store the patch
  under `third_party/<family>/`.
- Keep env-specific packaging declarative when possible.
  - prefer `setup.cfg`, Bazel data deps, and filegroups
  - avoid env-specific imperative hooks in `setup.py`
- Do not hide large upstream patch programs inside `envpool/workspace0.bzl`.
  `workspace0.bzl` may wire repository rules, but env-family patch payloads
  should live under `third_party/<family>/`.
- If an upstream registry list or scenario list is needed, generate it from the
  upstream source in tooling or `third_party/` rather than maintaining a copied
  handwritten list in runtime code.
- When extending an existing family with new upstream tasks, treat the new work
  as incomplete until the whole intended upstream task set is wired through the
  existing family harness. Do not stop at one representative env because it was
  enough to prove the plumbing.
- When extending an existing family, follow the already-green setup used by the
  older tasks in that family.
  - keep the same oracle/reset-sync strategy unless there is a documented
    reason not to
  - keep the same deterministic/render validation shape unless the new task
    fundamentally cannot support it
  - keep the same GL/bootstrap strategy before inventing a platform-specific
    workaround
- If a tolerance is unavoidable, make it:
  - as small as possible
  - platform-scoped if possible
  - documented in the test
- Do not widen tolerances or add platform-only workarounds until the root cause
  is understood well enough to explain why bitwise matching is impossible.
  "macOS only" is not an acceptable first fix.
- Do not use `skipTest` to hide a new-env regression on a supported platform
  when the family already has a known-good render / GL setup. Exhaust the
  existing in-tree strategy first, then document the real blocker if one
  remains.
- Before adding a broader tolerance for a new task family, inspect nearby
  existing envs in the same suite to confirm the issue is truly isolated. A new
  task sitting at `1e-5` while older tasks hold `1e-11` is a root-cause
  investigation signal, not a reason to immediately relax the whole suite.
- Minimize packaged assets for upstream-backed families. If only a subset of
  upstream assets is actually needed at runtime, exclude the rest and verify
  that release packaging still works.

## Implementation Loop

1. Read the existing human docs first:
   - `docs/content/new_env.rst`
   - `docs/content/contributing.rst`
2. Pick the closest in-tree reference implementation.
   - `envpool/dummy/` for skeletons
   - a real env family with similar action/obs/render structure
3. If the family wraps an upstream project, design the `third_party/<family>/`
   layout first.
   - repo rule / archive wiring
   - BUILD glue
   - patch files
   - generated upstream metadata helpers if needed
4. Create the env family folder under `envpool/<family>/`.
5. Add the minimum file set:
   - `BUILD`
   - `__init__.py`
   - `registration.py`
   - C++ headers / sources
   - family tests
6. Implement the native spec first.
   - config
   - observation/state spec
   - action spec
   - reward / termination / truncation
   - info keys
7. Implement native stepping logic in C++.
8. Implement native rendering in C++ if the family supports render.
9. Register every supported env ID and EnvPool alias.
10. Add docs:
   - env doc page under `docs/env/`
   - docs index entry if needed
   - README support-list entry if this is a new supported family
11. If the env has visual output and render is shipped, update the doc page with
    render results.
    - Keep render images on the doc page, not in README.
    - Prefer EnvPool-vs-official comparison images when the official renderer
      exists, with EnvPool on the left and official on the right.

## Preferred File Pattern

The exact filenames vary by family, but most ports need:

- native env / task implementation
- pybind entrypoint
- registration module
- `third_party/<family>/` repo / patch / metadata glue when upstream-backed
- deterministic test
- coverage / smoke test over all registered IDs
- alignment test
- render test

If the family wraps an upstream project with many task IDs, separate the
shared scene / task / observation logic from registration.

## Test Standards

A new env family is not done until the following are covered.

### 1. Registry Coverage

Verify:

- every intended ID is registered
- aliases resolve correctly
- public observation/action spaces match the intended contract
- upstream scenario / env-ID coverage is generated or checked against upstream
  when practical

If there is an upstream registry, add a test that fails when EnvPool misses an
upstream ID that is supposed to be supported.

### 2. Deterministic Test

At minimum:

- same seed
- same action sequence
- same observations
- same rewards
- same terminated / truncated outputs
- same rendered frames if render exists

Run this for multiple steps, not just reset or one step.
The determinism bar is end-to-end: resetting with the same seed and replaying
the same external action sequence must produce the same rollout, not just the
same initial state.
For new tasks added to an existing family, mirror the coverage shape of the old
tasks instead of only testing one new ID.

### 3. Alignment Test

When an official implementation exists, add oracle-based alignment tests.

Default standard:

- synchronize internal state at most once immediately after reset if needed to
  eliminate RNG/bootstrap differences between implementations
- after that reset-time sync, drive both sides only with the same external
  actions; do not resynchronize state mid-episode
- bitwise obs / reward / terminated / truncated / exposed info
- bitwise render when the renderer is shared or expected to match exactly
- cover every registered env ID, not just one representative case

If bitwise fails only because of platform math or renderer backend details,
keep the residual check narrow:

- only for the affected envs
- only for the affected platform
- with the smallest possible epsilon or pixel budget

Do not relax unrelated envs.
Do not skip entire envs or platforms to get CI green.
Do not replace state-sync oracle checks with long-horizon score-only checks.
When a family already has older align tests, the new-task align setup should
look the same unless a documented upstream difference forces a change.

### 4. Render Test

If render is supported:

- test reset render
- test multi-step render
- test batched render shape / dtype / env-id selection as applicable
- add doc render comparison images when the family has an official visual
  reference

Do not only test the first frame.
When adding new tasks to an existing renderable family, extend the all-task
render coverage rather than validating only one cherry-picked env.

### 5. Rollout Length

Use nontrivial rollout lengths.

- Do not rely on reset-only checks.
- Do not rely on a single action repeated for a handful of steps.
- Include multiple action patterns and enough total steps to exercise terminal
  logic and state transitions.
- For oracle alignment, prefer running to episode end when feasible. Very short
  rollouts are only acceptable as an early smoke check, not as the final
  alignment bar.
- Heuristic return / score regression checks are useful only as a coarse sanity
  check. They do not establish semantic alignment when a step-level oracle
  comparison is available.

### 6. Multi-Agent / Multi-Player Cases

If the env family supports multiple players:

- test player-shaped actions and observations
- test done / terminated semantics carefully
- check whether official APIs expose per-player values or env-level values
- document any remaining API mismatch as a TODO in code

## Validation Commands

Run the narrowest useful checks while developing, then the broader checks
before sending for review.

Common local checks:

```bash
make ruff py-format
make mypy
make cpplint
make clang-format
bazel test --test_output=errors //envpool/<family>/...
git diff --check
```

Broader checks when the family is ready:

```bash
make bazel-test BAZEL_TEST_TARGETS=//envpool:make_test
```

If docs changed, also run the relevant doc validation path.
Before declaring the family done, ensure the corresponding GitHub Actions test
and release jobs are green on all supported platforms.

## Review Bar

Before opening or updating a PR, verify:

- no Python bridge leaked into runtime implementation
- tests cover all registered IDs
- docs match actual supported IDs and render results
- any tolerance is justified and scoped
- reset-time oracle state sync, if used, happens only once and is documented
- no score-only "correctness" gate is standing in for step-level alignment
- no large env-specific patch blobs remain in `workspace0.bzl` or ad hoc
  `setup.py` logic
- upstream dependencies and patches live under `third_party/<family>/`
- no env-specific tests are skipped on any supported platform
- any new tasks added to an existing family are covered by deterministic,
  alignment, and render tests in the same style as the preexisting tasks
- any unusually large new-task drift has been compared against older tasks in
  the same suite and explained before tolerances were widened
- unnecessary upstream assets were removed from packaging when safe, and the
  release path was revalidated afterward
- all TODOs describe real remaining API gaps, not incomplete implementation

If a new env family cannot meet these standards yet, it is still in progress.
