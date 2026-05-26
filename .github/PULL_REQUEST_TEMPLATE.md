## Description

Describe your changes in detail.

## Motivation and Context

Why is this change required? What problem does it solve?
If it fixes an open issue, please link to the issue here.
You can use the syntax `close #233` if this solves the issue #233

- [ ] I have raised an issue to propose this change ([required](https://envpool.readthedocs.io/en/latest/pages/contributing.html) for new features and bug fixes)

## Types of changes

What types of changes does your code introduce? Put an `x` in all the boxes that apply:

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds core functionality)
- [ ] New environment (non-breaking change which adds 3rd-party environment)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation (update in the documentation)
- [ ] Example (update in the folder of example)

## Implemented Tasks

- [ ] Subtask 1
- [ ] Subtask 2
- [ ] Subtask 3

## Checklist

Go over all the following points, and put an `x` in all the boxes that apply.
If you are unsure about any of these, don't hesitate to ask. We are here to help!

- [ ] I have read the [CONTRIBUTION](https://envpool.readthedocs.io/en/latest/pages/contributing.html) guide (**required**)
- [ ] My change requires a change to the documentation.
- [ ] I have updated the tests accordingly (*required for a bug fix or a new feature*).
- [ ] I have updated the documentation accordingly.
- [ ] I have reformatted the code using `make format` (**required**)
- [ ] I have checked the code using `make lint` (**required**)
- [ ] I have ensured `make bazel-test` pass. (**required**)

## New Environment Checklist

For PRs that add a new environment family or new upstream task family:

- [ ] Runtime logic is native C++ and does not bridge to the official Python environment.
- [ ] All intended upstream task IDs/scenarios are registered, documented, and covered by tests.
- [ ] The upstream oracle/version is pinned, and tests check EnvPool registration/configs against it.
- [ ] Determinism tests cover reset plus multi-step rollouts for every registered ID, including render frames when rendering is supported.
- [ ] Oracle alignment tests compare step-level observations, rewards, done/truncation, info, and renders after at most one reset-time state sync.
- [ ] Render tests cover reset, multi-step, batched render/env-id selection, and docs include EnvPool-vs-official images when an official renderer exists.
- [ ] `envpool/make_test.py`, release packaging, docs, and README support lists are updated.
