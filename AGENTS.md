# Agent guide — kit-cuda

The CUDA-graph delivery mechanism for the mlopt game —
`CudaGraphRunner`, the `mlopt_core.ports.Runner` that captures a
delivered module into a `torch.cuda.CUDAGraph` (carrier executors via
their own `capture_cuda_graph`, generic `IRModule` tapes via static
input buffers).  Split from `handl-lang/graphopt`
`packages/mlopt-cuda` in P3 Step D-1c (the third package of the
graphopt domain-package dissolution; the `mlopt-*` → `kit-*` rename).
Python 3.13, torch + `mlopt-core`/`mlopt-torch` on the eqsat/graphopt
substrate.

## Layout

- `src/kit_cuda/` — the one shipped package:
  - `runners.py` — `CudaGraphRunner` (`multi_device` per-device
    capture cache, opt-in `alias_inputs`/`mutable_inputs` alias-aware
    capture) + `_GraphReplayModule` / `_DeviceGraphModule`;
  - `__init__.py` — re-exports `CudaGraphRunner`.
- `tests/` — the package-bound suite (`test_runners.py`, carved out
  of graphopt's `tests/` — every import stays inside this package's
  dep closure: `kit_cuda` + `mlopt_core`/`mlopt_torch`/
  `mlopt_orchestrator` + torch).  Cross-stack tests (anything touching
  `kit_discovery`, `kit_carriers`, `kit_onnx`,
  `kit_host_tensor`/`mlopt_host_tensor`, gamekit, benchkit) stayed in
  graphopt and import `kit_cuda` as a git dep.

## Contracts

- **Top of the torch stack** — not a leaf: the runner consumes
  `mlopt_torch.device` (`device_graph_key`, `storage_fingerprint`)
  and `mlopt_torch.device_placement` (`ShardedAwareDevicePolicy`).
  It never imports `mlopt_orchestrator` — the `Runner` protocol is
  duck-typed (`apply(module, example_input, stats)`); the
  import-linter `forbidden` contract enforces the upward edges
  (orchestrator, gamekit/benchkit/discoverykit, `kit_discovery`,
  `kit_host_tensor`/`mlopt_host_tensor`, `kit_carriers`, `kit_onnx`).
- Substrate deps: `mlopt-core` (handl-lang/eqsat,
  `packages/mlopt-core` subdir, tag-pinned); `mlopt-torch`
  (handl-lang/graphopt, `packages/mlopt-torch` subdir, resolved by
  the *consumer's* `[tool.uv.sources]` — this repo deliberately has
  no `mlopt-torch` source entry; its own dev env binds it via a
  dev-group direct-URL, the kit-carriers dep-source lesson).
- `requires_cuda` tests skip cleanly at setup when CUDA is absent or
  OOMs on a probe allocation (shared `tests/conftest.py`).
- History was preserved via `git subtree split` of
  `graphopt:packages/mlopt-cuda` — `git log` here reaches back
  through the graphopt history.

## Verification

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src
.venv/bin/ruff format --check src
.venv/bin/ty check
.venv/bin/lint-imports
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
