# kit-cuda

The CUDA-graph delivery mechanism for the mlopt game.

`kit_cuda.CudaGraphRunner` is the `mlopt_core.ports.Runner` that
captures a delivered module into a `torch.cuda.CUDAGraph` — collapsing
a carrier executor's per-level kernel launches, or a generic
`IRModule`'s per-leaf tape dispatch, into one replayable graph.  It is
the only piece of the pipeline that is *device*-coupled, not merely
torch-coupled, so it is separately installable.

It sits at the **top** of the torch stack: it depends on
`mlopt-core` + `kit-torch` (handl-lang/eqsat + handl-lang/graphopt)
and `torch` — never on `mlopt_orchestrator` (the `Runner` protocol is
duck-typed), the game layer, or any sibling backend.

## Develop

```sh
uv sync --group dev
```

or as a git dependency:

```toml
kit-cuda = { git = "https://github.com/handl-lang/kit-cuda.git", tag = "v0.1.0" }
```

Substrate pins (`mlopt-core`, `mlopt-orchestrator`, `mlopt-dtypes`)
resolve from `handl-lang/eqsat`; `kit-torch` resolves through the
consumer's own `[tool.uv.sources]` (graphopt workspace member or git
subdir pin) — dep sources are honored transitively, so this repo
deliberately carries no `kit-torch` source entry.

## Verify

```sh
uv run scripts/gate.py --coverage   # pytest → ruff → ty → lint-imports → coverage floor
```

`scripts/gate.py` is a stdlib runner over the dev toolchain; `ci.sh`
is the gate entry.

## License

MIT.
