# Project Guidelines

## Code Style

- Python ≥3.10 with `from __future__ import annotations` in every file
- Modern type hints: `dict[str, Tensor]`, `list[str] | None` (no `Union`/`Dict`/`List`)
- Import convention: `from torch import Tensor` for annotations; `torch.xxx()` for operations
- Google-style docstrings with `Args:` and `Returns:` sections; document tensor shapes as `[batch, seq, d_model]`
- Store sub-modules in `nn.ModuleDict`/`nn.ModuleList` (not plain dicts/lists) so parameters register
- Learnable scalars via `nn.Parameter(torch.tensor(...))`, not raw tensors
- Validate preconditions early with descriptive `RuntimeError` messages; no silent failures
- Use `@dataclass` for data containers; `field(init=False)` for computed fields

## Architecture

FLX is a cortical, delta-native, thermally-routed LLM with persistent memory.

**Pipeline**: `input → SharedTrunk → ThalamicRouter → DomainCortices → CrossCorticalBridges → CortexMerger → MemoryController → Decoder → logits`

Key abstractions:
- **τ (tau)**: Scalar float in (0,1) — thermal arousal signal that gates depth, bridges, memory, and loops. Same value for all samples in a batch.
- **Deltas**: Low-rank (B @ A) parameter modifications with learned confidence. B is zero-initialized; A uses kaiming init.
- **Router**: Returns `dict[str, Tensor]` keyed by cortex name (not raw tensors). Merged downstream by `CortexMerger`.
- **Strata**: Three difficulty layers per cortex (intermediate/expert/frontier) with thermal gating via `tau_min`.

See [spec/](spec/) for the full architecture specification.

## Build and Test

```bash
make install      # pip install -e ".[dev]"
make test         # pytest tests/ -v
make test-quick   # pytest tests/ -q
make clean        # remove build artifacts and caches
```

No linter is configured. Triton is optional (`pip install -e ".[triton]"`).

## Conventions

- `flx/__init__.py` exports only `__version__`; users import submodules directly (`from flx.model import FLXNano`)
- Loss functions are standalone functions, not class methods
- Training is phased (0–4); each phase in its own module under `flx/training/`
- Tests use class-based organization with `_make_*()` helpers for defaults; no fixtures or mocking; verify shapes with `assert out.shape == (...)` and floats with `pytest.approx()` or `torch.allclose()`
- `DeltaMetadata` is a plain dataclass for provenance tracking — not an `nn.Module`
