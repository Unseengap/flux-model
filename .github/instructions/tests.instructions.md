---
description: "Use when writing or editing FLX tests. Covers test structure, assertion patterns, and helper conventions for pytest."
applyTo: "tests/**"
---
# Test Conventions

## Structure

- Class-based organization: `class TestComponentName`
- Factory helpers `_make_*(**kwargs)` build objects with sensible defaults, overridden by kwargs:
  ```python
  def _make_nano(self, **kwargs):
      defaults = dict(vocab_size=1000, d_model=64, nhead=4,
                      layers_per_stratum=1, dim_feedforward=128)
      defaults.update(kwargs)
      return FLXNano(**defaults)
  ```
- No fixtures, no mocking — instantiate real objects with small dimensions (`d_model=64`, `vocab_size=1000`)

## Assertions

- Shapes: `assert out.shape == (batch, seq, d_model)`
- Floats: `pytest.approx()` for scalars, `torch.allclose()` for tensors
- Bounds: `assert (scores >= 0).all() and (scores <= 1).all()`
- Types: `assert isinstance(result, dict)`

## Patterns

- Use small dimensions for speed (d_model=64, nhead=4, seq=10–20, batch=2)
- Test thermal gating: verify outputs are zeros when `tau < tau_min`
- Test dict-keyed outputs from router: verify keys match cortex names
- Separate tests for shape correctness, value correctness, and edge cases
