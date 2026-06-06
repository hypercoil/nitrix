# doc-drift: `_iir.py` module docstring says `backend='scan' (default)`

> **Status (2026-06-06): open — doc-drift from `nitrix-perf-bench`.** Found while
> hardening the sosfilt/sosfiltfilt perf cases to measure the *default* engine.
> Docstring-only; no behaviour change. Authored perf-bench-side.

## The drift

`src/nitrix/signal/_iir.py`'s **module docstring** (the engine list, ~line 22)
still says:

```
- ``backend='scan'`` (default) -- sequential ``lax.scan`` over time,
```

But the actual default is `backend='auto'` (resolved by `_resolve_iir_backend`
to **`'fft'` on GPU**, `'scan'` on CPU) -- the B12 FFT-convolution win. The
`sosfilt` / `sosfiltfilt` *function* docstrings are correct (`'auto'` (default)
... fft on GPU); only the module-level header is stale, naming the wrong default
engine.

## Why it matters

The default engine *is* the headline B12 result (FFT-convolution on GPU, ~200x
faster than the scan recurrence at obs=32768 on the L4 -- measured in the
perf-bench sosfilt/sosfiltfilt cases). A module docstring that says the default
is `scan` understates the op to anyone reading the file top-down, and it was a
contributing reason the perf-bench case had pinned `backend='scan'` (now fixed:
the headline row calls the no-kwarg default).

## Fix

Update the module docstring engine list to state `backend='auto'` is the default
and resolves to `'fft'` on GPU / `'scan'` on CPU (mirror the function docstring
+ `_resolve_iir_backend`). One-line docstring edit.

## Cross-references

- `signal/_iir.py:~22` (module docstring), vs `:60` (`_resolve_iir_backend`)
  and `:571` (`sosfilt` docstring, correct).
- [`iir-filter-gpu-backend.md`](iir-filter-gpu-backend.md) (B12 — the shipped
  FFT engine + auto default).
