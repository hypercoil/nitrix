# Doc-fix: `tsconv` is documented as "convolution" but implements cross-correlation

> **Status (2026-06-02): RESOLVED.** Added a Notes clarification to `tsconv`:
> cross-correlation convention (kernel not flipped), as in conv layers;
> reverse the kernel along its last axis for a true convolution. See
> `IMPLEMENTATION_PLAN.md §10.3` (2026-06-02 entry). Provenance: surfaced
> building a `nitrix-perf-bench` case; ledger context in
> [`perf-bench-feedback.md`](../perf-bench-feedback.md).

`src/nitrix/signal/tsconv.py:45` — *"1-D convolution along the trailing
axis"* — wraps `jax.lax.conv_general_dilated` (line 67), which does **not**
flip the kernel (cross-correlation). Verified: an impulse `[0,0,1,0,0,0,0]`
⊛ `[1,2,3]` returns `[0,0,3,2,1,0,0]` (the kernel reversed about the centre
= correlation, not convolution). This is the standard deep-learning
convention (`torch.nn.Conv1d` is also cross-correlation), so it is fine for
ML users — but in a module named `signal` a DSP user expects a
*flipped*-kernel convolution.

**Fix (low priority, clarity only).** One line in the docstring:
*"cross-correlation convention (kernel not flipped), as in conv layers;
reverse the kernel for a true convolution."*

## Cross-references

- [`perf-bench-feedback.md`](../perf-bench-feedback.md) — the doc-drift ledger.
- `src/nitrix/signal/tsconv.py:45`.
