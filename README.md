# cube-awq

AWQ kernels and modules for the `minicpm_rust` workspace.

## Scope

- AWQ dequantization and GEMM kernels (`src/kernel/`)
- AWQ module wrappers (`src/modules/`)
- Benchmarks (`benches/`) and tests (`tests/`)

## Quick Start

```bash
# run tests
cargo test -p cube-awq

# run kernel benchmark
cargo bench -p cube-awq awq_kernel_gemm
```

LLM inference shaped cases are included by default (decode + prefill).

```bash
# only decode-like cases
AWQ_BENCH_CASE_FILTER=decode cargo bench -p cube-awq awq_kernel_gemm

# longer run with larger prefill/decode batches
AWQ_BENCH_FULL=1 cargo bench -p cube-awq awq_kernel_gemm
```

## Public Repo Checklist

Before making the repo public:

1. Verify no weights or private assets are tracked.
2. Verify generated benchmark artifacts are ignored.
3. Add a project license (`LICENSE`) and attribution if third-party code is included.
4. Keep dependency revisions pinned in the workspace for reproducible builds.
