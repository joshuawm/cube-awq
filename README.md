# cube-awq

GPU-accelerated [AWQ](https://arxiv.org/abs/2306.00978) (Activation-aware Weight Quantization) kernels written in Rust with [CubeCL](https://github.com/tracel-ai/cubecl), designed for LLM inference on any GPU backend (WebGPU, CUDA, etc.).

## Features

- **Fused AWQ GEMM** — INT4 weight dequantization fused into matrix multiplication, avoiding a separate dequantize pass
- **Standalone dequantization** — extract full-precision weights from AWQ-packed tensors
- **Drop-in Burn modules** — `AWQGEMMLinear` and `AWQLinear` that work as Burn `Module`s with `.forward()` API
- **Optimized for LLM inference** — double-buffered shared memory, register blocking, and a dedicated decode kernel path for single-token generation (M ≤ 8)
- **Cross-platform** — runs on any GPU backend supported by CubeCL (WebGPU, CUDA, Vulkan)

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
cube-awq = { git = "https://github.com/joshuawm/cube-awq" }
```

### Fused AWQ GEMM (recommended)

Use `AWQGEMMLinear` for fused dequant + matmul in a single kernel launch:

```rust
use cube_awq::modules::{AWQGEMMLinearConfig, AWQGEMMLinear};
use burn::prelude::*;
use burn_cubecl::CubeBackend;
use cubecl::wgpu::WgpuRuntime;

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

let device = Default::default();

// d_input=4096, d_output=4096, group_size=128
let layer = AWQGEMMLinearConfig::new(4096, 4096, 128)
    .with_bias(false)
    .build::<Backend>(&device);

// Load your AWQ weights into layer.qweight, qzeros, scales via Burn's state loading...

let input = Tensor::<Backend, 3>::zeros([1, 1, 4096], &device);
let output = layer.forward(input); // [1, 1, 4096]
```

### Standalone Dequantization

Use `AWQLinear` when you need the dequantized weight matrix:

```rust
use cube_awq::modules::{AWQLinearConfig, AWQLinear};

let layer = AWQLinearConfig::new(4096, 4096, 128).build::<Backend>(&device);
let full_weight = layer.dequantize(); // [4096, 4096] f32 tensor
```

## AWQ Weight Format

This crate expects the standard [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) packing layout:

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `qweight` | `[K, N/8]` | `i32` / `u32` | 8 × 4-bit weights packed per 32-bit int |
| `qzeros` | `[K/group_size, N/8]` | `i32` / `u32` | 8 × 4-bit zero points packed per 32-bit int |
| `scales` | `[K/group_size, N]` | `f32` / `f16` | Per-group scale factors |

Dequantization formula: `weight = (qweight - qzeros) * scales`

## Kernel Optimizations

The fused GEMM kernel (`gemm_awq`) includes:

- **Double buffering** — ping-pong shared memory tiles overlap global loads with compute
- **Register blocking** — each thread computes multiple rows × N-lines to maximize arithmetic intensity
- **On-the-fly dequantization** — shared memory stores packed `u32` weights, dequantized during compute
- **Decode-specific path** — when M ≤ 8 (single-token decode), uses a specialized tile config (tile_m=8, 4 N-lines/thread) for lower latency
- **Adaptive tiling** — automatic tile size selection based on matrix dimensions (tile_n: 32/64/128, tile_m: 8/16/32)

## Constraints

- `d_output` must be divisible by 32
- `d_input` must be divisible by 64 (INNER_K)
- `d_input` must be divisible by `group_size`
- `group_size` must be ≥ 64 and divisible by 64

These constraints are satisfied by all common LLM architectures (LLaMA, Qwen, MiniCPM, etc.) with standard group sizes (64, 128).

## Benchmarks

```bash
# Run all benchmark cases
cargo bench -p cube-awq awq_kernel_gemm

# Decode-only (single-token latency)
AWQ_BENCH_CASE_FILTER=decode cargo bench -p cube-awq awq_kernel_gemm

# Extended benchmark suite
AWQ_BENCH_FULL=1 cargo bench -p cube-awq awq_kernel_gemm
```

## Tests

```bash
cargo test -p cube-awq
```

## Requirements

- Rust 1.70+
- A GPU with WebGPU / Vulkan / CUDA support
- Dependencies: [CubeCL](https://github.com/tracel-ai/cubecl) + [Burn](https://github.com/tracel-ai/burn)

## License

MIT — see [LICENSE](LICENSE) for details.
