use core::mem::size_of;
use std::fs::{self, File};
use std::io::Write;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cube_awq::kernel::awq_kernel_gemm::{awq_gemm_linear, ShapeConfig};
use cubecl::bytes::Bytes;
use cubecl::future;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

type R = WgpuRuntime;

const AWQ_LINE_SIZE: usize = 4;
const AWQ_PACK_FACTOR: usize = 8;
const DEFAULT_GROUP_SIZE: usize = 128;
const INNER_K_INFERENCE: usize = 64;
const TILE_M_DECODE: usize = 16;
const TILE_M_PREFILL: usize = 32;

#[derive(Clone, Copy, Debug)]
enum InferencePhase {
    Decode,
    Prefill,
}

impl InferencePhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

fn select_tile_n(n: usize) -> usize {
    if n >= 128 && n.is_multiple_of(128) {
        128
    } else if n >= 64 && n.is_multiple_of(64) {
        64
    } else {
        32
    }
}

#[derive(Clone, Copy, Debug)]
struct GemmCase {
    name: &'static str,
    phase: InferencePhase,
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    m_tile_size: usize,
    n_tile_size: usize,
    inner_k_size: usize,
}

impl GemmCase {
    fn llm_case(
        name: &'static str,
        phase: InferencePhase,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
    ) -> Self {
        let m_tile_size = match phase {
            InferencePhase::Decode => TILE_M_DECODE,
            InferencePhase::Prefill => TILE_M_PREFILL,
        };
        Self {
            name,
            phase,
            b,
            m,
            k,
            n,
            group_size,
            m_tile_size,
            n_tile_size: select_tile_n(n),
            inner_k_size: INNER_K_INFERENCE,
        }
    }

    fn shape_config(self) -> ShapeConfig {
        ShapeConfig::new(
            self.m,
            self.n,
            self.k,
            self.m_tile_size,
            self.n_tile_size,
            self.inner_k_size,
        )
    }

    fn matmul_flops(self) -> u64 {
        2_u64 * self.b as u64 * self.m as u64 * self.n as u64 * self.k as u64
    }

    fn tokens_per_iter(self) -> u64 {
        (self.b * self.m) as u64
    }

    fn with_tiling(self, m_tile: usize, n_tile: usize, inner_k: usize, group_size: usize) -> Self {
        Self {
            m_tile_size: m_tile,
            n_tile_size: n_tile,
            inner_k_size: inner_k,
            group_size,
            ..self
        }
    }
}

#[derive(Debug)]
struct SweepRecord {
    case_name: &'static str,
    phase: &'static str,
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    m_tile_size: usize,
    n_tile_size: usize,
    inner_k_size: usize,
    avg_ms: f64,
    gflops: f64,
}

struct AwqGemmBuffers {
    input: Handle,
    qweight: Handle,
    qzeros: Handle,
    scales: Handle,
    output: Handle,
    input_shape: [usize; 3],
    input_strides: [usize; 3],
    qweight_shape: [usize; 2],
    qweight_strides: [usize; 2],
    qzeros_shape: [usize; 2],
    qzeros_strides: [usize; 2],
    scales_shape: [usize; 2],
    scales_strides: [usize; 2],
    output_shape: [usize; 3],
    output_strides: [usize; 3],
}

fn pack8(values: &[u8; 8]) -> u32 {
    values.iter().enumerate().fold(0_u32, |acc, (idx, &value)| {
        acc | (((value & 0xF) as u32) << (idx * 4))
    })
}

fn build_input(case: GemmCase) -> Vec<f32> {
    (0..case.b * case.m * case.k)
        .map(|idx| {
            let centered = (idx % 31) as f32 - 15.0;
            centered * 0.0625
        })
        .collect()
}

fn build_qweight(case: GemmCase) -> Vec<u32> {
    let packed_n = case.n / AWQ_PACK_FACTOR;
    let mut values = vec![0_u32; case.k * packed_n];

    for row in 0..case.k {
        for col in 0..packed_n {
            let mut nibbles = [0_u8; 8];
            for (offset, nibble) in nibbles.iter_mut().enumerate() {
                *nibble = ((row + col + offset) % 15 + 1) as u8;
            }
            values[row * packed_n + col] = pack8(&nibbles);
        }
    }

    values
}

fn build_qzeros(case: GemmCase) -> Vec<u32> {
    let num_groups = case.k / case.group_size;
    let packed_n = case.n / AWQ_PACK_FACTOR;
    let mut values = vec![0_u32; num_groups * packed_n];

    for group in 0..num_groups {
        for col in 0..packed_n {
            let mut nibbles = [0_u8; 8];
            for (offset, nibble) in nibbles.iter_mut().enumerate() {
                *nibble = ((group * 3 + col + offset) % 8) as u8;
            }
            values[group * packed_n + col] = pack8(&nibbles);
        }
    }

    values
}

fn build_scales(case: GemmCase) -> Vec<f32> {
    let num_groups = case.k / case.group_size;
    let mut values = vec![0.0_f32; num_groups * case.n];

    for group in 0..num_groups {
        for col in 0..case.n {
            let base = ((group * 17 + col) % 23) as f32;
            values[group * case.n + col] = 0.02 + base * 0.01;
        }
    }

    values
}

fn prepare_buffers(client: &ComputeClient<R>, case: GemmCase) -> AwqGemmBuffers {
    let input = build_input(case);
    let qweight = build_qweight(case);
    let qzeros = build_qzeros(case);
    let scales = build_scales(case);

    let input_shape = [case.b, case.m, case.k];
    let input_strides = [case.m * case.k, case.k, 1];
    let qweight_shape = [case.k, case.n / AWQ_PACK_FACTOR];
    let qweight_strides = [case.n / AWQ_PACK_FACTOR, 1];
    let qzeros_shape = [case.k / case.group_size, case.n / AWQ_PACK_FACTOR];
    let qzeros_strides = [case.n / AWQ_PACK_FACTOR, 1];
    let scales_shape = [case.k / case.group_size, case.n];
    let scales_strides = [case.n, 1];
    let output_shape = [case.b, case.m, case.n];
    let output_strides = [case.m * case.n, case.n, 1];

    AwqGemmBuffers {
        input: client.create(Bytes::from_bytes_vec(bytemuck::cast_slice(&input).to_vec())),
        qweight: client.create(Bytes::from_bytes_vec(
            bytemuck::cast_slice(&qweight).to_vec(),
        )),
        qzeros: client.create(Bytes::from_bytes_vec(
            bytemuck::cast_slice(&qzeros).to_vec(),
        )),
        scales: client.create(Bytes::from_bytes_vec(
            bytemuck::cast_slice(&scales).to_vec(),
        )),
        output: client.empty(case.b * case.m * case.n * size_of::<f32>()),
        input_shape,
        input_strides,
        qweight_shape,
        qweight_strides,
        qzeros_shape,
        qzeros_strides,
        scales_shape,
        scales_strides,
        output_shape,
        output_strides,
    }
}

fn launch_once(
    client: &ComputeClient<R>,
    shape: &ShapeConfig,
    group_size: usize,
    buffers: &AwqGemmBuffers,
) {
    let input_ref = TensorHandleRef {
        handle: &buffers.input,
        strides: &buffers.input_strides,
        shape: &buffers.input_shape,
        elem_size: size_of::<f32>(),
        runtime: PhantomData,
    };
    let qweight_ref = TensorHandleRef {
        handle: &buffers.qweight,
        strides: &buffers.qweight_strides,
        shape: &buffers.qweight_shape,
        elem_size: size_of::<u32>(),
        runtime: PhantomData,
    };
    let qzeros_ref = TensorHandleRef {
        handle: &buffers.qzeros,
        strides: &buffers.qzeros_strides,
        shape: &buffers.qzeros_shape,
        elem_size: size_of::<u32>(),
        runtime: PhantomData,
    };
    let scales_ref = TensorHandleRef {
        handle: &buffers.scales,
        strides: &buffers.scales_strides,
        shape: &buffers.scales_shape,
        elem_size: size_of::<f32>(),
        runtime: PhantomData,
    };
    let output_ref = TensorHandleRef {
        handle: &buffers.output,
        strides: &buffers.output_strides,
        shape: &buffers.output_shape,
        elem_size: size_of::<f32>(),
        runtime: PhantomData,
    };

    awq_gemm_linear::<R, f32>(
        client,
        &input_ref,
        &qweight_ref,
        &qzeros_ref,
        &scales_ref,
        &output_ref,
        shape,
        group_size,
    )
    .expect("AWQ GEMM launch failed");

    future::block_on(client.sync()).expect("AWQ GEMM sync failed");
}

fn estimate_perf(
    case: GemmCase,
    client: &ComputeClient<R>,
    shape: &ShapeConfig,
    buffers: &AwqGemmBuffers,
    warmup_runs: usize,
    measure_runs: usize,
) -> (f64, f64) {
    for _ in 0..warmup_runs {
        launch_once(client, shape, case.group_size, buffers);
    }

    let begin = Instant::now();
    for _ in 0..measure_runs {
        launch_once(client, shape, case.group_size, buffers);
    }

    let sec_per_run = begin.elapsed().as_secs_f64() / measure_runs as f64;
    let ms_per_run = sec_per_run * 1_000.0;
    let gflops = case.matmul_flops() as f64 / sec_per_run / 1e9;

    (ms_per_run, gflops)
}

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_usize_list(name: &str, default: &[usize]) -> Vec<usize> {
    let Some(raw) = std::env::var(name).ok() else {
        return default.to_vec();
    };

    let parsed: Vec<usize> = raw
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter_map(|piece| {
            let trimmed = piece.trim();
            if trimmed.is_empty() {
                return None;
            }
            trimmed.parse::<usize>().ok()
        })
        .filter(|value| *value > 0)
        .collect();

    if parsed.is_empty() {
        default.to_vec()
    } else {
        parsed
    }
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn case_filter() -> Option<String> {
    std::env::var("AWQ_BENCH_CASE_FILTER")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
}

fn should_run_case(filter: &Option<String>, case_name: &str) -> bool {
    match filter {
        Some(filter) => case_name.to_ascii_lowercase().contains(filter),
        None => true,
    }
}

fn is_valid_tiling(
    case: GemmCase,
    m_tile: usize,
    n_tile: usize,
    inner_k: usize,
    group_size: usize,
    max_workgroup_units: usize,
) -> bool {
    if m_tile == 0
        || n_tile == 0
        || inner_k == 0
        || group_size == 0
        || !m_tile.is_multiple_of(2)
        || m_tile * n_tile > max_workgroup_units
    {
        return false;
    }

    case.n.is_multiple_of(n_tile)
        && case.k.is_multiple_of(inner_k)
        && case.k.is_multiple_of(group_size)
        && case.n.is_multiple_of(AWQ_LINE_SIZE)
        && case.k.is_multiple_of(AWQ_LINE_SIZE)
        && inner_k.is_multiple_of(AWQ_LINE_SIZE)
        && group_size >= inner_k
        && group_size.is_multiple_of(inner_k)
        && n_tile.is_multiple_of(AWQ_LINE_SIZE * AWQ_PACK_FACTOR)
}

fn write_sweep_csv(path: &str, records: &[SweepRecord]) -> std::io::Result<()> {
    let csv_path = std::path::Path::new(path);
    if let Some(parent) = csv_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(csv_path)?;
    writeln!(
        file,
        "case,phase,b,m,k,n,group_size,m_tile_size,n_tile_size,inner_k_size,avg_ms,gflops"
    )?;

    for record in records {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{:.6},{:.6}",
            record.case_name,
            record.phase,
            record.b,
            record.m,
            record.k,
            record.n,
            record.group_size,
            record.m_tile_size,
            record.n_tile_size,
            record.inner_k_size,
            record.avg_ms,
            record.gflops,
        )?;
    }

    Ok(())
}

fn run_sweep_mode(base_cases: &[GemmCase]) {
    let m_tiles = parse_usize_list("AWQ_SWEEP_M_TILES", &[16, 32]);
    let n_tiles = parse_usize_list("AWQ_SWEEP_N_TILES", &[32, 64, 128]);
    let inner_ks = parse_usize_list("AWQ_SWEEP_INNER_KS", &[64]);
    let group_sizes = parse_usize_list("AWQ_SWEEP_GROUP_SIZES", &[128]);

    let warmup_runs = parse_usize("AWQ_SWEEP_WARMUP", 2);
    let measure_runs = parse_usize("AWQ_SWEEP_RUNS", 6);
    let topk = parse_usize("AWQ_SWEEP_TOPK", 10);
    let max_workgroup_units = parse_usize("AWQ_SWEEP_MAX_UNITS", 1024);
    let csv_path = std::env::var("AWQ_SWEEP_CSV")
        .unwrap_or_else(|_| "cube-awq/target/awq_kernel_gemm_sweep.csv".to_string());

    println!(
        "[awq_kernel_gemm] sweep start: warmup={}, runs={}, max_units={}, m_tiles={:?}, n_tiles={:?}, inner_ks={:?}, group_sizes={:?}",
        warmup_runs, measure_runs, max_workgroup_units, m_tiles, n_tiles, inner_ks, group_sizes
    );

    let device = WgpuDevice::default();
    let client = R::client(&device);

    let mut records = Vec::<SweepRecord>::new();

    for base in base_cases {
        println!(
            "[awq_kernel_gemm] sweep case={} (B={}, M={}, K={}, N={})",
            base.name, base.b, base.m, base.k, base.n
        );

        for &group_size in &group_sizes {
            for &m_tile in &m_tiles {
                for &n_tile in &n_tiles {
                    for &inner_k in &inner_ks {
                        if !is_valid_tiling(
                            *base,
                            m_tile,
                            n_tile,
                            inner_k,
                            group_size,
                            max_workgroup_units,
                        ) {
                            continue;
                        }

                        let case = base.with_tiling(m_tile, n_tile, inner_k, group_size);
                        let shape = case.shape_config();
                        let buffers = prepare_buffers(&client, case);
                        let (avg_ms, gflops) = estimate_perf(
                            case,
                            &client,
                            &shape,
                            &buffers,
                            warmup_runs,
                            measure_runs,
                        );

                        println!(
                            "  - g={} mt={} nt={} ik={} => {:.3} ms, {:.2} GFLOPS",
                            group_size, m_tile, n_tile, inner_k, avg_ms, gflops
                        );

                        records.push(SweepRecord {
                            case_name: case.name,
                            phase: case.phase.as_str(),
                            b: case.b,
                            m: case.m,
                            k: case.k,
                            n: case.n,
                            group_size,
                            m_tile_size: m_tile,
                            n_tile_size: n_tile,
                            inner_k_size: inner_k,
                            avg_ms,
                            gflops,
                        });
                    }
                }
            }
        }
    }

    if records.is_empty() {
        println!("[awq_kernel_gemm] no valid sweep config found.");
        return;
    }

    records.sort_by(|lhs, rhs| rhs.gflops.total_cmp(&lhs.gflops));

    println!("[awq_kernel_gemm] top-{} configs:", topk.min(records.len()));
    for (idx, record) in records.iter().take(topk).enumerate() {
        println!(
            "  #{:02} {} g={} mt={} nt={} ik={} => {:.3} ms, {:.2} GFLOPS",
            idx + 1,
            record.case_name,
            record.group_size,
            record.m_tile_size,
            record.n_tile_size,
            record.inner_k_size,
            record.avg_ms,
            record.gflops
        );
    }

    match write_sweep_csv(&csv_path, &records) {
        Ok(()) => println!("[awq_kernel_gemm] wrote sweep csv: {}", csv_path),
        Err(err) => println!(
            "[awq_kernel_gemm] failed to write csv {}: {}",
            csv_path, err
        ),
    }
}

fn bench_cases() -> Vec<GemmCase> {
    let mut cases = vec![
        // Decode (single-token, latency-oriented)
        GemmCase::llm_case(
            "decode_qkv_b1",
            InferencePhase::Decode,
            1,
            1,
            896,
            896,
            DEFAULT_GROUP_SIZE,
        ),
        GemmCase::llm_case(
            "decode_o_proj_b1",
            InferencePhase::Decode,
            1,
            1,
            896,
            896,
            DEFAULT_GROUP_SIZE,
        ),
        GemmCase::llm_case(
            "decode_mlp_up_b1",
            InferencePhase::Decode,
            1,
            1,
            896,
            4864,
            DEFAULT_GROUP_SIZE,
        ),
        GemmCase::llm_case(
            "decode_mlp_down_b1",
            InferencePhase::Decode,
            1,
            1,
            4864,
            896,
            DEFAULT_GROUP_SIZE,
        ),
        // Prefill (longer sequence, throughput-oriented)
        GemmCase::llm_case(
            "prefill_qkv_s128",
            InferencePhase::Prefill,
            1,
            128,
            896,
            896,
            DEFAULT_GROUP_SIZE,
        ),
        GemmCase::llm_case(
            "prefill_o_proj_s128",
            InferencePhase::Prefill,
            1,
            128,
            896,
            896,
            DEFAULT_GROUP_SIZE,
        ),
        GemmCase::llm_case(
            "prefill_mlp_up_s128",
            InferencePhase::Prefill,
            1,
            128,
            896,
            4864,
            DEFAULT_GROUP_SIZE,
        ),
        GemmCase::llm_case(
            "prefill_mlp_down_s128",
            InferencePhase::Prefill,
            1,
            128,
            4864,
            896,
            DEFAULT_GROUP_SIZE,
        ),
    ];

    // Set AWQ_BENCH_FULL=1 for heavier decode batching and long-context prefill.
    if env_flag("AWQ_BENCH_FULL") {
        cases.push(GemmCase::llm_case(
            "decode_mlp_up_b8",
            InferencePhase::Decode,
            8,
            1,
            896,
            4864,
            DEFAULT_GROUP_SIZE,
        ));
        cases.push(GemmCase::llm_case(
            "prefill_mlp_up_s512",
            InferencePhase::Prefill,
            1,
            512,
            896,
            4864,
            DEFAULT_GROUP_SIZE,
        ));
        cases.push(GemmCase::llm_case(
            "prefill_mlp_down_s512",
            InferencePhase::Prefill,
            1,
            512,
            4864,
            896,
            DEFAULT_GROUP_SIZE,
        ));
    }

    cases
}

fn bench_awq_kernel_gemm(c: &mut Criterion) {
    let filter = case_filter();
    let cases: Vec<GemmCase> = bench_cases()
        .into_iter()
        .filter(|case| should_run_case(&filter, case.name))
        .collect();

    if let Some(filter) = &filter {
        println!("[awq_kernel_gemm] case filter: {}", filter);
    }
    if cases.is_empty() {
        println!("[awq_kernel_gemm] no case matched filter.");
        return;
    }

    if env_flag("AWQ_BENCH_SWEEP") {
        run_sweep_mode(&cases);
        return;
    }

    let mut group = c.benchmark_group("awq_kernel_gemm");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));

    for case in cases {
        let device = WgpuDevice::default();
        let client = R::client(&device);
        let shape = case.shape_config();
        let buffers = prepare_buffers(&client, case);

        let (avg_ms, est_gflops) = estimate_perf(case, &client, &shape, &buffers, 3, 10);
        let tokens_per_sec = case.tokens_per_iter() as f64 / (avg_ms / 1_000.0);
        let ms_per_token = avg_ms / case.tokens_per_iter() as f64;
        println!(
            "[awq_kernel_gemm] {} ({}) estimate: {:.3} ms, {:.3} ms/token, {:.1} tok/s, {:.2} GFLOPS",
            case.name,
            case.phase.as_str(),
            avg_ms,
            ms_per_token,
            tokens_per_sec,
            est_gflops
        );

        // Use one element as one token so throughput is interpreted as token/s.
        group.throughput(Throughput::Elements(case.tokens_per_iter()));
        group.bench_with_input(
            BenchmarkId::new(case.phase.as_str(), case.name),
            &case,
            |b, _| {
                b.iter(|| {
                    launch_once(&client, &shape, case.group_size, &buffers);
                    black_box(&buffers.output);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_awq_kernel_gemm);
criterion_main!(benches);
