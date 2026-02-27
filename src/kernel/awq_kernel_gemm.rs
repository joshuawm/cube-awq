use cubecl::prelude::*;
use cubecl::{
    calculate_cube_count_elemwise,
    std::tensor::layout::linear::{linear_view, LinearView},
};

// The helper funcs. AutoAWQ GEMM packs lanes in [0,2,4,6,1,3,5,7].
// These unpackers directly return natural lane order [0,1,2,3] and [4,5,6,7]
// to avoid per-call reorder logic in the hot loop.
#[cube]
fn unpack_u32_to_i4_0_4<F: Float>(packed_value: u32) -> Line<F> {
    let mut unpacked_line = Line::<F>::empty(4usize);

    unpacked_line[0] = F::cast_from((packed_value >> 0u32) & 0x0f);
    unpacked_line[1] = F::cast_from((packed_value >> 16u32) & 0x0f);
    unpacked_line[2] = F::cast_from((packed_value >> 4u32) & 0x0f);
    unpacked_line[3] = F::cast_from((packed_value >> 20u32) & 0x0f);
    unpacked_line
}

#[cube]
fn unpack_u32_to_i4_4_8<F: Float>(packed_value: u32) -> Line<F> {
    let mut unpacked_line = Line::<F>::empty(4usize);

    unpacked_line[0] = F::cast_from((packed_value >> 8u32) & 0x0f);
    unpacked_line[1] = F::cast_from((packed_value >> 24u32) & 0x0f);
    unpacked_line[2] = F::cast_from((packed_value >> 12u32) & 0x0f);
    unpacked_line[3] = F::cast_from((packed_value >> 28u32) & 0x0f);
    unpacked_line
}

// input: (M,K)
// weight: (K,N)
// output: (M,N)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeConfig {
    m: usize,
    k: usize,
    n: usize,
    tile_m: usize,    // 8|16|32
    tile_n: usize,    // 32|64|128
    inner_k: usize,   // 32|64
    line_size: usize, // 4
    m_valid: usize,
}

impl ShapeConfig {
    pub fn new(
        m: usize,
        n: usize,
        k: usize,
        m_tile_size: usize,
        n_tile_size: usize,
        inner_k_size: usize,
    ) -> Self {
        Self::new_with_valid_m(m, n, k, m_tile_size, n_tile_size, inner_k_size, m)
    }

    pub fn new_with_valid_m(
        m: usize,
        n: usize,
        k: usize,
        m_tile_size: usize,
        n_tile_size: usize,
        inner_k_size: usize,
        m_valid: usize,
    ) -> Self {
        Self {
            m,
            k,
            n,
            tile_m: m_tile_size,
            tile_n: n_tile_size,
            inner_k: inner_k_size,
            line_size: 4,
            m_valid,
        }
    }
}

pub const DECODE_M_THRESHOLD: usize = 8;
pub const DECODE_TILE_M: usize = 8;

const ROWS_PER_THREAD: usize = 2;
const N_LINES_PER_THREAD: usize = 2;
const ROWS_PER_THREAD_DECODE: usize = 1;
const N_LINES_PER_THREAD_DECODE: usize = 4;

fn validate_launch_params_impl(
    shape: &ShapeConfig,
    group_size: usize,
    rows_per_thread: usize,
    n_lines_per_thread: usize,
) {
    assert!(shape.m > 0, "m must be > 0");
    assert!(shape.line_size > 0, "line_size must be > 0");
    assert!(shape.tile_m > 0, "m_tile_size must be > 0");
    assert!(shape.tile_n > 0, "n_tile_size must be > 0");
    assert!(shape.inner_k > 0, "inner_k_size must be > 0");
    assert!(group_size > 0, "group_size must be > 0");

    // line-based indexing safety
    assert!(
        shape.k.is_multiple_of(shape.line_size),
        "k must be divisible by line_size"
    );
    assert!(
        shape.n.is_multiple_of(shape.line_size),
        "n must be divisible by line_size"
    );
    assert!(
        shape.inner_k.is_multiple_of(shape.line_size),
        "inner_k_size must be divisible by line_size"
    );

    // tile loop safety
    assert!(
        shape.n.is_multiple_of(shape.tile_n),
        "n must be divisible by n_tile_size"
    );
    assert!(
        shape.k.is_multiple_of(shape.inner_k),
        "k must be divisible by inner_k_size"
    );

    // packed-4bit loading layout safety
    assert!(
        shape.tile_n.is_multiple_of(shape.line_size * 8),
        "n_tile_size must be divisible by line_size*8"
    );
    assert!(
        shape.n.is_multiple_of(shape.line_size * 8),
        "n must be divisible by line_size*8"
    );

    // AWQ group indexing safety
    assert!(
        shape.k.is_multiple_of(group_size),
        "k must be divisible by group_size"
    );
    assert!(
        group_size >= shape.inner_k && group_size.is_multiple_of(shape.inner_k),
        "fast path requires group_size >= inner_k_size and divisible by inner_k_size"
    );
    assert!(shape.m_valid <= shape.m, "m_valid must be <= m");

    // This kernel maps one thread to N_LINES_PER_THREAD output lines across ROWS_PER_THREAD rows.
    assert!(
        shape
            .tile_n
            .is_multiple_of(shape.line_size * n_lines_per_thread),
        "tile_n must be divisible by line_size * N_LINES_PER_THREAD"
    );
    assert!(
        shape.tile_m.is_multiple_of(rows_per_thread),
        "m_tile_size must be divisible by ROWS_PER_THREAD"
    );
}

fn validate_launch_params(shape: &ShapeConfig, group_size: usize) {
    validate_launch_params_impl(shape, group_size, ROWS_PER_THREAD, N_LINES_PER_THREAD);
}

fn validate_decode_launch_params(shape: &ShapeConfig, group_size: usize) {
    validate_launch_params_impl(
        shape,
        group_size,
        ROWS_PER_THREAD_DECODE,
        N_LINES_PER_THREAD_DECODE,
    );
}

// input: (M,K)
// weight: (N/8,K), transposed weight
// output: (M,N)
//
// Optimizations:
//   1. Double buffering — ping-pong shared memory tiles to overlap global loads with compute
//   2. N-direction register blocking — each thread computes 2 rows × 2 Lines (8 cols)
//   3. On-the-fly dequant — shared memory stores packed u32 weights, dequantized in compute
#[cube(launch)]
pub fn gemm_awq<F: Float>(
    input: &LinearView<Line<F>>,
    qweight: &LinearView<u32>,
    qzeros: &LinearView<u32>,
    scales: &LinearView<Line<F>>,
    output: &mut LinearView<Line<F>, ReadWrite>,
    #[comptime] shape: &ShapeConfig,
    #[comptime] group_size: usize,
) {
    let inner_k_lines = shape.inner_k / shape.line_size;
    let tile_n_lines = shape.tile_n / shape.line_size;
    let n_groups = tile_n_lines / N_LINES_PER_THREAD;
    let packs_per_row = shape.tile_n / 8;
    // K-major packed weight layout: [k][pack]
    let sm_weight_stride_packs = packs_per_row;
    // Per-buffer element counts
    let sm_input_buf_size = shape.tile_m * inner_k_lines;
    let sm_weight_buf_size = shape.inner_k * sm_weight_stride_packs;

    // Double-buffered shared memory (2× size)
    let mut sm_input = SharedMemory::<F>::new_lined(2 * sm_input_buf_size, shape.line_size);
    let mut sm_weight = SharedMemory::<u32>::new(2 * sm_weight_buf_size);

    // Shared cache for qzeros/scales per 8-col pack (single-buffered)
    // Store scales as Line<F> (vectorized) instead of scalar
    let mut sm_zero_pack = SharedMemory::<u32>::new(shape.tile_n / 8);
    let mut sm_scale_line = SharedMemory::<F>::new_lined(tile_n_lines, shape.line_size);

    // Tile origin
    let mc = (CUBE_POS / (shape.n / shape.tile_n)) * shape.tile_m;
    let nc = (CUBE_POS % (shape.n / shape.tile_n)) * shape.tile_n;
    let k_loop = shape.k / shape.inner_k;

    // Thread mapping: each thread computes 2 rows × 2 N-lines (8 cols)
    let unit_pos = UNIT_POS as usize;
    let thread_rows = shape.tile_m / ROWS_PER_THREAD;
    let idx_n_group = unit_pos % n_groups;
    let idx_tile_m = unit_pos / n_groups;
    let total_threads = thread_rows * n_groups;

    let n_line0 = idx_n_group * N_LINES_PER_THREAD;
    let n_line1 = n_line0 + 1;

    let row0 = idx_tile_m;
    let row1 = idx_tile_m + thread_rows;
    let row0_valid = (mc + row0) < shape.m_valid;
    let row1_valid = (mc + row1) < shape.m_valid;

    // Vectorized accumulators: 2 rows × 2 N-lines = 4 Line<F>
    let zero_val = F::cast_from(0);
    let mut acc_r0_n0 = Line::<F>::empty(shape.line_size).fill(zero_val);
    let mut acc_r0_n1 = Line::<F>::empty(shape.line_size).fill(zero_val);
    let mut acc_r1_n0 = Line::<F>::empty(shape.line_size).fill(zero_val);
    let mut acc_r1_n1 = Line::<F>::empty(shape.line_size).fill(zero_val);

    // Fast path only: each K tile stays inside a single AWQ group.
    let tiles_per_group = group_size / shape.inner_k;

    // ===================================================================
    // Prologue: load the first tile (idx_k=0) into buffer 0
    // ===================================================================
    // Cache zeros/scales for the first group (group_k=0)
    if unit_pos < packs_per_row {
        let pack = unit_pos;
        let unit_n = pack * 8;

        let qzeros_idx = nc + unit_n; // group_k=0 so group_k * shape.n = 0
        sm_zero_pack[pack] = qzeros[qzeros_idx / 8];

        let scale_idx = qzeros_idx / shape.line_size;
        let pack_line = pack * 2;
        sm_scale_line[pack_line] = scales[scale_idx];
        sm_scale_line[pack_line + 1] = scales[scale_idx + 1];
    }
    sync_cube();

    // Load input tile 0 into buffer 0 (in_off=0, idx_k=0)
    let required_input_units = sm_input_buf_size;
    let mut t = unit_pos;
    while t < required_input_units {
        let unit_m = t / inner_k_lines;
        let unit_k_line = t % inner_k_lines;
        if (mc + unit_m) < shape.m_valid {
            let unit_k = unit_k_line * shape.line_size;
            let input_idx = (mc + unit_m) * shape.k + unit_k; // idx_k=0 so offset is 0
            sm_input[unit_m * inner_k_lines + unit_k_line] = input[input_idx / shape.line_size];
        }
        t += total_threads;
    }

    // Load packed weight tile 0 into buffer 0 (w_off=0, idx_k=0), K-major layout
    let required_weight_units = shape.inner_k * packs_per_row;
    let mut w = unit_pos;
    while w < required_weight_units {
        let unit_k = w / packs_per_row;
        let pack = w % packs_per_row;
        let unit_n = pack * 8;
        let k_tmp = unit_k; // idx_k=0 so k_tmp = unit_k
        let qweight_idx = k_tmp * shape.n + nc + unit_n;
        let line_base = unit_k * sm_weight_stride_packs;
        sm_weight[line_base + pack] = qweight[qweight_idx / 8];

        w += total_threads;
    }
    // Publish buffer 0
    sync_cube();

    // ===================================================================
    // Main loop: compute on current buffer while prefetching next into other buffer
    // ===================================================================
    for idx_k in 0..k_loop {
        let curr_in_off = (idx_k % 2) * sm_input_buf_size;
        let curr_w_off = (idx_k % 2) * sm_weight_buf_size;
        let next_in_off = ((idx_k + 1) % 2) * sm_input_buf_size;
        let next_w_off = ((idx_k + 1) % 2) * sm_weight_buf_size;
        let has_next = (idx_k + 1) < k_loop;
        let next_k = idx_k + 1;

        // Prefetch next input tile into other buffer
        if has_next {
            let required_input_units = sm_input_buf_size;
            let mut t = unit_pos;
            while t < required_input_units {
                let unit_m = t / inner_k_lines;
                let unit_k_line = t % inner_k_lines;
                if (mc + unit_m) < shape.m_valid {
                    let unit_k = unit_k_line * shape.line_size;
                    let input_idx = (mc + unit_m) * shape.k + next_k * shape.inner_k + unit_k;
                    sm_input[next_in_off + unit_m * inner_k_lines + unit_k_line] =
                        input[input_idx / shape.line_size];
                }
                t += total_threads;
            }
        }

        // Prefetch next packed weight tile into other buffer (K-major layout)
        if has_next {
            let required_weight_units = shape.inner_k * packs_per_row;
            let mut w = unit_pos;
            while w < required_weight_units {
                let unit_k = w / packs_per_row;
                let pack = w % packs_per_row;
                let unit_n = pack * 8;
                let k_tmp = next_k * shape.inner_k + unit_k;
                let qweight_idx = k_tmp * shape.n + nc + unit_n;
                let line_base = next_w_off + unit_k * sm_weight_stride_packs;
                sm_weight[line_base + pack] = qweight[qweight_idx / 8];

                w += total_threads;
            }
        }

        // -------------------------
        // Compute on current buffer — dequantize packed weights on the fly
        // -------------------------
        let a_base0 = curr_in_off + row0 * inner_k_lines;
        let a_base1 = curr_in_off + row1 * inner_k_lines;
        let w_base = curr_w_off;
        let pack_idx = idx_n_group;
        let packed_z = sm_zero_pack[pack_idx];
        let sc_n0 = sm_scale_line[n_line0];
        let sc_n1 = sm_scale_line[n_line1];
        let zero_n0 = unpack_u32_to_i4_0_4::<F>(packed_z);
        let zero_n1 = unpack_u32_to_i4_4_8::<F>(packed_z);
        #[unroll]
        for t4 in 0..inner_k_lines {
            let kk = t4 * shape.line_size;

            let packed_w0 = sm_weight[w_base + (kk + 0) * sm_weight_stride_packs + pack_idx];
            let packed_w1 = sm_weight[w_base + (kk + 1) * sm_weight_stride_packs + pack_idx];
            let packed_w2 = sm_weight[w_base + (kk + 2) * sm_weight_stride_packs + pack_idx];
            let packed_w3 = sm_weight[w_base + (kk + 3) * sm_weight_stride_packs + pack_idx];

            let deq0_n0 = (unpack_u32_to_i4_0_4::<F>(packed_w0) - zero_n0) * sc_n0;
            let deq1_n0 = (unpack_u32_to_i4_0_4::<F>(packed_w1) - zero_n0) * sc_n0;
            let deq2_n0 = (unpack_u32_to_i4_0_4::<F>(packed_w2) - zero_n0) * sc_n0;
            let deq3_n0 = (unpack_u32_to_i4_0_4::<F>(packed_w3) - zero_n0) * sc_n0;
            let deq0_n1 = (unpack_u32_to_i4_4_8::<F>(packed_w0) - zero_n1) * sc_n1;
            let deq1_n1 = (unpack_u32_to_i4_4_8::<F>(packed_w1) - zero_n1) * sc_n1;
            let deq2_n1 = (unpack_u32_to_i4_4_8::<F>(packed_w2) - zero_n1) * sc_n1;
            let deq3_n1 = (unpack_u32_to_i4_4_8::<F>(packed_w3) - zero_n1) * sc_n1;

            if row0_valid {
                let a_line = sm_input[a_base0 + t4];
                let a0 = Line::<F>::empty(shape.line_size).fill(a_line[0]);
                let a1 = Line::<F>::empty(shape.line_size).fill(a_line[1]);
                let a2 = Line::<F>::empty(shape.line_size).fill(a_line[2]);
                let a3 = Line::<F>::empty(shape.line_size).fill(a_line[3]);

                acc_r0_n0 += a0 * deq0_n0;
                acc_r0_n0 += a1 * deq1_n0;
                acc_r0_n0 += a2 * deq2_n0;
                acc_r0_n0 += a3 * deq3_n0;

                acc_r0_n1 += a0 * deq0_n1;
                acc_r0_n1 += a1 * deq1_n1;
                acc_r0_n1 += a2 * deq2_n1;
                acc_r0_n1 += a3 * deq3_n1;
            }

            if row1_valid {
                let a_line = sm_input[a_base1 + t4];
                let a0 = Line::<F>::empty(shape.line_size).fill(a_line[0]);
                let a1 = Line::<F>::empty(shape.line_size).fill(a_line[1]);
                let a2 = Line::<F>::empty(shape.line_size).fill(a_line[2]);
                let a3 = Line::<F>::empty(shape.line_size).fill(a_line[3]);

                acc_r1_n0 += a0 * deq0_n0;
                acc_r1_n0 += a1 * deq1_n0;
                acc_r1_n0 += a2 * deq2_n0;
                acc_r1_n0 += a3 * deq3_n0;

                acc_r1_n1 += a0 * deq0_n1;
                acc_r1_n1 += a1 * deq1_n1;
                acc_r1_n1 += a2 * deq2_n1;
                acc_r1_n1 += a3 * deq3_n1;
            }
        }

        // Update zeros/scales cache for the next tile only after current compute is done.
        if has_next && unit_pos < packs_per_row && (next_k % tiles_per_group) == 0 {
            let pack = unit_pos;
            let unit_n = pack * 8;
            let k_tile0 = next_k * shape.inner_k;
            let group_k = k_tile0 / group_size;

            let qzeros_idx = group_k * shape.n + nc + unit_n;
            sm_zero_pack[pack] = qzeros[qzeros_idx / 8];

            let scale_idx = qzeros_idx / shape.line_size;
            let pack_line = pack * 2;
            sm_scale_line[pack_line] = scales[scale_idx];
            sm_scale_line[pack_line + 1] = scales[scale_idx + 1];
        }

        // Barrier: publish next buffer before swapping
        sync_cube();
    }

    // ===================================================================
    // Write results: 2 rows × 2 Lines each
    // ===================================================================
    if row0_valid {
        let global_row = mc + row0;
        let col0 = nc + n_line0 * shape.line_size;
        let col1 = nc + n_line1 * shape.line_size;
        output[(global_row * shape.n + col0) / shape.line_size] = acc_r0_n0;
        output[(global_row * shape.n + col1) / shape.line_size] = acc_r0_n1;
    }

    if row1_valid {
        let global_row = mc + row1;
        let col0 = nc + n_line0 * shape.line_size;
        let col1 = nc + n_line1 * shape.line_size;
        output[(global_row * shape.n + col0) / shape.line_size] = acc_r1_n0;
        output[(global_row * shape.n + col1) / shape.line_size] = acc_r1_n1;
    }
}

// Decode-specialized path:
//   - ROWS_PER_THREAD = 1 for better utilization when M is tiny.
//   - N_LINES_PER_THREAD = 4 for wider N blocking.
//   - Single-buffered SMEM to cut per-cube SMEM footprint.
#[cube(launch)]
pub fn gemm_awq_decode<F: Float>(
    input: &LinearView<Line<F>>,
    qweight: &LinearView<u32>,
    qzeros: &LinearView<u32>,
    scales: &LinearView<Line<F>>,
    output: &mut LinearView<Line<F>, ReadWrite>,
    #[comptime] shape: &ShapeConfig,
    #[comptime] group_size: usize,
) {
    let inner_k_lines = shape.inner_k / shape.line_size;
    let tile_n_lines = shape.tile_n / shape.line_size;
    let n_groups = tile_n_lines / N_LINES_PER_THREAD_DECODE;
    let packs_per_row = shape.tile_n / 8;
    // K-major packed weight layout: [k][pack]
    let sm_weight_stride_packs = packs_per_row;
    let sm_input_buf_size = shape.tile_m * inner_k_lines;
    let sm_weight_buf_size = shape.inner_k * sm_weight_stride_packs;

    let mut sm_input = SharedMemory::<F>::new_lined(sm_input_buf_size, shape.line_size);
    let mut sm_weight = SharedMemory::<u32>::new(sm_weight_buf_size);
    let mut sm_zero_pack = SharedMemory::<u32>::new(shape.tile_n / 8);
    let mut sm_scale_line = SharedMemory::<F>::new_lined(tile_n_lines, shape.line_size);

    // Tile origin
    let mc = (CUBE_POS / (shape.n / shape.tile_n)) * shape.tile_m;
    let nc = (CUBE_POS % (shape.n / shape.tile_n)) * shape.tile_n;
    let k_loop = shape.k / shape.inner_k;

    let unit_pos = UNIT_POS as usize;
    let thread_rows = shape.tile_m / ROWS_PER_THREAD_DECODE;
    let idx_n_group = unit_pos % n_groups;
    let idx_tile_m = unit_pos / n_groups;
    let total_threads = thread_rows * n_groups;

    let n_line0 = idx_n_group * N_LINES_PER_THREAD_DECODE;
    let n_line1 = n_line0 + 1;
    let n_line2 = n_line0 + 2;
    let n_line3 = n_line0 + 3;

    let row = idx_tile_m;
    let row_valid = (mc + row) < shape.m_valid;

    let zero_val = F::cast_from(0);
    let mut acc_n0 = Line::<F>::empty(shape.line_size).fill(zero_val);
    let mut acc_n1 = Line::<F>::empty(shape.line_size).fill(zero_val);
    let mut acc_n2 = Line::<F>::empty(shape.line_size).fill(zero_val);
    let mut acc_n3 = Line::<F>::empty(shape.line_size).fill(zero_val);

    // Fast path only: each K tile stays inside a single AWQ group.
    let tiles_per_group = group_size / shape.inner_k;

    for idx_k in 0..k_loop {
        if unit_pos < packs_per_row && (idx_k % tiles_per_group) == 0 {
            let pack = unit_pos;
            let unit_n = pack * 8;
            let k_tile0 = idx_k * shape.inner_k;
            let group_k = k_tile0 / group_size;

            let qzeros_idx = group_k * shape.n + nc + unit_n;
            sm_zero_pack[pack] = qzeros[qzeros_idx / 8];

            let scale_idx = qzeros_idx / shape.line_size;
            let pack_line = pack * 2;
            sm_scale_line[pack_line] = scales[scale_idx];
            sm_scale_line[pack_line + 1] = scales[scale_idx + 1];
        }

        let mut t = unit_pos;
        while t < sm_input_buf_size {
            let unit_m = t / inner_k_lines;
            let unit_k_line = t % inner_k_lines;
            if (mc + unit_m) < shape.m_valid {
                let unit_k = unit_k_line * shape.line_size;
                let input_idx = (mc + unit_m) * shape.k + idx_k * shape.inner_k + unit_k;
                sm_input[unit_m * inner_k_lines + unit_k_line] = input[input_idx / shape.line_size];
            }
            t += total_threads;
        }

        let mut w = unit_pos;
        while w < sm_weight_buf_size {
            let unit_k = w / packs_per_row;
            let pack = w % packs_per_row;
            let unit_n = pack * 8;
            let k_tmp = idx_k * shape.inner_k + unit_k;
            let qweight_idx = k_tmp * shape.n + nc + unit_n;
            let line_base = unit_k * sm_weight_stride_packs;
            sm_weight[line_base + pack] = qweight[qweight_idx / 8];
            w += total_threads;
        }

        sync_cube();

        let a_base = row * inner_k_lines;
        let pack_idx0 = n_line0 / 2;
        let pack_idx1 = pack_idx0 + 1;
        let packed_z0 = sm_zero_pack[pack_idx0];
        let packed_z1 = sm_zero_pack[pack_idx1];
        let sc_n0 = sm_scale_line[n_line0];
        let sc_n1 = sm_scale_line[n_line1];
        let sc_n2 = sm_scale_line[n_line2];
        let sc_n3 = sm_scale_line[n_line3];
        let zero_n0 = unpack_u32_to_i4_0_4::<F>(packed_z0);
        let zero_n1 = unpack_u32_to_i4_4_8::<F>(packed_z0);
        let zero_n2 = unpack_u32_to_i4_0_4::<F>(packed_z1);
        let zero_n3 = unpack_u32_to_i4_4_8::<F>(packed_z1);

        if row_valid {
            #[unroll]
            for t4 in 0..inner_k_lines {
                let kk = t4 * shape.line_size;

                let packed_w0_n01 = sm_weight[(kk + 0) * sm_weight_stride_packs + pack_idx0];
                let packed_w1_n01 = sm_weight[(kk + 1) * sm_weight_stride_packs + pack_idx0];
                let packed_w2_n01 = sm_weight[(kk + 2) * sm_weight_stride_packs + pack_idx0];
                let packed_w3_n01 = sm_weight[(kk + 3) * sm_weight_stride_packs + pack_idx0];
                let packed_w0_n23 = sm_weight[(kk + 0) * sm_weight_stride_packs + pack_idx1];
                let packed_w1_n23 = sm_weight[(kk + 1) * sm_weight_stride_packs + pack_idx1];
                let packed_w2_n23 = sm_weight[(kk + 2) * sm_weight_stride_packs + pack_idx1];
                let packed_w3_n23 = sm_weight[(kk + 3) * sm_weight_stride_packs + pack_idx1];

                let deq0_n0 = (unpack_u32_to_i4_0_4::<F>(packed_w0_n01) - zero_n0) * sc_n0;
                let deq1_n0 = (unpack_u32_to_i4_0_4::<F>(packed_w1_n01) - zero_n0) * sc_n0;
                let deq2_n0 = (unpack_u32_to_i4_0_4::<F>(packed_w2_n01) - zero_n0) * sc_n0;
                let deq3_n0 = (unpack_u32_to_i4_0_4::<F>(packed_w3_n01) - zero_n0) * sc_n0;
                let deq0_n1 = (unpack_u32_to_i4_4_8::<F>(packed_w0_n01) - zero_n1) * sc_n1;
                let deq1_n1 = (unpack_u32_to_i4_4_8::<F>(packed_w1_n01) - zero_n1) * sc_n1;
                let deq2_n1 = (unpack_u32_to_i4_4_8::<F>(packed_w2_n01) - zero_n1) * sc_n1;
                let deq3_n1 = (unpack_u32_to_i4_4_8::<F>(packed_w3_n01) - zero_n1) * sc_n1;
                let deq0_n2 = (unpack_u32_to_i4_0_4::<F>(packed_w0_n23) - zero_n2) * sc_n2;
                let deq1_n2 = (unpack_u32_to_i4_0_4::<F>(packed_w1_n23) - zero_n2) * sc_n2;
                let deq2_n2 = (unpack_u32_to_i4_0_4::<F>(packed_w2_n23) - zero_n2) * sc_n2;
                let deq3_n2 = (unpack_u32_to_i4_0_4::<F>(packed_w3_n23) - zero_n2) * sc_n2;
                let deq0_n3 = (unpack_u32_to_i4_4_8::<F>(packed_w0_n23) - zero_n3) * sc_n3;
                let deq1_n3 = (unpack_u32_to_i4_4_8::<F>(packed_w1_n23) - zero_n3) * sc_n3;
                let deq2_n3 = (unpack_u32_to_i4_4_8::<F>(packed_w2_n23) - zero_n3) * sc_n3;
                let deq3_n3 = (unpack_u32_to_i4_4_8::<F>(packed_w3_n23) - zero_n3) * sc_n3;

                let a_line = sm_input[a_base + t4];
                let a0 = Line::<F>::empty(shape.line_size).fill(a_line[0]);
                let a1 = Line::<F>::empty(shape.line_size).fill(a_line[1]);
                let a2 = Line::<F>::empty(shape.line_size).fill(a_line[2]);
                let a3 = Line::<F>::empty(shape.line_size).fill(a_line[3]);

                acc_n0 += a0 * deq0_n0;
                acc_n0 += a1 * deq1_n0;
                acc_n0 += a2 * deq2_n0;
                acc_n0 += a3 * deq3_n0;
                acc_n1 += a0 * deq0_n1;
                acc_n1 += a1 * deq1_n1;
                acc_n1 += a2 * deq2_n1;
                acc_n1 += a3 * deq3_n1;
                acc_n2 += a0 * deq0_n2;
                acc_n2 += a1 * deq1_n2;
                acc_n2 += a2 * deq2_n2;
                acc_n2 += a3 * deq3_n2;
                acc_n3 += a0 * deq0_n3;
                acc_n3 += a1 * deq1_n3;
                acc_n3 += a2 * deq2_n3;
                acc_n3 += a3 * deq3_n3;
            }
        }

        // Ensure all threads finished reading current shared tiles before overwrite.
        sync_cube();
    }

    if row_valid {
        let global_row = mc + row;
        let col0 = nc + n_line0 * shape.line_size;
        let col1 = nc + n_line1 * shape.line_size;
        let col2 = nc + n_line2 * shape.line_size;
        let col3 = nc + n_line3 * shape.line_size;
        output[(global_row * shape.n + col0) / shape.line_size] = acc_n0;
        output[(global_row * shape.n + col1) / shape.line_size] = acc_n1;
        output[(global_row * shape.n + col2) / shape.line_size] = acc_n2;
        output[(global_row * shape.n + col3) / shape.line_size] = acc_n3;
    }
}

pub fn awq_gemm_linear<R: Runtime, F: Float>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    qweight: &TensorHandleRef<R>,
    qzeros: &TensorHandleRef<R>,
    scales: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    shape: &ShapeConfig,
    group_size: usize,
) -> Result<(), LaunchError> {
    assert!(
        input.shape.len() >= 2,
        "input must have shape [.., M, K] or [M, K]"
    );
    assert!(qweight.shape.len() >= 2, "qweight must have shape [K, N/8]");
    assert!(qzeros.shape.len() >= 2, "qzeros must have shape [K/G, N/8]");
    assert!(scales.shape.len() >= 2, "scales must have shape [K/G, N]");
    assert!(
        output.shape.len() >= 2,
        "output must have shape [.., M, N] or [M, N]"
    );

    let input_k = *input.shape.last().expect("input rank checked");
    assert!(input_k > 0, "input K must be > 0");
    assert_eq!(input_k, shape.k, "input K must match shape.k");

    let input_numel = input.shape.iter().product::<usize>();
    let launch_m = input_numel / input_k;
    assert_eq!(
        input_numel % input_k,
        0,
        "input elements must be divisible by K"
    );

    let output_n = *output.shape.last().expect("output rank checked");
    assert!(output_n > 0, "output N must be > 0");
    assert_eq!(output_n, shape.n, "output N must match shape.n");
    let output_numel = output.shape.iter().product::<usize>();
    let output_rows = output_numel / output_n;
    assert_eq!(
        output_numel % output_n,
        0,
        "output elements must be divisible by N"
    );
    assert_eq!(
        output_rows, launch_m,
        "output rows must match flattened input rows"
    );

    let launch_m_valid = if shape.m == launch_m {
        shape.m_valid
    } else if shape.m > 0 && launch_m % shape.m == 0 && shape.m_valid == shape.m {
        launch_m
    } else {
        launch_m.min(shape.m_valid)
    };
    let launch_shape = ShapeConfig::new_with_valid_m(
        launch_m,
        shape.n,
        shape.k,
        shape.tile_m,
        shape.tile_n,
        shape.inner_k,
        launch_m_valid,
    );
    let use_decode_kernel = launch_shape.m <= DECODE_M_THRESHOLD;
    if use_decode_kernel {
        validate_decode_launch_params(&launch_shape, group_size);
    } else {
        validate_launch_params(&launch_shape, group_size);
    }

    let num_groups = launch_shape.k / group_size;
    assert_eq!(
        qweight.shape[0], launch_shape.k,
        "qweight K must match shape.k"
    );
    assert_eq!(
        qweight.shape[1],
        launch_shape.n / 8,
        "qweight packed N must be N/8"
    );
    assert_eq!(
        qzeros.shape[0], num_groups,
        "qzeros rows must be K/group_size"
    );
    assert_eq!(
        qzeros.shape[1],
        launch_shape.n / 8,
        "qzeros packed N must be N/8"
    );
    assert_eq!(
        scales.shape[0], num_groups,
        "scales rows must be K/group_size"
    );
    assert_eq!(
        scales.shape[1], launch_shape.n,
        "scales N must match shape.n"
    );

    let m_tiles = launch_shape.m.div_ceil(launch_shape.tile_m);
    let n_tiles = launch_shape.n / launch_shape.tile_n;
    let total_cubes = m_tiles * n_tiles;

    if use_decode_kernel {
        let thread_rows = launch_shape.tile_m / ROWS_PER_THREAD_DECODE;
        let n_groups = (launch_shape.tile_n / launch_shape.line_size) / N_LINES_PER_THREAD_DECODE;
        let threads_per_cube = thread_rows * n_groups;
        let cube_dim = CubeDim::new_2d(n_groups as u32, thread_rows as u32);
        let cube_count =
            calculate_cube_count_elemwise(client, total_cubes * threads_per_cube, cube_dim);

        gemm_awq_decode::launch::<F, R>(
            client,
            cube_count,
            cube_dim,
            linear_view(client, input, launch_shape.line_size),
            linear_view(client, qweight, 1),
            linear_view(client, qzeros, 1),
            linear_view(client, scales, launch_shape.line_size),
            linear_view(client, output, launch_shape.line_size),
            launch_shape,
            group_size,
        )
    } else {
        let thread_rows = launch_shape.tile_m / ROWS_PER_THREAD;
        let n_groups = (launch_shape.tile_n / launch_shape.line_size) / N_LINES_PER_THREAD;
        let threads_per_cube = thread_rows * n_groups;
        let cube_dim = CubeDim::new_2d(n_groups as u32, thread_rows as u32);
        let cube_count =
            calculate_cube_count_elemwise(client, total_cubes * threads_per_cube, cube_dim);

        gemm_awq::launch::<F, R>(
            client,
            cube_count,
            cube_dim,
            linear_view(client, input, launch_shape.line_size),
            linear_view(client, qweight, 1),
            linear_view(client, qzeros, 1),
            linear_view(client, scales, launch_shape.line_size),
            linear_view(client, output, launch_shape.line_size),
            launch_shape,
            group_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::bytes::Bytes;
    use std::marker::PhantomData;
    // ─── helpers ────────────────────────────────────────────────────────────

    /// Pack 8 nibbles (each 0..15) into one u32 (nibble-0 in bits 3..0).
    fn pack8(n: &[u8; 8]) -> u32 {
        n.iter()
            .enumerate()
            .fold(0u32, |v, (i, &x)| v | (((x & 0xF) as u32) << (i * 4)))
    }

    /// Every nibble in the u32 is the same value `v`.
    fn uniform_u32(v: u8) -> u32 {
        pack8(&[v; 8])
    }

    /// CPU-side AWQ dequant + matmul (golden reference).
    fn cpu_ref(
        inp: &[f32],
        qw: &[u32],
        qz: &[u32],
        sc: &[f32],
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        gs: usize,
    ) -> Vec<f32> {
        let npk = n / 8;
        let mut w = vec![0f32; k * n];
        for r in 0..k {
            let g = r / gs;
            for c in 0..n {
                const AWQ_REVERSE_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
                let packed_lane = AWQ_REVERSE_ORDER[c % 8];
                let wv = ((qw[r * npk + c / 8] >> (packed_lane * 4)) & 0xF) as f32;
                let zv = ((qz[g * npk + c / 8] >> (packed_lane * 4)) & 0xF) as f32;
                w[r * n + c] = (wv - zv) * sc[g * n + c];
            }
        }
        let mut out = vec![0f32; b * m * n];
        for bi in 0..b {
            for mi in 0..m {
                for ni in 0..n {
                    let mut a = 0f32;
                    for ki in 0..k {
                        a += inp[bi * m * k + mi * k + ki] * w[ki * n + ni];
                    }
                    out[bi * m * n + mi * n + ni] = a;
                }
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let d = (a - e).abs();
            assert!(
                d <= tol,
                "[{i}] actual={a} expected={e} diff={d} > tol={tol}"
            );
        }
    }

    /// Build uniform qweight: every nibble == `nib`, shape K × N/8 u32.
    fn make_uniform_qw(k: usize, n: usize, nib: u8) -> Vec<u32> {
        vec![uniform_u32(nib); k * (n / 8)]
    }

    /// Build uniform qzeros: every nibble == `nib`, shape (K/gs) × (N/8) u32.
    fn make_uniform_qz(k: usize, n: usize, gs: usize, nib: u8) -> Vec<u32> {
        vec![uniform_u32(nib); (k / gs) * (n / 8)]
    }

    /// Uniform scales, shape (K/gs) × N.
    fn make_uniform_sc(k: usize, n: usize, gs: usize, val: f32) -> Vec<f32> {
        vec![val; (k / gs) * n]
    }

    // ─── GPU harness ────────────────────────────────────────────────────────
    // Adapt the body of `run_kernel` to your CubeCL version / backend.
    // The logic for every #[test] below is self-contained regardless.

    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    type R = WgpuRuntime;

    fn run_kernel(
        inp: &[f32],
        qw: &[u32],
        qz: &[u32],
        sc: &[f32],
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        gs: usize,
    ) -> Vec<f32> {
        let client = R::client(&WgpuDevice::default());
        let rows = b * m;
        let inner_k = if k >= 64 && k.is_multiple_of(64) && gs >= 64 && gs.is_multiple_of(64) {
            64
        } else {
            32
        };
        assert!(
            gs >= inner_k && gs.is_multiple_of(inner_k),
            "test setup must satisfy fast-path constraints"
        );
        let tile_n = if n >= 128 && n.is_multiple_of(128) {
            128
        } else if n >= 64 && n.is_multiple_of(64) {
            64
        } else {
            32
        };
        let shape = ShapeConfig::new(rows, n, k, 32, tile_n, inner_k);

        let ih = client.create(Bytes::from_bytes_vec(bytemuck::cast_slice(inp).to_vec()));
        let wh = client.create(Bytes::from_bytes_vec(bytemuck::cast_slice(qw).to_vec()));
        let zh = client.create(Bytes::from_bytes_vec(bytemuck::cast_slice(qz).to_vec()));
        let sh = client.create(Bytes::from_bytes_vec(bytemuck::cast_slice(sc).to_vec()));
        let oh = client.empty(rows * n * core::mem::size_of::<f32>());

        // Shapes/strides expected by linear_view - row-major, element-granularity.
        let i_shape = [rows, k];
        let i_str = [k, 1];
        let w_shape = [k, n / 8];
        let w_str = [n / 8, 1];
        let z_shape = [k / gs, n / 8];
        let z_str = [n / 8, 1];
        let s_shape = [k / gs, n];
        let s_str = [n, 1];
        let o_shape = [rows, n];
        let o_str = [n, 1];

        let ir = TensorHandleRef {
            handle: &ih,
            strides: &i_str,
            shape: &i_shape,
            elem_size: core::mem::size_of::<f32>(),
            runtime: PhantomData,
        };
        let wr = TensorHandleRef {
            handle: &wh,
            strides: &w_str,
            shape: &w_shape,
            elem_size: core::mem::size_of::<u32>(),
            runtime: PhantomData,
        };
        let zr = TensorHandleRef {
            handle: &zh,
            strides: &z_str,
            shape: &z_shape,
            elem_size: core::mem::size_of::<u32>(),
            runtime: PhantomData,
        };
        let sr = TensorHandleRef {
            handle: &sh,
            strides: &s_str,
            shape: &s_shape,
            elem_size: core::mem::size_of::<f32>(),
            runtime: PhantomData,
        };
        let or = TensorHandleRef {
            handle: &oh,
            strides: &o_str,
            shape: &o_shape,
            elem_size: core::mem::size_of::<f32>(),
            runtime: PhantomData,
        };

        awq_gemm_linear::<R, f32>(&client, &ir, &wr, &zr, &sr, &or, &shape, gs)
            .expect("kernel launch failed");

        let bytes = client.read_one(oh.clone()).to_vec();
        bytemuck::cast_slice(&bytes).to_vec()
    }
    // ═══════════════════════════════════════════════════════════════════════
    //  A.  VALIDATION / LAUNCH-GUARD TESTS  (CPU only, no GPU)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_accepts_minimal_valid_config() {
        let s = ShapeConfig::new(32, 32, 32, 32, 32, 32);
        validate_launch_params(&s, 32); // should not panic
    }

    #[test]
    fn validate_accepts_large_multi_tile_config() {
        let s = ShapeConfig::new(128, 256, 512, 32, 32, 32);
        validate_launch_params(&s, 128);
    }

    #[test]
    #[should_panic(expected = "line_size must be > 0")]
    fn validate_rejects_line_size_zero() {
        let mut s = ShapeConfig::new(32, 32, 32, 32, 32, 32);
        s.line_size = 0; // force via field access inside test module
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "m_tile_size must be > 0")]
    fn validate_rejects_m_tile_zero() {
        let s = ShapeConfig::new(32, 32, 32, 0, 32, 32);
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "n_tile_size must be > 0")]
    fn validate_rejects_n_tile_zero() {
        let s = ShapeConfig::new(32, 32, 32, 32, 0, 32);
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "inner_k_size must be > 0")]
    fn validate_rejects_inner_k_zero() {
        let s = ShapeConfig::new(32, 32, 32, 32, 32, 0);
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "group_size must be > 0")]
    fn validate_rejects_group_size_zero() {
        let s = ShapeConfig::new(32, 32, 32, 32, 32, 32);
        validate_launch_params(&s, 0);
    }

    #[test]
    #[should_panic(expected = "k must be divisible by line_size")]
    fn validate_rejects_k_not_aligned_to_line_size() {
        // k=6, line_size=4  → 6 % 4 ≠ 0
        let s = ShapeConfig::new(32, 32, 6, 32, 32, 4);
        validate_launch_params(&s, 2);
    }

    #[test]
    #[should_panic(expected = "n must be divisible by line_size")]
    fn validate_rejects_n_not_aligned_to_line_size() {
        // n=6, line_size=4
        let s = ShapeConfig::new(32, 6, 32, 32, 32, 32);
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "inner_k_size must be divisible by line_size")]
    fn validate_rejects_inner_k_not_aligned_to_line_size() {
        // inner_k=6, line_size=4  → 6 % 4 ≠ 0
        let s = ShapeConfig::new(32, 32, 32, 32, 32, 6);
        validate_launch_params(&s, 32);
    }

    #[test]
    fn validate_accepts_m_not_multiple_of_m_tile() {
        // Partial M tiles are allowed; out-of-range rows are masked by m_valid.
        let s = ShapeConfig::new(48, 32, 32, 32, 32, 32);
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "n must be divisible by n_tile_size")]
    fn validate_rejects_n_not_multiple_of_n_tile() {
        // n=48, n_tile=32  → 48 % 32 ≠ 0
        let s = ShapeConfig::new(32, 48, 32, 32, 32, 32);
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "k must be divisible by inner_k_size")]
    fn validate_rejects_k_not_multiple_of_inner_k() {
        // k=48, inner_k=32  → 48 % 32 ≠ 0
        let s = ShapeConfig::new(32, 32, 48, 32, 32, 32);
        validate_launch_params(&s, 16);
    }

    #[test]
    #[should_panic(expected = "n_tile_size must be divisible by line_size*8")]
    fn validate_rejects_n_tile_not_multiple_of_32() {
        // n_tile=16, line_size*8=32 → 16 % 32 ≠ 0
        let s = ShapeConfig::new(32, 16, 32, 32, 16, 32);
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "k must be divisible by group_size")]
    fn validate_rejects_k_not_multiple_of_group_size() {
        // k=32, group_size=24  → 32 % 24 ≠ 0
        let s = ShapeConfig::new(32, 32, 32, 32, 32, 32);
        validate_launch_params(&s, 24);
    }

    #[test]
    #[should_panic(expected = "m_valid must be <= m")]
    fn validate_rejects_m_valid_larger_than_padded_m() {
        let s = ShapeConfig::new_with_valid_m(32, 32, 32, 32, 32, 32, 33);
        validate_launch_params(&s, 32);
    }

    #[test]
    #[should_panic(expected = "m must be > 0")]
    fn validate_rejects_m_zero() {
        let s = ShapeConfig::new(0, 32, 32, 32, 32, 32);
        validate_launch_params(&s, 32);
    }

    // edge: every dimension at the exact minimum that passes
    #[test]
    fn validate_exact_boundary_all_equal() {
        // m = m_tile, n = n_tile = 32, k = inner_k = group_size = 32
        let s = ShapeConfig::new(32, 32, 32, 32, 32, 32);
        validate_launch_params(&s, 32);
    }

    // edge: group_size == k  (single quant group)
    #[test]
    fn validate_group_size_equals_k() {
        let s = ShapeConfig::new(32, 32, 64, 32, 32, 32);
        validate_launch_params(&s, 64); // gs=k=64
    }

    #[test]
    #[should_panic(expected = "fast path requires group_size")]
    fn validate_rejects_group_size_smaller_than_inner_k() {
        let s = ShapeConfig::new(32, 32, 32, 32, 32, 32);
        validate_launch_params(&s, 1);
    }

    #[test]
    #[should_panic(expected = "fast path requires group_size")]
    fn validate_rejects_group_size_not_multiple_of_inner_k() {
        let s = ShapeConfig::new(32, 32, 96, 32, 32, 32);
        validate_launch_params(&s, 48);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  B.  KERNEL CORRECTNESS TESTS  (need GPU)
    // ═══════════════════════════════════════════════════════════════════════

    // ----- B.1  trivial / zero cases -----

    #[test]
    fn kernel_zero_input_yields_zero_output() {
        let (b, m, k, n, gs) = (1, 32, 32, 32, 32);
        let inp = vec![0f32; b * m * k];
        let qw = make_uniform_qw(k, n, 7); // non-zero weight
        let qz = make_uniform_qz(k, n, gs, 0);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &vec![0f32; b * m * n], 1e-5);
    }

    #[test]
    fn kernel_zero_weight_yields_zero_output() {
        // w nibble == z nibble -> dequantized weight is 0
        let (b, m, k, n, gs) = (1, 32, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 5);
        let qz = make_uniform_qz(k, n, gs, 5); // same -> w-z=0
        let sc = make_uniform_sc(k, n, gs, 2.0);

        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &vec![0f32; b * m * n], 1e-5);
    }

    #[test]
    fn kernel_zero_scale_yields_zero_output() {
        let (b, m, k, n, gs) = (1, 32, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 15);
        let qz = make_uniform_qz(k, n, gs, 0);
        let sc = make_uniform_sc(k, n, gs, 0.0); // scale=0

        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &vec![0f32; b * m * n], 1e-5);
    }

    // ----- B.2  constant / uniform cases -----

    #[test]
    fn kernel_ones_input_uniform_weight_single_tile() {
        // input=1, w_nibble=5, z_nibble=0, scale=1  -> deq=5
        // each output elem = K * 1 * 5 = 160
        let (b, m, k, n, gs) = (1, 32, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 5);
        let qz = make_uniform_qz(k, n, gs, 0);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn kernel_uniform_with_nonzero_zero_point() {
        // w=8, z=3, scale=2  -> deq = (8-3)*2 = 10
        // each output elem = K * 1 * 10 = 320
        let (b, m, k, n, gs) = (1, 32, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 8);
        let qz = make_uniform_qz(k, n, gs, 3);
        let sc = make_uniform_sc(k, n, gs, 2.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn kernel_negative_effective_weight() {
        // w=2, z=7 -> deq = (2-7)*1 = -5. output elem = -160
        let (b, m, k, n, gs) = (1, 32, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 2);
        let qz = make_uniform_qz(k, n, gs, 7);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn kernel_weight_nibble_at_max_15() {
        // w=15 (max unsigned i4), z=0, scale=1 -> deq=15
        let (b, m, k, n, gs) = (1, 32, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 15);
        let qz = make_uniform_qz(k, n, gs, 0);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn kernel_weight_nibble_zero_zeropoint_fifteen() {
        // w=0, z=15 -> deq = (0-15)*1 = -15 (maximally negative)
        let (b, m, k, n, gs) = (1, 32, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 0);
        let qz = make_uniform_qz(k, n, gs, 15);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }

    // ----- B.3  multiple K-tiles -----

    #[test]
    fn kernel_two_k_tiles() {
        // K=64, inner_k=32 → 2 iterations of the k-loop
        let (b, m, k, n, gs) = (1, 32, 64, 32, 64);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 4);
        let qz = make_uniform_qz(k, n, gs, 1);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-2);
    }

    #[test]
    fn kernel_four_k_tiles() {
        // K=128, inner_k=32 → 4 iterations
        let (b, m, k, n, gs) = (1, 32, 128, 32, 128);
        let inp = vec![0.5f32; b * m * k];
        let qw = make_uniform_qw(k, n, 6);
        let qz = make_uniform_qz(k, n, gs, 2);
        let sc = make_uniform_sc(k, n, gs, 0.25);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-2);
    }

    // ----- B.4  multiple quantisation groups -----

    #[test]
    fn kernel_two_groups_uniform_scale() {
        // K=64, group_size=32 → 2 groups, same scale
        let (b, m, k, n, gs) = (1, 32, 64, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 5);
        let qz = make_uniform_qz(k, n, gs, 1);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-2);
    }

    #[test]
    fn kernel_two_groups_different_scales() {
        // group 0 → scale=1.0,  group 1 → scale=0.5
        let (b, m, k, n, gs) = (1, 32, 64, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 6);
        let qz = make_uniform_qz(k, n, gs, 2);
        // 2 groups × N scales
        let mut sc = vec![0f32; 2 * n];
        sc[..n].fill(1.0); // group 0
        sc[n..].fill(0.5); // group 1

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-2);
    }

    #[test]
    fn kernel_four_groups() {
        // K=128, group_size=32 → 4 groups
        let (b, m, k, n, gs) = (1, 32, 128, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 7);
        let qz = make_uniform_qz(k, n, gs, 3);
        let ngrp = k / gs;
        let mut sc = vec![0f32; ngrp * n];
        for g in 0..ngrp {
            let v = (g + 1) as f32 * 0.25; // 0.25, 0.5, 0.75, 1.0
            sc[g * n..(g + 1) * n].fill(v);
        }

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-1);
    }

    // ----- B.5  non-square M / N (multiple spatial tiles) -----

    #[test]
    fn kernel_m_double_n_single_tile() {
        // M=64 → 2 m-tiles,  N=32 → 1 n-tile
        let (b, m, k, n, gs) = (1, 64, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 5);
        let qz = make_uniform_qz(k, n, gs, 0);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn kernel_m_one_partial_tile() {
        // M=1 is the decode path; kernel must handle partial m-tile correctly.
        let (b, m, k, n, gs) = (1, 1, 32, 32, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 5);
        let qz = make_uniform_qz(k, n, gs, 0);
        let sc = make_uniform_sc(k, n, gs, 1.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn kernel_n_double_m_single_tile() {
        // M=32 → 1 m-tile,  N=64 → 2 n-tiles
        let (b, m, k, n, gs) = (1, 32, 32, 64, 32);
        let inp = vec![1f32; b * m * k];
        let qw = make_uniform_qw(k, n, 3);
        let qz = make_uniform_qz(k, n, gs, 1);
        let sc = make_uniform_sc(k, n, gs, 2.0);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn kernel_large_m_large_n() {
        // M=128, N=128 → 4×4 tiles
        let (b, m, k, n, gs) = (1, 128, 32, 128, 32);
        let inp: Vec<f32> = (0..b * m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
        let qw = make_uniform_qw(k, n, 4);
        let qz = make_uniform_qz(k, n, gs, 2);
        let sc = make_uniform_sc(k, n, gs, 0.5);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-1);
    }

    #[test]
    fn kernel_two_batches_match_cpu_ref() {
        // Ensure launch grid covers all batches, not only b=0.
        let (b, m, k, n, gs) = (2, 32, 32, 32, 32);
        let mut inp = vec![0f32; b * m * k];
        inp[..m * k].fill(1.0);
        inp[m * k..].fill(2.0);

        let qw = make_uniform_qw(k, n, 5);
        let qz = make_uniform_qz(k, n, gs, 1);
        let sc = make_uniform_sc(k, n, gs, 0.5);

        let expected = cpu_ref(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        let out = run_kernel(&inp, &qw, &qz, &sc, b, m, k, n, gs);
        assert_close(&out, &expected, 1e-3);
    }
}
