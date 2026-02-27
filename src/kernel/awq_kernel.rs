use cubecl::prelude::*;
use cubecl::{
    calculate_cube_count_elemwise,
    std::tensor::layout::linear::{linear_view, LinearView},
};

// The helper func
#[cube]
fn unpack_u32<F: Float>(packed_value: u32, #[comptime] size: usize) -> Line<F> {
    let mut unpacked_line = Line::<F>::empty(size);

    #[unroll]
    for idx in 0..size {
        let shift = idx * 4;

        unpacked_line[idx] = F::cast_from((packed_value >> shift) & 0x0f);
    }
    unpacked_line
}

// Working Unit
//the line size is 4
#[cube(launch_unchecked)]
fn dequantize_awq_symmetric<F: Float>(
    weights: &LinearView<Line<u32>>,
    zeros: &LinearView<Line<u32>>,
    scales: &LinearView<Line<F>>,
    output: &mut LinearView<Line<F>, ReadWrite>,
    out_width: usize,
    #[comptime] group_size: usize,
) {
    //Output (full_width, full_height)
    //weight (full_width, packed_height)
    let element_per_thread = 32;
    let num_packed = 8usize;
    //each unit processes a line
    // ABSOLUTE_POS is the general position for units
    let line = weights[ABSOLUTE_POS]; //take a line
    comptime! {assert!(line.size()==4usize)};

    let weight_row_stride = out_width / element_per_thread;
    let row_idx = ABSOLUTE_POS / weight_row_stride;
    let col_block_idx = ABSOLUTE_POS % weight_row_stride;

    let group_idx = row_idx / group_size;
    let scale_row_stride = out_width / 8;

    let zero = zeros[group_idx * weight_row_stride + col_block_idx];
    #[unroll]
    for i in 0..line.line_size() {
        let value = line[i];
        let out = unpack_u32::<F>(value, num_packed);
        comptime! {assert!(out.size()==num_packed)};

        let scale_line = scales[group_idx * scale_row_stride + col_block_idx * 4 + i];

        let z = unpack_u32(zero[i], num_packed);
        //8
        output[ABSOLUTE_POS * 4 + i] = (out - z) * scale_line; //这里的out是line的，不知道这么一次性写入对不对的，没有报错
    }
}

pub fn dequantize_native<R: Runtime, E: Float>(
    client: &ComputeClient<R>,
    q_weight: &TensorHandleRef<R>,
    q_zeros: &TensorHandleRef<R>,
    q_scales: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    group_size: usize,
) -> Result<(), LaunchError> {
    // We treat q_weight/q_zeros as packed 4-bit values inside 32-bit lanes.
    // Mixing element sizes (e.g. i16/i64 backends) will silently corrupt indexing/loads.
    assert_eq!(
        q_weight.elem_size,
        core::mem::size_of::<u32>(),
        "AWQ packed q_weight must use 32-bit elements (i32/u32), got elem_size={}.",
        q_weight.elem_size
    );
    assert_eq!(
        q_zeros.elem_size,
        core::mem::size_of::<u32>(),
        "AWQ packed q_zeros must use 32-bit elements (i32/u32), got elem_size={}.",
        q_zeros.elem_size
    );

    let out_width = *output.shape.last().unwrap();
    assert!(out_width % 32 == 0);

    let weight_linear = linear_view(client, q_weight, 4);

    let num_qweight = q_weight.size() / 4;

    let cube_dim = CubeDim::new(client, 256);
    let cube_count = calculate_cube_count_elemwise(client, num_qweight, cube_dim);

    unsafe {
        dequantize_awq_symmetric::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            weight_linear,
            linear_view(client, q_zeros, 4),
            linear_view(client, q_scales, 8),
            linear_view(client, output, 8),
            ScalarArg::new(out_width as usize),
            group_size,
        );
    };

    Ok(())
}
