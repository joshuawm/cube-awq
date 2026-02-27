pub mod awq_kernel;
pub mod awq_kernel_gemm;
use awq_kernel::dequantize_native;
use burn_cubecl::{
    tensor::CubeTensor, BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
};
use burn_tensor::{DType, Int, Tensor, TensorMetadata, TensorPrimitive};

pub fn dequantize_awq<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    weight: Tensor<CubeBackend<R, F, I, BT>, 2, Int>,
    zeros: Tensor<CubeBackend<R, F, I, BT>, 2, Int>,
    scales: Tensor<CubeBackend<R, F, I, BT>, 2>,
    group_size: usize,
) -> CubeTensor<R> {
    let device = weight.device();
    let w_primitive = weight.into_primitive();
    let z_primitive = zeros.into_primitive();
    let s_primitive = match scales.into_primitive() {
        TensorPrimitive::Float(f) => f,
        _ => panic!("dddddd"),
    };

    assert!(matches!(w_primitive.dtype(), DType::I32 | DType::U32));
    assert!(matches!(z_primitive.dtype(), DType::I32 | DType::U32));

    let client = w_primitive.client.clone();

    let h = *w_primitive.shape().first().unwrap();
    let w = *s_primitive.shape().last().unwrap();

    let out_shape = vec![h, w];

    let out: Tensor<CubeBackend<R, F, I, BT>, 2> = Tensor::empty(out_shape, &device);

    let o_primitive = match out.into_primitive() {
        TensorPrimitive::Float(f) => f,
        _ => panic!("dddddd"),
    };

    let _ = dequantize_native::<R, F>(
        &client,
        &w_primitive.as_handle_ref(),
        &z_primitive.as_handle_ref(),
        &s_primitive.as_handle_ref(),
        &o_primitive.as_handle_ref(),
        group_size,
    )
    .unwrap();

    o_primitive
}
