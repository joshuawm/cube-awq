use crate::kernel::dequantize_awq;
use burn::prelude::*;
use burn_cubecl::{
    kernel::matmul::{matmul, MatmulStrategy},
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
};
use burn_tensor::{DType, TensorPrimitive};

#[derive(Clone, Debug)]
pub struct AWQLinearConfig {
    d_input: usize,
    d_output: usize,
    group_size: usize,
}

impl AWQLinearConfig {
    pub fn new(d_input: usize, d_output: usize, group_size: usize) -> Self {
        Self {
            d_input,
            d_output,
            group_size,
        }
    }

    pub fn build<B: Backend>(&self, device: &Device<B>) -> AWQLinear<B> {
        assert!(self.d_output % 8 == 0);
        assert!(self.d_input % self.group_size == 0);
        assert!(self.d_output % 32 == 0);
        let d_out = self.d_output / 8;
        let g_input = self.d_input / self.group_size;

        let qweight = Tensor::<B, 2, Int>::empty([self.d_input, d_out], device);
        let qzeros = Tensor::<B, 2, Int>::empty([g_input, d_out], device);
        let scales = Tensor::<B, 2>::empty([g_input, self.d_output], device);

        AWQLinear {
            qweight,
            qzeros,
            scales,
        }
    }
}

#[derive(Debug, Module)]
pub struct AWQLinear<B: Backend> {
    qweight: Tensor<B, 2, Int>,
    qzeros: Tensor<B, 2, Int>,
    scales: Tensor<B, 2, Float>,
}

impl<B: Backend> AWQLinear<B> {
    pub fn group_size(&self) -> usize {
        let h = *self.qweight.shape().first().unwrap();
        let scale_h = *self.scales.shape().first().unwrap();
        h / scale_h
    }
}

impl<R: CubeRuntime, I: IntElement, F: FloatElement, BT: BoolElement>
    AWQLinear<CubeBackend<R, F, I, BT>>
{
    pub fn dequantize(&self) -> Tensor<CubeBackend<R, F, I, BT>, 2> {
        let weight_primitive = dequantize_awq(
            self.qweight.clone(),
            self.qzeros.clone(),
            self.scales.clone(),
            self.group_size(),
        );
        Tensor::<CubeBackend<R, F, I, BT>, 2>::from_primitive(TensorPrimitive::Float(
            weight_primitive,
        ))
    }

    pub fn forward<const D: usize>(
        &self,
        input: Tensor<CubeBackend<R, F, I, BT>, D>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, D> {
        let dtype = input.dtype();

        let weight_primitive = dequantize_awq(
            self.qweight.clone(),
            self.qzeros.clone(),
            self.scales.clone(),
            self.group_size(),
        );
        let weight_last_dim = *self.qweight.shape().last().unwrap();
        let mut input_shape = input.shape();
        if let Some(last) = input_shape.last_mut() {
            *last = weight_last_dim;
        }

        let _ = Tensor::<CubeBackend<R, F, I, BT>, D>::empty(input_shape, &input.device());
        let i_primitive = input.into_primitive();
        let input_primitive = match i_primitive {
            TensorPrimitive::Float(f) => f,
            TensorPrimitive::QFloat(_) => panic!("unsupported qfloat"),
        };

        let out_primitive = matmul(
            input_primitive,
            weight_primitive,
            None,
            MatmulStrategy::Autotune,
            dtype,
        )
        .unwrap();
        Tensor::<CubeBackend<R, F, I, BT>, D>::from_primitive(TensorPrimitive::Float(out_primitive))
    }
}

#[cfg(test)]
mod tests {
    use super::AWQLinear;
    use burn::Tensor;
    use burn_cubecl::CubeBackend;
    use burn_tensor::{Int, TensorData};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn pack_i4(values: [i32; 8]) -> i32 {
        let mut packed = 0i32;
        for (i, &v) in values.iter().enumerate() {
            packed |= (v & 0x0f) << (i * 4);
        }
        packed
    }

    fn unpack_i4(packed: i32, idx: usize) -> i32 {
        (packed >> (idx * 4)) & 0x0f
    }

    fn build_test_layer(
        device: &WgpuDevice,
    ) -> (AWQLinear<TestBackend>, Vec<f32>, usize, usize, usize) {
        let height = 64usize;
        let out_width = 32usize;
        let packed_width = out_width / 8;
        let group_size = height / out_width;
        let num_groups = height / group_size;

        let mut weights_packed = vec![0i32; height * packed_width];
        let mut zeros_packed = vec![0i32; num_groups * packed_width];
        let mut scales = vec![0.0f32; num_groups * out_width];

        for r in 0..height {
            for c_block in 0..packed_width {
                let mut vals = [0i32; 8];
                for i in 0..8 {
                    vals[i] = ((r + c_block + i) % 16) as i32;
                }
                weights_packed[r * packed_width + c_block] = pack_i4(vals);
            }
        }

        for g in 0..num_groups {
            for c_block in 0..packed_width {
                let mut vals = [0i32; 8];
                for i in 0..8 {
                    vals[i] = ((g + c_block + i) % 8) as i32;
                }
                zeros_packed[g * packed_width + c_block] = pack_i4(vals);
            }
            for c in 0..out_width {
                scales[g * out_width + c] = 0.5 + (g as f32) * 0.01 + (c as f32) * 0.001;
            }
        }

        let mut expected = vec![0.0f32; height * out_width];
        for r in 0..height {
            let g = r / group_size;
            for c in 0..out_width {
                let w_packed = weights_packed[r * packed_width + (c / 8)];
                let z_packed = zeros_packed[g * packed_width + (c / 8)];
                let w = unpack_i4(w_packed, c % 8) as f32;
                let z = unpack_i4(z_packed, c % 8) as f32;
                let s = scales[g * out_width + c];
                expected[r * out_width + c] = (w - z) * s;
            }
        }

        let q_weight = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(weights_packed, [height, packed_width]),
            device,
        );
        let q_zeros = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(zeros_packed, [num_groups, packed_width]),
            device,
        );
        let q_scales = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(scales, [num_groups, out_width]),
            device,
        );

        (
            AWQLinear::<TestBackend> {
                qweight: q_weight,
                qzeros: q_zeros,
                scales: q_scales,
            },
            expected,
            height,
            out_width,
            group_size,
        )
    }

    #[test]
    fn test_group_size() {
        let device = WgpuDevice::default();
        let (layer, _expected, _height, _out_width, group_size) = build_test_layer(&device);
        assert_eq!(layer.group_size(), group_size);
    }

    #[test]
    fn test_dequantize_matches_cpu() {
        let device = WgpuDevice::default();
        let (layer, expected, height, out_width, _group_size) = build_test_layer(&device);

        let dequantized = layer.dequantize();
        let [h, w] = dequantized.dims();
        assert_eq!(h, height);
        assert_eq!(w, out_width);

        let actual = dequantized.into_data().to_vec::<f32>().unwrap();
        let mut max_diff = 0.0f32;
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff <= 1e-4,
                "dequant mismatch at idx {idx}: expected {e}, got {a}, diff {diff}"
            );
        }
    }

    #[test]
    fn test_forward_matches_cpu_matmul() {
        let device = WgpuDevice::default();
        let (layer, expected_weight, height, out_width, _group_size) = build_test_layer(&device);

        let batch = 2usize;
        let input_data: Vec<f32> = (0..batch * height).map(|i| (i as f32) * 0.01).collect();
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(input_data.clone(), [batch, height]),
            &device,
        );

        let output = layer.forward(input);
        let [b, w] = output.dims();
        assert_eq!(b, batch);
        assert_eq!(w, out_width);

        let mut expected_out = vec![0.0f32; batch * out_width];
        for b in 0..batch {
            for c in 0..out_width {
                let mut acc = 0.0f32;
                for k in 0..height {
                    let x = input_data[b * height + k];
                    let w = expected_weight[k * out_width + c];
                    acc += x * w;
                }
                expected_out[b * out_width + c] = acc;
            }
        }

        let actual = output.into_data().to_vec::<f32>().unwrap();
        let mut max_diff = 0.0f32;
        for (idx, (a, e)) in actual.iter().zip(expected_out.iter()).enumerate() {
            let diff = (a - e).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff <= 1e-3,
                "forward mismatch at idx {idx}: expected {e}, got {a}, diff {diff}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "RankMismatch")]
    fn test_forward_rank3_panics_on_rank_mismatch() {
        let device = WgpuDevice::default();
        let (layer, _expected_weight, height, _out_width, _group_size) = build_test_layer(&device);

        let batch = 2usize;
        let seq = 3usize;
        let input_data: Vec<f32> = (0..batch * seq * height)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [batch, seq, height]),
            &device,
        );

        let _ = layer.forward(input);
    }

    #[test]
    #[should_panic(expected = "out_width % 32 == 0")]
    fn test_dequantize_panics_on_invalid_out_width() {
        let device = WgpuDevice::default();
        let height = 64usize;
        let out_width = 16usize;
        let packed_width = out_width / 8;
        let group_size = height / out_width;
        let num_groups = height / group_size;

        let weights_packed = vec![0i32; height * packed_width];
        let zeros_packed = vec![0i32; num_groups * packed_width];
        let scales = vec![1.0f32; num_groups * out_width];

        let q_weight = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(weights_packed, [height, packed_width]),
            &device,
        );
        let q_zeros = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(zeros_packed, [num_groups, packed_width]),
            &device,
        );
        let q_scales = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(scales, [num_groups, out_width]),
            &device,
        );

        let layer = AWQLinear::<TestBackend> {
            qweight: q_weight,
            qzeros: q_zeros,
            scales: q_scales,
        };

        let _ = layer.dequantize();
    }
}
