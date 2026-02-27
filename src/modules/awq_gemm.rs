use crate::kernel::awq_kernel_gemm::{
    awq_gemm_linear, ShapeConfig, DECODE_M_THRESHOLD, DECODE_TILE_M,
};
use burn::{
    module::{Param, ParamId},
    prelude::*,
};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_tensor::{DType, Int, TensorPrimitive};
use std::sync::OnceLock;

const AWQ_PACK_FACTOR: usize = 8;
const TILE_N_BASE: usize = 32;
const TILE_N_MEDIUM: usize = 64;
const TILE_N_LARGE: usize = 128;
const INNER_K: usize = 64;
const TILE_M_SMALL: usize = 16;
const TILE_M_LARGE: usize = 32;

fn forced_tile_m_from_env() -> Option<usize> {
    static FORCED_TILE_M: OnceLock<Option<usize>> = OnceLock::new();
    *FORCED_TILE_M.get_or_init(|| {
        std::env::var("AWQ_GEMM_FORCE_TILE_M")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|tile| matches!(tile, 8 | 16 | 32))
    })
}

fn select_tile_m(m: usize) -> usize {
    if let Some(forced_tile) = forced_tile_m_from_env() {
        return forced_tile;
    }

    if m <= DECODE_M_THRESHOLD {
        DECODE_TILE_M
    } else if m <= TILE_M_SMALL {
        TILE_M_SMALL
    } else {
        TILE_M_LARGE
    }
}

fn select_tile_n(n: usize) -> usize {
    if n >= TILE_N_LARGE && n % TILE_N_LARGE == 0 {
        TILE_N_LARGE
    } else if n >= TILE_N_MEDIUM && n % TILE_N_MEDIUM == 0 {
        TILE_N_MEDIUM
    } else {
        TILE_N_BASE
    }
}

#[derive(Clone, Debug)]
pub struct AWQGEMMLinearConfig {
    d_input: usize,
    d_output: usize,
    group_size: usize,
    bias: bool,
}

impl AWQGEMMLinearConfig {
    pub fn new(d_input: usize, d_output: usize, group_size: usize) -> Self {
        Self {
            d_input,
            d_output,
            group_size,
            bias: false,
        }
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn build<B: Backend>(&self, device: &Device<B>) -> AWQGEMMLinear<B> {
        assert!(self.group_size > 0, "group_size must be > 0");
        assert!(
            self.d_output.is_multiple_of(AWQ_PACK_FACTOR),
            "d_output must be divisible by {AWQ_PACK_FACTOR}"
        );
        assert!(
            self.d_output.is_multiple_of(TILE_N_BASE),
            "d_output must be divisible by TILE_N ({TILE_N_BASE})"
        );
        assert!(
            self.d_input.is_multiple_of(INNER_K),
            "d_input must be divisible by INNER_K ({INNER_K})"
        );
        assert!(
            self.d_input % self.group_size == 0,
            "d_input must be divisible by group_size"
        );

        let d_out_packed = self.d_output / AWQ_PACK_FACTOR;
        let num_groups = self.d_input / self.group_size;

        let qweight = Param::initialized(
            ParamId::new(),
            Tensor::<B, 2, Int>::empty([self.d_input, d_out_packed], device),
        );
        let qzeros = Param::initialized(
            ParamId::new(),
            Tensor::<B, 2, Int>::empty([num_groups, d_out_packed], device),
        );
        let scales = Param::initialized(
            ParamId::new(),
            Tensor::<B, 2>::empty([num_groups, self.d_output], device),
        );
        let bias = self.bias.then(|| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 1>::zeros([self.d_output], device),
            )
        });

        AWQGEMMLinear {
            qweight,
            qzeros,
            scales,
            bias,
        }
    }
}

#[derive(Debug, Module)]
pub struct AWQGEMMLinear<B: Backend> {
    qweight: Param<Tensor<B, 2, Int>>,
    qzeros: Param<Tensor<B, 2, Int>>,
    scales: Param<Tensor<B, 2, Float>>,
    bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> AWQGEMMLinear<B> {
    pub fn group_size(&self) -> usize {
        let k = *self.qweight.val().shape().first().unwrap();
        let groups = *self.scales.val().shape().first().unwrap();
        k / groups
    }
}

impl<R: CubeRuntime, I: IntElement, F: FloatElement, BT: BoolElement>
    AWQGEMMLinear<CubeBackend<R, F, I, BT>>
{
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<CubeBackend<R, F, I, BT>, D>,
    ) -> Tensor<CubeBackend<R, F, I, BT>, D> {
        assert!(D >= 2, "input tensor must have at least 2 dims: [..., K]");

        let mut output_dims = input.dims();
        let k = output_dims[D - 1];
        assert!(k > 0, "input last dim (K) must be > 0");

        let input_elements = input.shape().num_elements();
        assert!(
            input_elements.is_multiple_of(k),
            "input elements must be divisible by K"
        );

        let m = input_elements / k;
        let input = input.reshape([m, k]);
        let qweight = self.qweight.val();
        let qzeros = self.qzeros.val();
        let scales = self.scales.val();

        let k_from_weight = *qweight.shape().first().unwrap();
        let n_packed = *qweight.shape().last().unwrap();
        let n = n_packed * 8;
        let group_size = self.group_size();

        assert_eq!(
            k, k_from_weight,
            "input last dim (K) must match qweight rows"
        );
        assert!(
            matches!(qweight.dtype(), DType::I32 | DType::U32),
            "qweight must have i32/u32 dtype"
        );
        assert!(
            matches!(qzeros.dtype(), DType::I32 | DType::U32),
            "qzeros must have i32/u32 dtype"
        );

        let output = Tensor::<CubeBackend<R, F, I, BT>, 2>::empty([m, n], &input.device());

        let input_primitive = match input.into_primitive() {
            TensorPrimitive::Float(float) => float,
            TensorPrimitive::QFloat(_) => panic!("unsupported qfloat"),
        };

        let qweight_primitive = qweight.into_primitive();
        let qzeros_primitive = qzeros.into_primitive();
        let scales_primitive = match scales.into_primitive() {
            TensorPrimitive::Float(float) => float,
            TensorPrimitive::QFloat(_) => panic!("unsupported qfloat"),
        };

        let output_primitive = match output.into_primitive() {
            TensorPrimitive::Float(float) => float,
            TensorPrimitive::QFloat(_) => panic!("unsupported qfloat"),
        };

        let tile_m = select_tile_m(m);
        let tile_n = select_tile_n(n);
        let shape = ShapeConfig::new(m, n, k, tile_m, tile_n, INNER_K);

        awq_gemm_linear::<R, F>(
            &input_primitive.client,
            &input_primitive.as_handle_ref(),
            &qweight_primitive.as_handle_ref(),
            &qzeros_primitive.as_handle_ref(),
            &scales_primitive.as_handle_ref(),
            &output_primitive.as_handle_ref(),
            &shape,
            group_size,
        )
        .expect("AWQ GEMM launch failed");

        let output = Tensor::<CubeBackend<R, F, I, BT>, 2>::from_primitive(TensorPrimitive::Float(
            output_primitive,
        ));

        let output = match &self.bias {
            Some(bias) => output + bias.val().reshape([1, n]),
            None => output,
        };

        output_dims[D - 1] = n;
        output.reshape(output_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::{AWQGEMMLinear, AWQGEMMLinearConfig};
    use burn::{
        module::{Param, ParamId},
        Tensor,
    };
    use burn_cubecl::CubeBackend;
    use burn_tensor::{Int, TensorData};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    const AWQ_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    const AWQ_REVERSE_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

    #[test]
    fn select_tile_m_uses_small_tiles_for_decode_like_shapes() {
        assert_eq!(super::select_tile_m(1), 8);
        assert_eq!(super::select_tile_m(8), 8);
        assert_eq!(super::select_tile_m(16), 16);
        assert_eq!(super::select_tile_m(17), 32);
    }

    #[test]
    fn select_tile_n_uses_large_tile_when_possible() {
        assert_eq!(super::select_tile_n(32), 32);
        assert_eq!(super::select_tile_n(48), 32);
        assert_eq!(super::select_tile_n(64), 64);
        assert_eq!(super::select_tile_n(128), 128);
        assert_eq!(super::select_tile_n(192), 64);
        assert_eq!(super::select_tile_n(1536), 128);
        assert_eq!(super::select_tile_n(5632), 128);
    }

    fn pack_i4(values: [i32; 8]) -> i32 {
        let mut packed = 0_i32;
        for (packed_lane, logical_lane) in AWQ_ORDER.into_iter().enumerate() {
            let value = values[logical_lane];
            packed |= (value & 0x0f) << (packed_lane * 4);
        }
        packed
    }

    fn unpack_i4(packed: i32, logical_lane: usize) -> f32 {
        let packed_lane = AWQ_REVERSE_ORDER[logical_lane];
        (((packed as u32) >> (packed_lane * 4)) & 0x0f) as f32
    }

    fn build_layer(
        device: &WgpuDevice,
        k: usize,
        n: usize,
        group_size: usize,
        bias_values: Option<Vec<f32>>,
    ) -> AWQGEMMLinear<TestBackend> {
        let n_packed = n / 8;
        let num_groups = k / group_size;

        let mut qweight = vec![0_i32; k * n_packed];
        let mut qzeros = vec![0_i32; num_groups * n_packed];
        let mut scales = vec![0.0_f32; num_groups * n];

        for row in 0..k {
            for col in 0..n_packed {
                let mut vals = [0_i32; 8];
                for (lane, val) in vals.iter_mut().enumerate() {
                    *val = ((row + col + lane) % 16) as i32;
                }
                qweight[row * n_packed + col] = pack_i4(vals);
            }
        }

        for group in 0..num_groups {
            for col in 0..n_packed {
                let mut vals = [0_i32; 8];
                for (lane, val) in vals.iter_mut().enumerate() {
                    *val = ((group + col + lane) % 8) as i32;
                }
                qzeros[group * n_packed + col] = pack_i4(vals);
            }

            for col in 0..n {
                scales[group * n + col] = 0.05 + group as f32 * 0.01 + col as f32 * 0.002;
            }
        }

        let qweight = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(qweight, [k, n_packed]),
            device,
        );
        let qzeros = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(qzeros, [num_groups, n_packed]),
            device,
        );
        let scales =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(scales, [num_groups, n]), device);
        let bias = bias_values.map(|values| {
            Tensor::<TestBackend, 1>::from_data(TensorData::new(values, [n]), device)
        });

        AWQGEMMLinear {
            qweight: Param::initialized(ParamId::new(), qweight),
            qzeros: Param::initialized(ParamId::new(), qzeros),
            scales: Param::initialized(ParamId::new(), scales),
            bias: bias.map(|value| Param::initialized(ParamId::new(), value)),
        }
    }

    fn cpu_reference(
        input: &[f32],
        qweight: &[i32],
        qzeros: &[i32],
        scales: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let n_packed = n / 8;
        let mut out = vec![0.0_f32; batch * m * n];

        for b in 0..batch {
            for row in 0..m {
                for col in 0..n {
                    let mut acc = 0.0_f32;
                    for kk in 0..k {
                        let g = kk / group_size;
                        let packed_w = qweight[kk * n_packed + col / 8];
                        let packed_z = qzeros[g * n_packed + col / 8];
                        let scale = scales[g * n + col];
                        let w = unpack_i4(packed_w, col % 8);
                        let z = unpack_i4(packed_z, col % 8);
                        acc += input[b * m * k + row * k + kk] * ((w - z) * scale);
                    }

                    if let Some(bias_values) = bias {
                        acc += bias_values[col];
                    }

                    out[b * m * n + row * n + col] = acc;
                }
            }
        }

        out
    }

    #[test]
    fn test_config_build_shapes_without_bias() {
        let device = WgpuDevice::default();
        let layer = AWQGEMMLinearConfig::new(64, 32, 32).build::<TestBackend>(&device);

        assert_eq!(layer.qweight.val().dims(), [64, 4]);
        assert_eq!(layer.qzeros.val().dims(), [2, 4]);
        assert_eq!(layer.scales.val().dims(), [2, 32]);
        assert_eq!(layer.group_size(), 32);
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_config_build_shapes_with_bias() {
        let device = WgpuDevice::default();
        let layer = AWQGEMMLinearConfig::new(64, 32, 32)
            .with_bias(true)
            .build::<TestBackend>(&device);

        assert!(layer.bias.is_some());
        assert_eq!(layer.bias.as_ref().unwrap().val().dims(), [32]);
    }

    #[test]
    fn test_forward_matches_cpu_reference() {
        let device = WgpuDevice::default();
        let (batch, m, k, n, group_size) = (2, 32, 64, 32, 32);
        let layer = build_layer(&device, k, n, group_size, None);

        let input_data: Vec<f32> = (0..batch * m * k)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.1)
            .collect();
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [batch, m, k]),
            &device,
        );

        let output = layer.forward(input);
        assert_eq!(output.dims(), [batch, m, n]);

        let qweight = layer.qweight.val().into_data().to_vec::<i32>().unwrap();
        let qzeros = layer.qzeros.val().into_data().to_vec::<i32>().unwrap();
        let scales = layer.scales.val().into_data().to_vec::<f32>().unwrap();

        let expected = cpu_reference(
            &input_data,
            &qweight,
            &qzeros,
            &scales,
            None,
            batch,
            m,
            k,
            n,
            group_size,
        );

        let actual = output.into_data().to_vec::<f32>().unwrap();
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= 1e-2,
                "output mismatch at idx {idx}: expected {e}, got {a}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_forward_with_bias_matches_cpu_reference() {
        let device = WgpuDevice::default();
        let (batch, m, k, n, group_size) = (2, 32, 64, 32, 32);
        let bias_values: Vec<f32> = (0..n).map(|idx| -0.2 + idx as f32 * 0.01).collect();
        let layer = build_layer(&device, k, n, group_size, Some(bias_values));

        let input_data: Vec<f32> = (0..batch * m * k)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.07)
            .collect();
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [batch, m, k]),
            &device,
        );

        let output = layer.forward(input);
        assert_eq!(output.dims(), [batch, m, n]);

        let qweight = layer.qweight.val().into_data().to_vec::<i32>().unwrap();
        let qzeros = layer.qzeros.val().into_data().to_vec::<i32>().unwrap();
        let scales = layer.scales.val().into_data().to_vec::<f32>().unwrap();
        let bias = layer
            .bias
            .as_ref()
            .expect("bias should be present")
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let expected = cpu_reference(
            &input_data,
            &qweight,
            &qzeros,
            &scales,
            Some(&bias),
            batch,
            m,
            k,
            n,
            group_size,
        );

        let actual = output.into_data().to_vec::<f32>().unwrap();
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= 1e-2,
                "output (with bias) mismatch at idx {idx}: expected {e}, got {a}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_forward_handles_non_divisible_m() {
        let device = WgpuDevice::default();
        let (batch, m, k, n, group_size) = (1, 14, 64, 32, 32);
        let layer = build_layer(&device, k, n, group_size, None);
        let input = Tensor::<TestBackend, 3>::zeros([batch, m, k], &device);

        let output = layer.forward(input);

        assert_eq!(output.dims(), [batch, m, n]);
    }

    #[test]
    fn test_forward_decode_shape_matches_cpu_reference() {
        let device = WgpuDevice::default();
        let (batch, m, k, n, group_size) = (2, 1, 64, 32, 64);
        let layer = build_layer(&device, k, n, group_size, None);

        let input_data: Vec<f32> = (0..batch * m * k)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.09)
            .collect();
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [batch, m, k]),
            &device,
        );

        let output = layer.forward(input);
        assert_eq!(output.dims(), [batch, m, n]);

        let qweight = layer.qweight.val().into_data().to_vec::<i32>().unwrap();
        let qzeros = layer.qzeros.val().into_data().to_vec::<i32>().unwrap();
        let scales = layer.scales.val().into_data().to_vec::<f32>().unwrap();

        let expected = cpu_reference(
            &input_data,
            &qweight,
            &qzeros,
            &scales,
            None,
            batch,
            m,
            k,
            n,
            group_size,
        );

        let actual = output.into_data().to_vec::<f32>().unwrap();
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= 1e-2,
                "decode output mismatch at idx {idx}: expected {e}, got {a}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_forward_4d_input_matches_cpu_reference() {
        let device = WgpuDevice::default();
        let (d0, d1, d2, k, n, group_size) = (1, 16, 14, 64, 32, 32);
        let rows = d0 * d1 * d2;
        let layer = build_layer(&device, k, n, group_size, None);

        let input_data: Vec<f32> = (0..rows * k)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03)
            .collect();
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(input_data.clone(), [d0, d1, d2, k]),
            &device,
        );

        let output = layer.forward(input);
        assert_eq!(output.dims(), [d0, d1, d2, n]);

        let qweight = layer.qweight.val().into_data().to_vec::<i32>().unwrap();
        let qzeros = layer.qzeros.val().into_data().to_vec::<i32>().unwrap();
        let scales = layer.scales.val().into_data().to_vec::<f32>().unwrap();

        let expected = cpu_reference(
            &input_data,
            &qweight,
            &qzeros,
            &scales,
            None,
            1,
            rows,
            k,
            n,
            group_size,
        );

        let actual = output.into_data().to_vec::<f32>().unwrap();
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= 1e-2,
                "4D output mismatch at idx {idx}: expected {e}, got {a}, diff={diff}"
            );
        }
    }
}
