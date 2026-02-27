use cube_awq::kernel::awq_kernel::dequantize_native;
use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use std::marker::PhantomData;

// ==========================================
// 1. 自定义 TensorHandle (修复版)
// ==========================================
struct TensorHandle<R: Runtime> {
    handle: cubecl::server::Handle,
    shape: Vec<usize>,
    strides: Vec<usize>,
    elem_size: usize,
    _marker: PhantomData<R>,
}

impl<R: Runtime> TensorHandle<R> {
    // 创建并上传数据
    fn new<T: bytemuck::NoUninit>(
        client: &ComputeClient<R>,
        shape: Vec<usize>,
        data: Vec<T>,
    ) -> Self {
        let elem_size = std::mem::size_of::<T>();
        let slice = bytemuck::cast_slice(&data);

        // ✅ 修复 1: 使用 Bytes::new(vec) 替代 copy_from_slice
        // 因为 CubeCL 的 Bytes 只是 Vec<u8> 的简单封装，直接构造即可
        let bytes = Bytes::from_bytes_vec(slice.to_vec());
        let handle = client.create(bytes);

        // 计算 Strides
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        Self {
            handle,
            shape,
            strides,
            elem_size,
            _marker: PhantomData,
        }
    }

    // 创建空 Tensor
    fn new_empty(client: &ComputeClient<R>, shape: Vec<usize>) -> Self {
        let elem_size = 4; // f32
        let num_elements: usize = shape.iter().product();
        let size_bytes = num_elements * elem_size;

        let handle = client.empty(size_bytes);

        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        Self {
            handle,
            shape,
            strides,
            elem_size,
            _marker: PhantomData,
        }
    }

    // 生成 TensorHandleRef
    fn as_ref(&self) -> TensorHandleRef<R> {
        TensorHandleRef {
            handle: &self.handle,
            strides: &self.strides,
            shape: &self.shape,
            elem_size: self.elem_size,
            runtime: PhantomData,
        }
    }

    // 下载数据
    // 注意：这里的 read 是同步的，直接返回数据
    async fn read_data(&self, client: &ComputeClient<R>) -> Vec<u8> {
        let output_handle = self.handle.clone();

        // ✅ 修复 2: client.read 需要 Vec<Handle>，且是同步方法 (无 await)
        let bytes_vec = client.read(vec![output_handle]);

        // 取出第一个结果 (因为我们只请求了一个 Handle)
        let bytes = &bytes_vec[0];

        // Bytes 通常实现了 Deref<[u8]>，直接转 Vec
        bytes.to_vec()
    }
}

// ==========================================
// 2. CPU 辅助函数
// ==========================================
fn pack_u32_cpu(vals: &[u8]) -> u32 {
    assert_eq!(vals.len(), 8);
    let mut packed = 0u32;
    for (i, &v) in vals.iter().enumerate() {
        let v = v & 0x0F;
        packed |= (v as u32) << (i * 4);
    }
    packed
}

// ==========================================
// 3. 测试主函数
// ==========================================
#[tokio::test]
async fn test_awq_kernel() {
    type Runtime = WgpuRuntime;

    let device = WgpuDevice::default();
    let client = ComputeClient::<Runtime>::load(&device);

    // 参数设置
    let height = 128;
    let out_width = 32;
    let group_size = 128;
    let packed_width = out_width / 8;

    // 构造数据
    let mut weights_packed_cpu = vec![0u32; height * packed_width];
    let mut zeros_packed_cpu = vec![0u32; (height / group_size) * packed_width];
    let mut scales_cpu = vec![0.0f32; (height / group_size) * out_width];
    let mut expected_output_cpu = vec![0.0f32; height * out_width];

    println!("生成测试数据...");

    for r in 0..height {
        for c_block in 0..packed_width {
            let w_vals: Vec<u8> = (0..8).map(|i| ((r + c_block + i) % 15) as u8).collect();
            let z_vals: Vec<u8> = (0..8).map(|i| ((i + 1) % 8) as u8).collect();

            weights_packed_cpu[r * packed_width + c_block] = pack_u32_cpu(&w_vals);

            if r == 0 {
                zeros_packed_cpu[c_block] = pack_u32_cpu(&z_vals);
            }

            for i in 0..8 {
                let abs_col = c_block * 8 + i;
                let scale_val = 0.5 + (abs_col as f32) * 0.1;

                if r == 0 {
                    scales_cpu[abs_col] = scale_val;
                }

                let w = w_vals[i] as f32;
                let z = z_vals[i] as f32;
                let s = scale_val;

                expected_output_cpu[r * out_width + abs_col] = (w - z) * s;
            }
        }
    }

    // 上传 Tensor
    let q_weight_handle =
        TensorHandle::new(&client, vec![height, packed_width], weights_packed_cpu);
    let q_zeros_handle = TensorHandle::new(
        &client,
        vec![height / group_size, packed_width],
        zeros_packed_cpu,
    );
    let q_scales_handle =
        TensorHandle::new(&client, vec![height / group_size, out_width], scales_cpu);
    let output_handle = TensorHandle::<Runtime>::new_empty(&client, vec![height, out_width]);

    // 运行 Kernel
    println!("启动 Kernel...");

    dequantize_native::<Runtime, f32>(
        &client,
        &q_weight_handle.as_ref(),
        &q_zeros_handle.as_ref(),
        &q_scales_handle.as_ref(),
        &output_handle.as_ref(),
        group_size,
    )
    .unwrap();

    // 等待 GPU
    client.flush();

    // 下载结果
    let result_bytes = output_handle.read_data(&client).await;
    let result_f32: Vec<f32> = bytemuck::cast_slice(&result_bytes).to_vec();

    // 验证
    println!("验证结果...");
    let mut max_diff = 0.0f32;
    for i in 0..expected_output_cpu.len() {
        let cpu_val = expected_output_cpu[i];
        let gpu_val = result_f32[i];
        let diff = (cpu_val - gpu_val).abs();

        if diff > max_diff {
            max_diff = diff;
        }

        if diff > 1e-4 {
            let row = i / out_width;
            let col = i % out_width;
            panic!(
                "Mismatch at [{}, {}]: CPU={} vs GPU={}, Diff={}",
                row, col, cpu_val, gpu_val, diff
            );
        }
    }

    println!("✅ 测试通过！最大误差: {}", max_diff);
}

#[test]
#[should_panic(expected = "AWQ packed q_weight must use 32-bit elements")]
fn test_awq_kernel_rejects_i16_q_weight() {
    type Runtime = WgpuRuntime;

    let device = WgpuDevice::default();
    let client = ComputeClient::<Runtime>::load(&device);

    let height = 32usize;
    let out_width = 32usize;
    let group_size = 32usize;
    let packed_width = out_width / 8;

    let weights_packed_cpu = vec![0i16; height * packed_width];
    let zeros_packed_cpu = vec![0u32; (height / group_size) * packed_width];
    let scales_cpu = vec![1.0f32; (height / group_size) * out_width];

    let q_weight_handle =
        TensorHandle::new(&client, vec![height, packed_width], weights_packed_cpu);
    let q_zeros_handle = TensorHandle::new(
        &client,
        vec![height / group_size, packed_width],
        zeros_packed_cpu,
    );
    let q_scales_handle =
        TensorHandle::new(&client, vec![height / group_size, out_width], scales_cpu);
    let output_handle = TensorHandle::<Runtime>::new_empty(&client, vec![height, out_width]);

    let _ = dequantize_native::<Runtime, f32>(
        &client,
        &q_weight_handle.as_ref(),
        &q_zeros_handle.as_ref(),
        &q_scales_handle.as_ref(),
        &output_handle.as_ref(),
        group_size,
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "AWQ packed q_weight must use 32-bit elements")]
fn test_awq_kernel_rejects_i64_q_weight() {
    type Runtime = WgpuRuntime;

    let device = WgpuDevice::default();
    let client = ComputeClient::<Runtime>::load(&device);

    let height = 32usize;
    let out_width = 32usize;
    let group_size = 32usize;
    let packed_width = out_width / 8;

    let weights_packed_cpu = vec![0i64; height * packed_width];
    let zeros_packed_cpu = vec![0u32; (height / group_size) * packed_width];
    let scales_cpu = vec![1.0f32; (height / group_size) * out_width];

    let q_weight_handle =
        TensorHandle::new(&client, vec![height, packed_width], weights_packed_cpu);
    let q_zeros_handle = TensorHandle::new(
        &client,
        vec![height / group_size, packed_width],
        zeros_packed_cpu,
    );
    let q_scales_handle =
        TensorHandle::new(&client, vec![height / group_size, out_width], scales_cpu);
    let output_handle = TensorHandle::<Runtime>::new_empty(&client, vec![height, out_width]);

    let _ = dequantize_native::<Runtime, f32>(
        &client,
        &q_weight_handle.as_ref(),
        &q_zeros_handle.as_ref(),
        &q_scales_handle.as_ref(),
        &output_handle.as_ref(),
        group_size,
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "AWQ packed q_zeros must use 32-bit elements")]
fn test_awq_kernel_rejects_i16_q_zeros() {
    type Runtime = WgpuRuntime;

    let device = WgpuDevice::default();
    let client = ComputeClient::<Runtime>::load(&device);

    let height = 32usize;
    let out_width = 32usize;
    let group_size = 32usize;
    let packed_width = out_width / 8;

    let weights_packed_cpu = vec![0u32; height * packed_width];
    let zeros_packed_cpu = vec![0i16; (height / group_size) * packed_width];
    let scales_cpu = vec![1.0f32; (height / group_size) * out_width];

    let q_weight_handle =
        TensorHandle::new(&client, vec![height, packed_width], weights_packed_cpu);
    let q_zeros_handle = TensorHandle::new(
        &client,
        vec![height / group_size, packed_width],
        zeros_packed_cpu,
    );
    let q_scales_handle =
        TensorHandle::new(&client, vec![height / group_size, out_width], scales_cpu);
    let output_handle = TensorHandle::<Runtime>::new_empty(&client, vec![height, out_width]);

    let _ = dequantize_native::<Runtime, f32>(
        &client,
        &q_weight_handle.as_ref(),
        &q_zeros_handle.as_ref(),
        &q_scales_handle.as_ref(),
        &output_handle.as_ref(),
        group_size,
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "AWQ packed q_zeros must use 32-bit elements")]
fn test_awq_kernel_rejects_i64_q_zeros() {
    type Runtime = WgpuRuntime;

    let device = WgpuDevice::default();
    let client = ComputeClient::<Runtime>::load(&device);

    let height = 32usize;
    let out_width = 32usize;
    let group_size = 32usize;
    let packed_width = out_width / 8;

    let weights_packed_cpu = vec![0u32; height * packed_width];
    let zeros_packed_cpu = vec![0i64; (height / group_size) * packed_width];
    let scales_cpu = vec![1.0f32; (height / group_size) * out_width];

    let q_weight_handle =
        TensorHandle::new(&client, vec![height, packed_width], weights_packed_cpu);
    let q_zeros_handle = TensorHandle::new(
        &client,
        vec![height / group_size, packed_width],
        zeros_packed_cpu,
    );
    let q_scales_handle =
        TensorHandle::new(&client, vec![height / group_size, out_width], scales_cpu);
    let output_handle = TensorHandle::<Runtime>::new_empty(&client, vec![height, out_width]);

    let _ = dequantize_native::<Runtime, f32>(
        &client,
        &q_weight_handle.as_ref(),
        &q_zeros_handle.as_ref(),
        &q_scales_handle.as_ref(),
        &output_handle.as_ref(),
        group_size,
    )
    .unwrap();
}
