use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

pub fn example() {
    type BackendF = Wgpu<f32>;
    type B = autodiff::Autodiff<BackendF>; // 학습하려면 Autodiff 래핑

    let device = WgpuDevice::default();

    let _x_train = Tensor::<B, 2>::from_floats([[1.0_f32], [2.0], [3.0]], &device);

    println!("{}", _x_train);
}
