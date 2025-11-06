use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

pub fn example() {
    type BackendF = Wgpu<f32>;
    type B = autodiff::Autodiff<BackendF>; // 학습하려면 Autodiff 래핑

    let device = WgpuDevice::default();

    let x_train = Tensor::<B, 2>::from_floats([[1.0_f32], [2.0], [3.0]], &device);
    let y_train = Tensor::<B, 2>::from_floats([[2.0_f32], [4.0], [6.0]], &device);

    let w = Tensor::<B, 1>::zeros([1], &device).require_grad();
    let b = Tensor::<B, 1>::zeros([1], &device).require_grad(); // shape: [1]
    println!("{}", x_train);
    println!("{:?}", x_train.shape());
    println!("{}", y_train);

    println!("{}",w);
}
