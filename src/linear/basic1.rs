use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, LearningRate, Optimizer, SgdConfig};
use burn::tensor::Tensor;

const LEARNING_RATE: LearningRate = 0.01;
const EPOCHS: usize = 500;

pub fn example() {
    type BackendF = Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;

    let device = WgpuDevice::default();
    let x_train = Tensor::<AD, 2>::from_floats([[1.0f32], [2.0], [3.0]], &device);
    let y_train = Tensor::<AD, 2>::from_floats([[2.0f32], [4.0], [6.0]], &device);

    // 단일 선형 계층 (입력 1, 출력 1)
    let mut model: Linear<AD> = LinearConfig::new(1, 1).init(&device);
    let mut optimizer = SgdConfig::new().init();

    for epoch in 0..EPOCHS {
        let preds = model.forward(x_train.clone());
        let loss = (preds - y_train.clone()).powi_scalar(2).mean();
        let loss_value = loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("loss to_vec")[0];

        let grads = loss.backward();
        let grad_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(LEARNING_RATE, model, grad_params);

        if epoch % 100 == 0 || epoch == EPOCHS - 1 {
            println!("epoch {epoch:03}: loss = {loss_value:.6}");
        }
    }

    // 추론 예시
    let x_test = Tensor::<AD, 2>::from_floats([[4.0f32]], &device);
    let prediction = model
        .forward(x_test)
        .into_data()
        .to_vec::<f32>()
        .expect("prediction to_vec")[0];

    println!("SGD lr={LEARNING_RATE}: f(4) ≈ {prediction:.3}");
}
