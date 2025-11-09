use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use rand::{rngs::StdRng, Rng, SeedableRng};

const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 1999;

pub fn example() {
    type BackendF = Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;

    let device = WgpuDevice::default();

    let x_train = Tensor::<AD, 1>::from_floats([1.0f32, 2.0, 3.0], &device);
    let y_train = Tensor::<AD, 1>::from_floats([2.0f32, 4.0, 6.0], &device);
    let sample_len = x_train.dims()[0];

    let mut w_value = 0.0f32;
    let mut b_value = 0.0f32;

    for epoch in 0..=EPOCHS {
        let w = Tensor::<AD, 1>::from_floats([w_value], &device).require_grad();
        let b = Tensor::<AD, 1>::from_floats([b_value], &device).require_grad();

        let hypothesis =
            x_train.clone() * w.clone().expand([sample_len]) + b.clone().expand([sample_len]);
        let loss = (hypothesis - y_train.clone()).powi_scalar(2).mean();
        let loss_value = loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("loss to_vec")[0];

        let grads = loss.backward();
        let grad_w = w
            .grad(&grads)
            .expect("Missing grad for w")
            .into_data()
            .to_vec::<f32>()
            .expect("grad w to_vec")[0];
        let grad_b = b
            .grad(&grads)
            .expect("Missing grad for b")
            .into_data()
            .to_vec::<f32>()
            .expect("grad b to_vec")[0];

        w_value -= LEARNING_RATE * grad_w;
        b_value -= LEARNING_RATE * grad_b;

        if epoch % 100 == 0 {
            println!(
                "Epoch {:4}/{:4} W: {:.3}, b: {:.3} Cost: {:.6}",
                epoch, EPOCHS, w_value, b_value, loss_value
            );
        }
    }

    let prediction = 4.0 * w_value + b_value;
    println!("학습 완료 → f(4) ≈ {prediction:.3}");

    // PyTorch: torch.manual_seed(3); torch.rand(1)
    println!("\n랜덤 시드가 3일 때");
    let mut rng = StdRng::seed_from_u64(3);
    for i in 1..=2 {
        let sample: f32 = rng.random();
        println!("rand sample #{i}: [{sample:.6}]");
    }
}
