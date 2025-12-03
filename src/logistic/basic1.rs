use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::tensor::activation::sigmoid;
use burn::tensor::{Int, Tensor};

const LEARNING_RATE: f32 = 0.1;
const EPOCHS: usize = 1000;

/// 간단한 로지스틱 회귀 실습 예제
pub fn example() {
    type BackendF = Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;

    let device = WgpuDevice::default();

    // x: [공부시간, 과외시간], y: 합격 여부(0/1)
    let x_train = Tensor::<AD, 2>::from_floats(
        [
            [1., 2.],
            [2., 3.],
            [3., 1.],
            [4., 3.],
            [5., 3.],
            [6., 2.],
        ],
        &device,
    );
    let y_train = Tensor::<AD, 2, Int>::from_ints([[0], [0], [0], [1], [1], [1]], &device);
    let sample_len = x_train.dims()[0];

    // 초기 가중치/편향
    let mut weights = [[0.0f32], [0.0]];
    let mut bias = 0.0f32;

    // Binary Cross Entropy (logits 입력)
    let bce = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device);

    for epoch in 0..=EPOCHS {
        let w = Tensor::<AD, 2>::from_floats(weights, &device).require_grad();
        let b = Tensor::<AD, 1>::from_floats([bias], &device).require_grad();

        // 선형 결합 후 시그모이드 (loss에서는 logits 그대로 사용)
        let logits = x_train.clone().matmul(w.clone()) + b.clone().expand([sample_len, 1]);
        let loss = bce.forward(logits, y_train.clone());

        let loss_value = loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("loss to_vec")[0];

        let grads = loss.backward();
        let grad_w = w
            .grad(&grads)
            .expect("grad for weights")
            .into_data()
            .to_vec::<f32>()
            .expect("grad w to_vec");
        let grad_b = b
            .grad(&grads)
            .expect("grad for bias")
            .into_data()
            .to_vec::<f32>()
            .expect("grad b to_vec")[0];

        for (row, grad) in weights.iter_mut().zip(grad_w.iter()) {
            row[0] -= LEARNING_RATE * grad;
        }
        bias -= LEARNING_RATE * grad_b;

        if epoch % 100 == 0 {
            println!(
                "Epoch {:4}/{:4} W: [{:>7.4}, {:>7.4}] b: {:>7.4} Loss: {:.6}",
                epoch, EPOCHS, weights[0][0], weights[1][0], bias, loss_value
            );
        }
    }

    println!("\n=== 학습된 파라미터 ===");
    println!("W: [{:.4}, {:.4}], b: {:.4}", weights[0][0], weights[1][0], bias);

    println!("\n=== 새 샘플 예측 ===");
    let test_inputs = [[5.0f32, 2.0], [2.0, 1.0]];
    let x_test = Tensor::<BackendF, 2>::from_floats(test_inputs, &device);
    let w_trained = Tensor::<BackendF, 2>::from_floats(weights, &device);
    let b_trained = Tensor::<BackendF, 1>::from_floats([bias], &device);

    let logits = x_test.matmul(w_trained) + b_trained.expand([test_inputs.len(), 1]);
    let probs = sigmoid(logits)
        .into_data()
        .to_vec::<f32>()
        .expect("probs to_vec");

    for (i, prob) in probs.iter().enumerate() {
        println!(
            "입력 {:?} -> 합격 확률 {:.4}",
            test_inputs[i],
            prob
        );
    }
}
