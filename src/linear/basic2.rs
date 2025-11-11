use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

const LR_SCALAR: f32 = 1e-5;
const LR_MATRIX: f32 = 1e-5;

pub fn example() {
    type BackendF = Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;

    let device = WgpuDevice::default();

    println!("=== 1. 다중 특성을 개별 변수로 다루기 ===");

    let x1_train = Tensor::<AD, 1>::from_floats([73., 93., 89., 96., 73.], &device);
    let x2_train = Tensor::<AD, 1>::from_floats([80., 88., 91., 98., 66.], &device);
    let x3_train = Tensor::<AD, 1>::from_floats([75., 93., 90., 100., 70.], &device);
    let y_train = Tensor::<AD, 1>::from_floats([152., 185., 180., 196., 142.], &device);
    let sample_len = x1_train.dims()[0];

    let mut w1_val = 0.0f32;
    let mut w2_val = 0.0f32;
    let mut w3_val = 0.0f32;
    let mut b_val = 0.0f32;

    for epoch in 0..=1000 {
        let w1 = Tensor::<AD, 1>::from_floats([w1_val], &device).require_grad();
        let w2 = Tensor::<AD, 1>::from_floats([w2_val], &device).require_grad();
        let w3 = Tensor::<AD, 1>::from_floats([w3_val], &device).require_grad();
        let b = Tensor::<AD, 1>::from_floats([b_val], &device).require_grad();

        let hypothesis = x1_train.clone() * w1.clone().expand([sample_len])
            + x2_train.clone() * w2.clone().expand([sample_len])
            + x3_train.clone() * w3.clone().expand([sample_len])
            + b.clone().expand([sample_len]);

        let loss = (hypothesis - y_train.clone()).powi_scalar(2).mean();
        let loss_value = loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("loss to_vec")[0];

        let grads = loss.backward();

        let grad_w1 = extract_scalar_grad(&w1, &grads);
        let grad_w2 = extract_scalar_grad(&w2, &grads);
        let grad_w3 = extract_scalar_grad(&w3, &grads);
        let grad_b = extract_scalar_grad(&b, &grads);

        w1_val -= LR_SCALAR * grad_w1;
        w2_val -= LR_SCALAR * grad_w2;
        w3_val -= LR_SCALAR * grad_w3;
        b_val -= LR_SCALAR * grad_b;

        if epoch % 100 == 0 {
            println!(
                "Epoch {:4}/{} w1: {:>7.4} w2: {:>7.4} w3: {:>7.4} b: {:>7.4} Cost: {:.6}",
                epoch, 1000, w1_val, w2_val, w3_val, b_val, loss_value
            );
        }
    }

    println!("\n=== 2. 행렬 연산으로 일반화하기 ===");

    let x_train = Tensor::<AD, 2>::from_floats(
        [
            [73., 80., 75.],
            [93., 88., 93.],
            [89., 91., 80.],
            [96., 98., 100.],
            [73., 66., 70.],
        ],
        &device,
    );
    let y_train_mat =
        Tensor::<AD, 2>::from_floats([[152.], [185.], [180.], [196.], [142.]], &device);

    println!("x_train shape = {:?}", x_train.shape());
    println!("y_train shape = {:?}", y_train_mat.shape());

    let mut w_matrix = [[0.0f32], [0.0], [0.0]];
    let mut b_scalar = 0.0f32;

    for epoch in 0..=20 {
        let w = Tensor::<AD, 2>::from_floats(w_matrix, &device).require_grad();
        let b = Tensor::<AD, 1>::from_floats([b_scalar], &device).require_grad();

        let hypothesis = x_train.clone().matmul(w.clone())
            + b.clone().expand([x_train.dims()[0], 1]);
        let loss = (hypothesis.clone() - y_train_mat.clone()).powi_scalar(2).mean();

        let loss_value = loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("loss to_vec")[0];

        let grads = loss.backward();
        let grad_w = w
            .grad(&grads)
            .expect("Grad for W")
            .into_data()
            .to_vec::<f32>()
            .expect("grad w data");
        let grad_b = extract_scalar_grad(&b, &grads);

        update_matrix(&mut w_matrix, &grad_w, LR_MATRIX);
        b_scalar -= LR_MATRIX * grad_b;

        let hypo_vec = hypothesis
            .into_data()
            .to_vec::<f32>()
            .expect("hypothesis to_vec");

        println!(
            "Epoch {:2}/20 hypothesis: {:?} Cost: {:.6}",
            epoch,
            hypo_vec,
            loss_value
        );
    }

    println!("\n=== 3. 학습된 모델로 예측하기 ===");
    let new_input = [75.0f32, 85.0, 72.0];
    let prediction = new_input[0] * w_matrix[0][0]
        + new_input[1] * w_matrix[1][0]
        + new_input[2] * w_matrix[2][0]
        + b_scalar;
    println!(
        "Predicted value for input {:?}: {:.6}",
        new_input, prediction
    );
}

fn extract_scalar_grad<const D: usize, AD: burn::tensor::backend::AutodiffBackend>(
    tensor: &Tensor<AD, D>,
    grads: &AD::Gradients,
) -> f32 {
    tensor
        .grad(grads)
        .expect("Grad lookup")
        .into_data()
        .to_vec::<f32>()
        .expect("grad to_vec")[0]
}

fn update_matrix(weights: &mut [[f32; 1]; 3], grad: &[f32], lr: f32) {
    for (row, grad_row) in weights.iter_mut().zip(grad.chunks_exact(1)) {
        row[0] -= lr * grad_row[0];
    }
}
