use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::{
    transformer::{TransformerEncoderConfig, TransformerEncoderInput},
    PositionalEncodingConfig,
};
use burn::tensor::Tensor;

pub fn example() {
    type Backend = Wgpu<f32>;

    let device = WgpuDevice::default();
    let batch_size = 2;
    let seq_len = 4;
    let d_model = 8;

    // 임의의 임베딩(배치, 시퀀스 길이, 임베딩 차원)
    let tokens = Tensor::<Backend, 3>::from_floats(
        [
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0.2, 0.1, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.1],
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2],
            ],
            [
                [0.0, 0.3, 0.6, 0.9, 0.3, 0.6, 0.9, 0.3],
                [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5],
                [0.5, 0.4, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
                [0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.2, 0.4],
            ],
        ],
        &device,
    );

    // 위치 정보 추가
    let pos_enc = PositionalEncodingConfig::new(d_model).init::<Backend>(&device);
    let tokens = pos_enc.forward(tokens);

    // 가장 기본적인 인코더 한 층짜리 설정
    let encoder = TransformerEncoderConfig::new(d_model, d_model * 4, 2, 1)
        .with_dropout(0.0)
        .init::<Backend>(&device);

    let output = encoder.forward(TransformerEncoderInput::new(tokens));

    println!("입력 배치 크기  : {}", batch_size);
    println!("시퀀스 길이     : {}", seq_len);
    println!("모델 차원(d_model): {}", d_model);
    println!("결과 텐서 shape : {:?}", output.shape());

    let flat = output.into_data().to_vec::<f32>().expect("tensor to vec");
    println!("첫 배치 첫 토큰 결과 값: {:?}", &flat[0..d_model]);
}
