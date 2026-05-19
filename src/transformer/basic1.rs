use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::AutodiffModule;
use burn::nn::{
    loss::{MseLoss, Reduction},
    transformer::{TransformerEncoderConfig, TransformerEncoderInput},
    Linear, LinearConfig, PositionalEncoding, PositionalEncodingConfig,
};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::{Int, Tensor};
use serde::Deserialize;

const CSV_PATH: &str = "dataset/transformer_series.csv";
const SEQ_LEN: usize = 6;
const INPUT_DIM: usize = 4;
const D_MODEL: usize = 16;
const D_FF: usize = 64;
const N_HEADS: usize = 4;
const N_LAYERS: usize = 2;
const EPOCHS: usize = 60;
const LEARNING_RATE: f64 = 8e-3;

#[derive(Clone, Debug, Deserialize)]
struct SeriesRow {
    #[allow(dead_code)]
    step: usize,
    trend: f32,
    wave: f32,
    event: f32,
    value: f32,
}

#[derive(Clone, Debug)]
struct WindowSample {
    x: [[f32; INPUT_DIM]; SEQ_LEN],
    y: f32,
}

#[derive(Clone, Debug)]
struct SeriesBatch<B: Backend> {
    x: Tensor<B, 3>,
    y: Tensor<B, 2>,
}

#[derive(Module, Debug)]
struct TransformerRegressor<B: Backend> {
    input: Linear<B>,
    position: PositionalEncoding<B>,
    encoder: burn::nn::transformer::TransformerEncoder<B>,
    output: Linear<B>,
}

impl<B: Backend> TransformerRegressor<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(INPUT_DIM, D_MODEL).init(device),
            position: PositionalEncodingConfig::new(D_MODEL)
                .with_max_sequence_size(SEQ_LEN)
                .init(device),
            encoder: TransformerEncoderConfig::new(D_MODEL, D_FF, N_HEADS, N_LAYERS)
                .with_dropout(0.0)
                .init(device),
            output: LinearConfig::new(D_MODEL, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch = x.dims()[0];
        let x = self.input.forward(x);
        let x = self.position.forward(x);
        let x = self.encoder.forward(TransformerEncoderInput::new(x));

        let last_index =
            Tensor::<B, 1, Int>::arange((SEQ_LEN as i64 - 1)..SEQ_LEN as i64, &x.device());
        let last = x.select(1, last_index).reshape([batch, D_MODEL]);

        self.output.forward(last)
    }

    fn loss(&self, batch: SeriesBatch<B>) -> Tensor<B, 1> {
        let pred = self.forward(batch.x);
        MseLoss::new().forward(pred, batch.y, Reduction::Mean)
    }
}

fn load_rows(path: &str) -> Vec<SeriesRow> {
    let mut reader = csv::Reader::from_path(path).expect("CSV 파일을 열 수 없습니다");
    reader
        .deserialize()
        .map(|row| row.expect("CSV 파싱 실패"))
        .collect()
}

fn make_windows(rows: &[SeriesRow]) -> Vec<WindowSample> {
    if rows.len() <= SEQ_LEN {
        panic!(
            "데이터가 너무 적습니다. 최소 {}행 이상 필요합니다.",
            SEQ_LEN + 1
        );
    }

    let mut samples = Vec::with_capacity(rows.len() - SEQ_LEN);
    for start in 0..(rows.len() - SEQ_LEN) {
        let mut x = [[0.0; INPUT_DIM]; SEQ_LEN];
        for offset in 0..SEQ_LEN {
            let row = &rows[start + offset];
            x[offset] = [row.trend, row.wave, row.event, row.value];
        }

        samples.push(WindowSample {
            x,
            y: rows[start + SEQ_LEN].value,
        });
    }

    samples
}

fn make_batch<B: Backend>(samples: &[WindowSample], device: &B::Device) -> SeriesBatch<B> {
    let mut xs = Vec::with_capacity(samples.len());
    let mut ys = Vec::with_capacity(samples.len());

    for sample in samples {
        xs.push(Tensor::<B, 2>::from_floats(sample.x, device));
        ys.push(Tensor::<B, 2>::from_floats([[sample.y]], device));
    }

    let x = Tensor::cat(xs, 0).reshape([samples.len(), SEQ_LEN, INPUT_DIM]);
    let y = Tensor::cat(ys, 0);

    SeriesBatch { x, y }
}

fn loss_value<B: Backend>(loss: Tensor<B, 1>) -> f32 {
    loss.into_data()
        .to_vec::<f32>()
        .expect("loss tensor to vec 실패")[0]
}

fn predict_one<B: Backend>(
    model: &TransformerRegressor<B>,
    sample: &WindowSample,
    device: &B::Device,
) -> f32 {
    model
        .forward(Tensor::<B, 2>::from_floats(sample.x, device).reshape([1, SEQ_LEN, INPUT_DIM]))
        .into_data()
        .to_vec::<f32>()
        .expect("prediction tensor to vec 실패")[0]
}

pub fn example() {
    type BackendF = Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;

    let device = WgpuDevice::default();
    let rows = load_rows(CSV_PATH);
    let samples = make_windows(&rows);
    let split = samples.len() - 5;
    let train_samples = &samples[..split];
    let valid_samples = &samples[split..];

    let mut model = TransformerRegressor::<AD>::new(&device);
    let mut optim = AdamConfig::new().init::<AD, TransformerRegressor<AD>>();

    println!("Transformer CSV 학습 예시");
    println!("CSV: {CSV_PATH}");
    println!("윈도우: {SEQ_LEN}행 입력 -> 다음 value 예측");
    println!(
        "학습 샘플: {}, 추론 확인 샘플: {}",
        train_samples.len(),
        valid_samples.len()
    );

    for epoch in 1..=EPOCHS {
        let batch = make_batch::<AD>(train_samples, &device);
        let loss = model.loss(batch);
        let printed_loss = loss_value(loss.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(LEARNING_RATE, model, grads);

        if epoch == 1 || epoch % 10 == 0 || epoch == EPOCHS {
            println!("epoch {epoch:02} loss {printed_loss:.6}");
        }
    }

    let trained = model.valid();
    println!();
    println!("추론 결과");
    for (index, sample) in valid_samples.iter().enumerate() {
        let pred = predict_one(&trained, sample, &device);
        println!(
            "sample {} -> predicted {:.4}, actual {:.4}",
            index + 1,
            pred,
            sample.y
        );
    }
}
