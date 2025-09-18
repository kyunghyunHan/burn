use burn::backend::autodiff;
use burn::backend::autodiff::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::data::dataloader::{batcher::Batcher, DataLoaderBuilder};
use burn::nn::{
    loss::{MseLoss, Reduction},
    Linear, LinearConfig, Lstm, LstmConfig,
};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use burn::train::{
    metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
};
use serde::Deserialize;
use std::path::Path;
// 하이퍼파라미터
const INPUT_DIM: usize = 5;
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 1;
const SEQ_LEN: usize = 7;
const BATCH: usize = 64;
const EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 1e-3;

// CSV 한 줄
#[derive(Clone, Debug, Deserialize)]
struct StockRow {
    Open: f32,
    High: f32,
    Low: f32,
    Volume: f32,
    Close: f32,
}

// DataLoader가 반환할 배치
#[derive(Clone, Debug)]
struct StockBatch<B: Backend> {
    x: Tensor<B, 3>, // [batch, seq, features]
    y: Tensor<B, 2>, // [batch, 1]
}

// CSV 전체를 메모리에 올림
#[derive(Clone)]
struct StockDataset {
    rows: Vec<StockRow>,
}

impl StockDataset {
    fn load_csv(path: &str) -> Self {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .unwrap();
        let rows: Vec<StockRow> = rdr.deserialize().map(|r| r.unwrap()).collect();
        Self { rows }
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
    fn get(&self, idx: usize) -> Option<StockRow> {
        self.rows.get(idx).cloned()
    }
}

// Batcher<B,I,O>
struct StockBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> StockBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, StockRow, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<StockRow>, device: &B::Device) -> StockBatch<B> {
        let mut seqs = Vec::new();
        let mut targs = Vec::new();

        // SEQ_LEN + 1 길이의 윈도우에서 마지막 값을 예측
        for window in items.windows(SEQ_LEN + 1) {
            let mut seq_array = [[0.0f32; INPUT_DIM]; SEQ_LEN];
            for (i, row) in window[..SEQ_LEN].iter().enumerate() {
                seq_array[i] = [row.Open, row.High, row.Low, row.Volume, row.Close];
            }

            seqs.push(Tensor::<B, 2>::from_floats(seq_array, device));
            targs.push(Tensor::<B, 2>::from_floats(
                [[window[SEQ_LEN].Close]],
                device,
            ));
        }

        let x = Tensor::cat(seqs, 0).reshape([items.len() - SEQ_LEN, SEQ_LEN, INPUT_DIM]);
        let y = Tensor::cat(targs, 0);

        StockBatch { x, y }
    }
}

// LSTM 모델
#[derive(Module, Debug)]
struct LstmNet<B: Backend> {
    lstm: Lstm<B>,
    fc: Linear<B>,
}

impl<B: Backend> LstmNet<B> {
    fn new(dev: &B::Device) -> Self {
        Self {
            lstm: LstmConfig::new(INPUT_DIM, HIDDEN_DIM, true).init(dev),
            fc: LinearConfig::new(HIDDEN_DIM, OUTPUT_DIM).init(dev),
        }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (out, _) = self.lstm.forward(x, None);
        let indices = Tensor::<B, 1, Int>::from_ints(
            [SEQ_LEN as i64 - 1],   // i64 배열
            &out.device(),           // 디바이스
        );
        
        let last = out.select(1, indices).squeeze(1); // [batch, hidden]
        self.fc.forward(last)
    }
    fn forward_reg(&self, x: Tensor<B, 3>, y: Tensor<B, 2>) -> RegressionOutput<B> {
        let pred = self.forward(x);
        let loss = MseLoss::new().forward(pred.clone(), y.clone(), Reduction::Mean);
        RegressionOutput::new(loss, pred, y)
    }
}

// 학습/검증 step
impl<B: AutodiffBackend> TrainStep<StockBatch<B>, RegressionOutput<B>> for LstmNet<B> {
    fn step(&self, batch: StockBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let out = self.forward_reg(batch.x, batch.y);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}
impl<B: Backend> ValidStep<StockBatch<B>, RegressionOutput<B>> for LstmNet<B> {
    fn step(&self, batch: StockBatch<B>) -> RegressionOutput<B> {
        self.forward_reg(batch.x, batch.y)
    }
}
use burn::tensor::Int;

pub fn example() {
    type BackendF = burn::backend::wgpu::Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;
    let device = WgpuDevice::default();

    // 데이터 로드
    let dataset = StockDataset::load_csv("dataset/train.csv");

    let batcher_train = StockBatcher::<AD>::new(device.clone());

    type Inner = <AD as burn::tensor::backend::AutodiffBackend>::InnerBackend;

    let batcher_valid = StockBatcher::<Inner>::new(device.clone());
    // DataLoader
    let loader = DataLoaderBuilder::<AD, StockRow, StockBatch<AD>>::new(batcher_train)
        .batch_size(BATCH)
        .shuffle(42)
        .num_workers(1)
        .build(dataset.clone());

    let loader_valid = DataLoaderBuilder::<Inner, StockRow, StockBatch<Inner>>::new(batcher_valid)
        .batch_size(BATCH)
        .shuffle(42)
        .num_workers(1)
        .build(dataset);

    // 모델 & 옵티마이저
    let model = LstmNet::new(&device);
    let optim_cfg = AdamConfig::new(); // ⚡ Config만 생성
    let optim = optim_cfg.init(); // ⚡ OptimizerAdaptor 로 변환

    // 학습
    let learner = LearnerBuilder::new("./model")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(EPOCHS)
        .build(model, optim, LEARNING_RATE);

    let trained = learner.fit(loader, loader_valid);

    trained
        .save_file("./model/final", &CompactRecorder::new())
        .unwrap();
}

use burn::data::dataset::Dataset;

impl Dataset<StockRow> for StockDataset {
    fn get(&self, index: usize) -> Option<StockRow> {
        self.rows.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
}


pub  fn infer_example() {
    type BackendF = burn::backend::wgpu::Wgpu<f32>;
    type B = burn::backend::autodiff::Autodiff<BackendF>;
    let device = WgpuDevice::default();

    // 모델 로드
    let mut model = LstmNet::new(&device);
    let record = CompactRecorder::new()
        .load("./model/final".into(), &device)
        .unwrap();
    model = model.load_record(record);

    // 테스트 입력 (임의 데이터)
    let input = [[[1.0f32,1.1,0.9,1000.0,1.05]; SEQ_LEN]; 1];
    let x = Tensor::<B, 3>::from_floats(input, &device);

    // 추론
    let out = model.forward(x);
    println!("예측 값: {:?}", out.to_data());
}
// use burn::tensor::Int;
// use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;

pub fn evaluate() {
    type BackendF = burn::backend::wgpu::Wgpu<f32>;
    type B = burn::backend::autodiff::Autodiff<BackendF>;
    let device = WgpuDevice::default();

    // 📂 모델 로드
    let mut model = LstmNet::new(&device);
    let record = CompactRecorder::new()
        .load("./model/final".into(), &device)
        .expect("모델 로드 실패");
    model = model.load_record(record);

    // 📂 테스트 데이터 로드
    let test = StockDataset::load_csv("dataset/test.csv");
    let mut predictions = Vec::new();
    let mut targets     = Vec::new();

    // 시퀀스 단위로 예측
    for i in 0..test.len().saturating_sub(SEQ_LEN) {
        let mut seq_array = [[0.0f32; INPUT_DIM]; SEQ_LEN];
        for j in 0..SEQ_LEN {
            let row = &test.rows[i + j];
            seq_array[j] = [row.Open, row.High, row.Low, row.Volume, row.Close];
        }
        let x = Tensor::<B, 3>::from_floats([seq_array], &device);
        let pred = model
        .forward(x)
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];  
        predictions.push(pred);
        targets.push(test.rows[i + SEQ_LEN].Close);
    }

    // 🎯 MAE / RMSE 계산
    let n = predictions.len() as f32;
    let mae: f32 = predictions.iter()
        .zip(&targets)
        .map(|(p, t)| (p - t).abs())
        .sum::<f32>() / n;
    let rmse: f32 = (predictions.iter()
        .zip(&targets)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>() / n).sqrt();

    println!("테스트 결과 ({} 샘플)", predictions.len());
    println!("Mean Absolute Error (MAE): {:.4}", mae);
    println!("Root Mean Square Error (RMSE): {:.4}", rmse);
}
