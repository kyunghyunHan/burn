use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::data::dataloader::{batcher::Batcher, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::lr_scheduler::constant::ConstantLr;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use burn::train::{
    metric::LossMetric, ExecutionStrategy, InferenceStep, Learner, RegressionOutput,
    SupervisedTraining, TrainOutput, TrainStep, TrainingStrategy,
};
use serde::Deserialize;

// =========================
// 하이퍼파라미터 (선형 모델)
// =========================
const INPUT_DIM: usize = 5;
const OUTPUT_DIM: usize = 1;
const SEQ_LEN: usize = 14;
const BATCH: usize = 128;
const EPOCHS: usize = 40;
const LEARNING_RATE: f64 = 5e-4;
const EPS: f32 = 1e-6;

// =========================
// CSV 한 줄
// =========================
#[derive(Clone, Debug, Deserialize)]
struct StockRow {
    Open: f32,
    High: f32,
    Low: f32,
    Volume: f32,
    Close: f32,
}

// =========================
// Dataset: CSV 전체를 메모리에 로드
// =========================
#[derive(Clone)]
struct StockDataset {
    rows: Vec<StockRow>,
}

impl StockDataset {
    fn load_csv(path: &str) -> Self {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .expect("CSV 파일을 열 수 없습니다");
        let rows: Vec<StockRow> = rdr
            .deserialize()
            .map(|r| r.expect("CSV 파싱 실패"))
            .collect();
        if rows.len() <= SEQ_LEN {
            panic!(
                "데이터가 너무 적습니다 ({}행). 최소 {}행 이상 필요합니다.",
                rows.len(),
                SEQ_LEN + 1
            );
        }
        Self { rows }
    }

    fn len(&self) -> usize {
        self.rows.len()
    }
}

impl Dataset<StockRow> for StockDataset {
    fn get(&self, index: usize) -> Option<StockRow> {
        self.rows.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
}

// =========================
// 정규화 통계
// =========================
#[derive(Clone, Copy, Debug)]
struct NormStats {
    mean: [f32; INPUT_DIM],
    std: [f32; INPUT_DIM],
    y_mean: f32,
    y_std: f32,
}

fn compute_stats(rows: &[StockRow]) -> NormStats {
    let n = rows.len() as f32;
    let mut sum = [0.0f32; INPUT_DIM];
    let mut sum_sq = [0.0f32; INPUT_DIM];
    let mut y_sum = 0.0f32;
    let mut y_sum_sq = 0.0f32;

    for r in rows {
        let feats = [r.Open, r.High, r.Low, r.Volume, r.Close];
        for i in 0..INPUT_DIM {
            sum[i] += feats[i];
            sum_sq[i] += feats[i] * feats[i];
        }
        y_sum += r.Close;
        y_sum_sq += r.Close * r.Close;
    }

    let mut mean = [0.0f32; INPUT_DIM];
    let mut std = [0.0f32; INPUT_DIM];
    for i in 0..INPUT_DIM {
        mean[i] = sum[i] / n;
        let var = (sum_sq[i] / n) - mean[i] * mean[i];
        std[i] = var.max(0.0).sqrt().max(EPS);
    }

    let y_mean = y_sum / n;
    let y_var = (y_sum_sq / n) - y_mean * y_mean;
    let y_std = y_var.max(0.0).sqrt().max(EPS);

    NormStats {
        mean,
        std,
        y_mean,
        y_std,
    }
}

fn normalize(v: f32, mean: f32, std: f32) -> f32 {
    (v - mean) / std
}

fn denormalize(v: f32, mean: f32, std: f32) -> f32 {
    v * std + mean
}

// =========================
// DataLoader가 반환할 배치
// =========================
#[derive(Clone, Debug)]
struct StockBatch<B: Backend> {
    x: Tensor<B, 3>, // [batch, seq, features]
    y: Tensor<B, 2>, // [batch, 1]
}

// =========================
// Batcher: Vec<StockRow> -> (x,y) 시퀀스 윈도우
// =========================
struct StockBatcher<B: Backend> {
    device: B::Device,
    stats: NormStats,
}

impl<B: Backend> StockBatcher<B> {
    fn new(device: B::Device, stats: NormStats) -> Self {
        Self { device, stats }
    }
}

impl<B: Backend> Batcher<B, StockRow, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<StockRow>, _device: &B::Device) -> StockBatch<B> {
        if items.len() <= SEQ_LEN {
            panic!(
                "배치 내 샘플 수가 시퀀스 길이보다 작습니다: items.len()={}, SEQ_LEN={}.",
                items.len(),
                SEQ_LEN
            );
        }

        let mut seqs: Vec<Tensor<B, 2>> = Vec::with_capacity(items.len() - SEQ_LEN);
        let mut targs: Vec<Tensor<B, 2>> = Vec::with_capacity(items.len() - SEQ_LEN);

        for i in 0..(items.len() - SEQ_LEN) {
            let mut seq_array = [[0.0f32; INPUT_DIM]; SEQ_LEN];
            for j in 0..SEQ_LEN {
                let row = &items[i + j];
                let feats = [row.Open, row.High, row.Low, row.Volume, row.Close];
                for k in 0..INPUT_DIM {
                    seq_array[j][k] = normalize(feats[k], self.stats.mean[k], self.stats.std[k]);
                }
            }

            let target_close = items[i + SEQ_LEN].Close;
            let target_norm = normalize(target_close, self.stats.y_mean, self.stats.y_std);

            seqs.push(Tensor::<B, 2>::from_floats(seq_array, &self.device));
            targs.push(Tensor::<B, 2>::from_floats([[target_norm]], &self.device));
        }

        let x = Tensor::cat(seqs, 0).reshape([items.len() - SEQ_LEN, SEQ_LEN, INPUT_DIM]);
        let y = Tensor::cat(targs, 0);

        StockBatch { x, y }
    }
}

// =========================
// 선형 모델
// =========================
#[derive(Module, Debug)]
struct LinearNet<B: Backend> {
    fc: Linear<B>,
}

impl<B: Backend> LinearNet<B> {
    fn new(dev: &B::Device) -> Self {
        let in_dim = SEQ_LEN * INPUT_DIM;
        Self {
            fc: LinearConfig::new(in_dim, OUTPUT_DIM).init(dev),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch = x.dims()[0];
        let flat = x.reshape([batch, SEQ_LEN * INPUT_DIM]);
        self.fc.forward(flat)
    }

    fn forward_reg(&self, x: Tensor<B, 3>, y: Tensor<B, 2>) -> RegressionOutput<B> {
        let pred = self.forward(x);
        let loss = MseLoss::new().forward(pred.clone(), y.clone(), Reduction::Mean);
        RegressionOutput::new(loss, pred, y)
    }
}

impl<B: AutodiffBackend> TrainStep for LinearNet<B> {
    type Input = StockBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let out = self.forward_reg(batch.x, batch.y);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}

impl<B: Backend> InferenceStep for LinearNet<B> {
    type Input = StockBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_reg(batch.x, batch.y)
    }
}

// =========================
// 학습 엔트리
// =========================
pub fn example() {
    type BackendF = Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;
    let device = WgpuDevice::default();

    let dataset = StockDataset::load_csv("dataset/train.csv");
    let stats = compute_stats(&dataset.rows);

    let batcher_train = StockBatcher::<AD>::new(device.clone(), stats);
    type Inner = <AD as AutodiffBackend>::InnerBackend;
    let batcher_valid = StockBatcher::<Inner>::new(device.clone(), stats);

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

    let model = LinearNet::new(&device);
    let optim = AdamConfig::new().init();

    let learner = Learner::new(model, optim, ConstantLr::new(LEARNING_RATE));

    let trained = SupervisedTraining::new("./model", loader, loader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .with_training_strategy(TrainingStrategy::Default(ExecutionStrategy::single(
            device.clone(),
        )))
        .num_epochs(EPOCHS)
        .launch(learner);

    trained
        .model
        .save_file("./model/final", &CompactRecorder::new())
        .expect("모델 저장 실패");

    println!("✅ 학습 완료 (선형 모델)");
}

// =========================
// 추론 예시
// =========================
pub fn infer_example() {
    type B = Wgpu<f32>;
    let device = WgpuDevice::default();

    let train = StockDataset::load_csv("dataset/train.csv");
    let stats = compute_stats(&train.rows);

    let mut model = LinearNet::<B>::new(&device);
    let record = CompactRecorder::new()
        .load("./model/final".into(), &device)
        .expect("모델 로드 실패");
    model = model.load_record(record);

    let input = [[[1.0f32, 1.1, 0.9, 1000.0, 1.05]; SEQ_LEN]; 1];
    let mut norm = [[[0.0f32; INPUT_DIM]; SEQ_LEN]; 1];
    for t in 0..SEQ_LEN {
        for k in 0..INPUT_DIM {
            norm[0][t][k] = normalize(input[0][t][k], stats.mean[k], stats.std[k]);
        }
    }

    let x = Tensor::<B, 3>::from_floats(norm, &device);
    let out = model
        .forward(x)
        .to_data()
        .to_vec::<f32>()
        .expect("to_vec 실패");
    let pred = denormalize(out[0], stats.y_mean, stats.y_std);
    println!("예측 값: {:.6}", pred);
}

// =========================
// 평가 (MAE / RMSE)
// =========================
pub fn evaluate() {
    type B = Wgpu<f32>;
    let device = WgpuDevice::default();

    let train = StockDataset::load_csv("dataset/train.csv");
    let stats = compute_stats(&train.rows);

    let mut model = LinearNet::<B>::new(&device);
    let record = CompactRecorder::new()
        .load("./model/final".into(), &device)
        .expect("모델 로드 실패");
    model = model.load_record(record);

    let test = StockDataset::load_csv("dataset/test.csv");
    if test.len() <= SEQ_LEN {
        panic!(
            "테스트 데이터가 너무 적습니다 ({}행). 최소 {}행 이상 필요.",
            test.len(),
            SEQ_LEN + 1
        );
    }

    let mut predictions = Vec::with_capacity(test.len() - SEQ_LEN);
    let mut targets = Vec::with_capacity(test.len() - SEQ_LEN);

    for i in 0..(test.len() - SEQ_LEN) {
        let mut seq_array = [[0.0f32; INPUT_DIM]; SEQ_LEN];
        for j in 0..SEQ_LEN {
            let row = &test.rows[i + j];
            let feats = [row.Open, row.High, row.Low, row.Volume, row.Close];
            for k in 0..INPUT_DIM {
                seq_array[j][k] = normalize(feats[k], stats.mean[k], stats.std[k]);
            }
        }

        let x = Tensor::<B, 3>::from_floats([seq_array], &device);
        let pred_norm = model
            .forward(x)
            .into_data()
            .to_vec::<f32>()
            .expect("to_vec 실패")[0];
        let pred = denormalize(pred_norm, stats.y_mean, stats.y_std);

        predictions.push(pred);
        targets.push(test.rows[i + SEQ_LEN].Close);
    }

    let n = predictions.len() as f32;
    let mae: f32 = predictions
        .iter()
        .zip(&targets)
        .map(|(p, t)| (p - t).abs())
        .sum::<f32>()
        / n;

    let rmse: f32 = ((predictions
        .iter()
        .zip(&targets)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>())
        / n)
        .sqrt();

    println!("테스트 결과 ({} 샘플)", predictions.len());
    println!("Mean Absolute Error (MAE): {:.4}", mae);
    println!("Root Mean Square Error (RMSE): {:.4}", rmse);
}
