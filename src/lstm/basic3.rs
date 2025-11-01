use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::data::dataloader::{batcher::Batcher, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::nn::{
    loss::{MseLoss, Reduction},
    Linear, LinearConfig, Lstm, LstmConfig,
};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use rand::Rng;
use burn::train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep};

const SEQ_LEN: usize = 10;
const INPUT_DIM: usize = 1;
const HIDDEN: usize = 20;
const BATCH: usize = 32;
const EPOCHS: usize = 10;
const LR: f64 = 1e-3;

// ----------------------
// Sin 데이터셋
// ----------------------
#[derive(Clone)]
struct SinDataset {
    data: Vec<f32>,
}

impl SinDataset {
    fn new(n_samples: usize, seq_length: usize) -> Self {
        let mut rng = rand::rng();
        let mut data = Vec::new();

        for _ in 0..n_samples {
            let freq = rng.random_range(0.1..1.0);
            for i in 0..seq_length {
                data.push((i as f32 * freq).sin());
            }
        }
        Self { data }
    }
}

impl Dataset<f32> for SinDataset {
    fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

// ----------------------
// 배처 (시퀀스 → (x,y))
// ----------------------
#[derive(Clone)]
struct SinBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SinBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
struct SinBatch<B: Backend> {
    x: Tensor<B, 3>,
    y: Tensor<B, 2>,
}

impl<B: Backend> Batcher<B, f32, SinBatch<B>> for SinBatcher<B> {
    fn batch(&self, items: Vec<f32>, _d: &B::Device) -> SinBatch<B> {
        if items.len() <= SEQ_LEN {
            panic!(
                "배치 크기가 시퀀스 길이보다 작습니다: items.len()={}, SEQ_LEN={}",
                items.len(),
                SEQ_LEN
            );
        }

        let n = items.len() - 1;
        let mut xs: Vec<Tensor<B, 2>> = Vec::new();
        let mut ys: Vec<Tensor<B, 2>> = Vec::new();

        for i in 0..(n - SEQ_LEN) {
            let mut seq = [[0.0f32; INPUT_DIM]; SEQ_LEN];
            for j in 0..SEQ_LEN {
                seq[j][0] = items[i + j];
            }
            xs.push(Tensor::from_floats(seq, &self.device));
            ys.push(Tensor::from_floats([[items[i + SEQ_LEN]]], &self.device));
        }

        let x = Tensor::cat(xs, 0).reshape([(n - SEQ_LEN), SEQ_LEN, INPUT_DIM]);
        let y = Tensor::cat(ys, 0);
        SinBatch { x, y }
    }
}

// ----------------------
// LSTM 모델
// ----------------------
#[derive(Module, Debug)]
struct LstmNet<B: Backend> {
    lstm: Lstm<B>,
    fc: Linear<B>,
}

impl<B: Backend> LstmNet<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            lstm: LstmConfig::new(INPUT_DIM, HIDDEN, true).init(device),
            fc: LinearConfig::new(HIDDEN, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (out, _) = self.lstm.forward(x, None);
        let idx = Tensor::<B, 1, Int>::from_ints([SEQ_LEN as i64 - 1], &out.device());
        let last = out.select(1, idx).squeeze(1);
        self.fc.forward(last)
    }

    fn forward_loss(&self, x: Tensor<B, 3>, y: Tensor<B, 2>) -> RegressionOutput<B> {
        let pred = self.forward(x.clone());
        let loss = MseLoss::new().forward(pred.clone(), y.clone(), Reduction::Mean);
        RegressionOutput::new(loss, pred, y)
    }
}

impl<B: AutodiffBackend> TrainStep<SinBatch<B>, RegressionOutput<B>> for LstmNet<B> {
    fn step(&self, batch: SinBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let out = self.forward_loss(batch.x, batch.y);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}

impl<B: Backend> ValidStep<SinBatch<B>, RegressionOutput<B>> for LstmNet<B> {
    fn step(&self, batch: SinBatch<B>) -> RegressionOutput<B> {
        self.forward_loss(batch.x, batch.y)
    }
}

// ----------------------
// 실행
// ----------------------
pub fn example() {
    type BackendF = Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;
    type Inner = <AD as AutodiffBackend>::InnerBackend;

    let device = WgpuDevice::default();
    let dataset_train = SinDataset::new(1000, SEQ_LEN + 1);
    let dataset_valid = SinDataset::new(200, SEQ_LEN + 1);

    let train_loader = DataLoaderBuilder::<AD, f32, SinBatch<AD>>::new(SinBatcher::new(device.clone()))
        .batch_size(BATCH)
        .shuffle(42)
        .num_workers(1)
        .build(dataset_train);

    let valid_loader =
        DataLoaderBuilder::<Inner, f32, SinBatch<Inner>>::new(SinBatcher::new(device.clone()))
            .batch_size(BATCH)
            .shuffle(42)
            .num_workers(1)
            .build(dataset_valid);

    let model = LstmNet::<AD>::new(&device);
    let optim = AdamConfig::new().init();

    let learner = LearnerBuilder::new("./sine_model")
        .devices(vec![device.clone()])
        .num_epochs(EPOCHS)
        .build(model, optim, LR);

    let trained = learner.fit(train_loader, valid_loader);

    println!("✅ 학습 완료!");

    // 테스트 샘플
    let mut test = Vec::new();
    for i in 0..SEQ_LEN {
        test.push((i as f32 * 0.5).sin());
    }

    let mut seq = [[0.0f32; INPUT_DIM]; SEQ_LEN];
    for (idx, value) in test.iter().enumerate() {
        seq[idx][0] = *value;
    }

    let infer_model = trained.valid();
    let x = Tensor::<BackendF, 3>::from_floats([seq], &device);
    let y = infer_model
        .forward(x)
        .into_data()
        .to_vec::<f32>()
        .expect("결과 변환 실패");

    println!("예측 결과 = {:?}", y);
}
