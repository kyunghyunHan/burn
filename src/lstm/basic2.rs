use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::data::dataloader::{batcher::Batcher, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::nn::{
    loss::{MseLoss, Reduction},
    Linear, LinearConfig, Lstm, LstmConfig,
};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Int;
use burn::train::{
    metric::LossMetric, LearnerBuilder, LearningStrategy, RegressionOutput, TrainOutput, TrainStep,
    ValidStep,
};

// ===========================
// 간단한 데이터셋 (사인파)
// ===========================
#[derive(Clone)]
struct SineDataset {
    data: Vec<f32>,
}

impl SineDataset {
    fn new(len: usize) -> Self {
        let data = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
        Self { data }
    }
}

impl Dataset<f32> for SineDataset {
    fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

// ===========================
// 배치 구조
// ===========================
#[derive(Clone, Debug)]
struct SineBatch<B: Backend> {
    x: Tensor<B, 3>, // [batch, seq, 1]
    y: Tensor<B, 2>, // [batch, 1]
}

struct SineBatcher<B: Backend> {
    device: B::Device,
    seq_len: usize,
}

impl<B: Backend> SineBatcher<B> {
    fn new(device: B::Device, seq_len: usize) -> Self {
        Self { device, seq_len }
    }
}
impl<B: Backend> Batcher<B, f32, SineBatch<B>> for SineBatcher<B> {
    fn batch(&self, items: Vec<f32>, _device: &B::Device) -> SineBatch<B> {
        let mut xs = Vec::new();
        let mut ys = Vec::new();

        for i in 0..(items.len() - self.seq_len) {
            // ✅ [seq_len, 1] 형태의 2D 배열 생성
            let seq_slice: Vec<f32> = (0..self.seq_len)
                .map(|j| items[i + j])
                .collect();

            let x_tensor = Tensor::<B, 1>::from_floats(seq_slice.as_slice(), &self.device)
                .reshape([self.seq_len, 1]); // [seq_len, 1]

            let value = [[items[i + self.seq_len]]]; // ✅ [1, 1] 2D 배열
            let y_tensor = Tensor::<B, 2>::from_floats(value, &self.device);

            xs.push(x_tensor);
            ys.push(y_tensor);
        }

        // ✅ 최종 텐서 모양: [batch, seq, feature], [batch, 1]
        let x =
            Tensor::cat(xs, 0).reshape([(items.len() - self.seq_len), self.seq_len, 1]);
        let y = Tensor::cat(ys, 0);

        SineBatch { x, y }
    }
}


// ===========================
// LSTM 모델
// ===========================
#[derive(Module, Debug)]
struct LstmModel<B: Backend> {
    lstm: Lstm<B>,
    fc: Linear<B>,
}

impl<B: Backend> LstmModel<B> {
    fn new(device: &B::Device, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            lstm: LstmConfig::new(input_dim, hidden_dim, true).init(device), // ✅ bias 추가
            fc: LinearConfig::new(hidden_dim, output_dim).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (out, _) = self.lstm.forward(x, None);

        let seq_dim = out.dims()[1] as i64 - 1;
        let last_index = Tensor::<B, 1, Int>::from_ints([seq_dim], &out.device());
        let last = out.select(1, last_index).squeeze::<2>(); // ✅ [batch, hidden]
        self.fc.forward(last)
    }

    fn forward_reg(&self, x: Tensor<B, 3>, y: Tensor<B, 2>) -> RegressionOutput<B> {
        let pred = self.forward(x);
        let loss = MseLoss::new().forward(pred.clone(), y.clone(), Reduction::Mean);
        RegressionOutput::new(loss, pred, y)
    }
}

impl<B: AutodiffBackend> TrainStep<SineBatch<B>, RegressionOutput<B>> for LstmModel<B> {
    fn step(&self, batch: SineBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let out = self.forward_reg(batch.x, batch.y);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}

impl<B: Backend> ValidStep<SineBatch<B>, RegressionOutput<B>> for LstmModel<B> {
    fn step(&self, batch: SineBatch<B>) -> RegressionOutput<B> {
        self.forward_reg(batch.x, batch.y)
    }
}

// ===========================
// 메인 학습 함수
// ===========================
pub fn example() {
    type BackendF = Wgpu<f32>;
    type AD = autodiff::Autodiff<BackendF>;
    let device = WgpuDevice::default();

    const SEQ_LEN: usize = 5;
    const HIDDEN_DIM: usize = 8;
    const EPOCHS: usize = 20;

    // 데이터 준비
    let dataset = SineDataset::new(200);
    let batcher_train = SineBatcher::<AD>::new(device.clone(), SEQ_LEN);
    let batcher_valid =
        SineBatcher::<<AD as AutodiffBackend>::InnerBackend>::new(device.clone(), SEQ_LEN);

    let loader_train = DataLoaderBuilder::<AD, f32, SineBatch<AD>>::new(batcher_train)
        .batch_size(32)
        .num_workers(1)
        .build(dataset.clone());

    let loader_valid = DataLoaderBuilder::<BackendF, f32, SineBatch<BackendF>>::new(batcher_valid)
        .batch_size(32)
        .num_workers(1)
        .build(dataset.clone());

    // 모델 & 옵티마이저
    let model = LstmModel::new(&device, 1, HIDDEN_DIM, 1);
    let optim = AdamConfig::new().init();

    let learner = LearnerBuilder::new("./checkpoints")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(EPOCHS)
        .build(model, optim, 1e-3);

    let trained = learner.fit(loader_train, loader_valid);

    trained
        .model
        .save_file("./checkpoints/final", &CompactRecorder::new())
        .unwrap();
}
pub fn test() {
    type BackendF = Wgpu<f32>;
    let device = WgpuDevice::default();

    // ✅ 1. 모델 생성 및 가중치 로드
    let mut model = LstmModel::<BackendF>::new(&device, 1, 8, 1);
    let record = CompactRecorder::new()
        .load("./checkpoints/final".into(), &device)
        .expect("모델 로드 실패");
    model = model.load_record(record);

    // ✅ 2. 테스트용 입력 생성 (사인파 일부 구간)
    let seq_len = 5;
    let test_input: Vec<f32> = (0..seq_len)
        .map(|i| (i as f32 * 0.1).sin()) // 예: 0~0.5 구간 사인파
        .collect();

    // shape 맞추기: [1, seq_len, 1]
    let x = Tensor::<BackendF, 1>::from_floats(
        test_input.as_slice(),
        &device,
    ).reshape([1, seq_len as i32, 1]);

    // ✅ 3. 추론 실행
    let output = model.forward(x).to_data();
    println!("예측 결과: {:?}", output);

    // 필요하다면 .to_vec() 으로 꺼내기
    let pred: Vec<f32> = output.to_vec().unwrap();
    println!("다음 시점 예측값: {:?}", pred[0]);
}
