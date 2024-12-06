use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::data::dataset::vision::MnistItem;
use burn::data::dataset::InMemDataset;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::DropoutConfig;
// use crossbeam_epoch::atomic::Pointable;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::Lstm;
use burn::nn::Relu;
use burn::nn::{Dropout, LstmConfig};
use burn::prelude::Backend;
use burn::prelude::Int;
use burn::prelude::Module;
use burn::prelude::Tensor;
use burn::prelude::TensorData;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::tensor::ElementConversion;
use burn::train::metric::AccuracyMetric;
use burn::train::metric::LossMetric;
use burn::train::ClassificationOutput;
use burn::train::RegressionOutput;
use burn::train::TrainOutput;
use burn::train::TrainStep;
use burn::train::ValidStep;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    tensor::backend::AutodiffBackend,
    train::{
        renderer::{MetricState, MetricsRenderer, TrainingProgress},
        LearnerBuilder,
    },
};
//7일의 정보를 활용하므로 Sequence 7 output = 1
use serde::{Deserialize, Serialize};
use std::path::Path;
// 상수 정의
const SEQUENCE_LENGTH: usize = 10; // 시퀀스 길이
const HIDDEN_SIZE: usize = 64; // LSTM 히든 크기
const LEARNING_RATE: f64 = 0.001; // 학습률

// 데이터 구조체
#[derive(Clone, Debug, Deserialize)]
struct TimeSeriesData {
    Open: f32,    // i64에서 f32로 변경
    High: f32,
    Low: f32,
    Volume: f32,  // i32에서 f32로 변경
    Close: f32,
}
// 배치 구조체
#[derive(Clone, Debug)]
struct TimeSeriesBatch<B: Backend> {
    sequence: Tensor<B, 3>, // [batch_size, sequence_length, features]
    targets: Tensor<B, 2>,  // [batch_size, 1]
}

// 데이터셋 구조체
pub struct StockDataset {
    dataset: InMemDataset<TimeSeriesData>,
}

impl Dataset<TimeSeriesData> for StockDataset {
    fn get(&self, index: usize) -> Option<TimeSeriesData> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl StockDataset {
    pub fn new() -> Result<Self, std::io::Error> {
        let path = Path::new(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("../dataset/test.csv");

        let dataset = InMemDataset::from_csv(
            path,
            &csv::ReaderBuilder::new()
                .has_headers(true)
                .delimiter(b',')
                .flexible(true),
        )
        .unwrap();

        Ok(Self { dataset })
    }
}

// 배처 구현
#[derive(Clone)]
struct TimeSeriesBatcher<B: Backend> {
    device: B::Device,
    sequence_length: usize,
}

impl<B: Backend> TimeSeriesBatcher<B> {
    fn new(device: B::Device, sequence_length: usize) -> Self {
        Self {
            device,
            sequence_length,
        }
    }
}

impl<B: Backend> Batcher<TimeSeriesData, TimeSeriesBatch<B>> for TimeSeriesBatcher<B> {
    fn batch(&self, items: Vec<TimeSeriesData>) -> TimeSeriesBatch<B> {
        let mut sequences = Vec::new();
        let mut targets = Vec::new();

        for window in items.windows(self.sequence_length + 1) {
            // 5개의 특성을 가진 시퀀스 배열 생성 (Open, High, Low, Volume, Close)
            let mut seq_array = [[0.0f32; 5]; SEQUENCE_LENGTH];
            
            for (i, item) in window[..self.sequence_length].iter().enumerate() {
                seq_array[i][0] = item.Open as f32;  // i64를 f32로 변환
                seq_array[i][1] = item.High;
                seq_array[i][2] = item.Low;
                seq_array[i][3] = item.Volume as f32;  // i32를 f32로 변환
                seq_array[i][4] = item.Close;
            }

            sequences.push(Tensor::<B, 2>::from_floats(seq_array, &self.device));

            // 다음 날의 종가(Close)를 타겟으로 사용
            targets.push(Tensor::<B, 2>::from_floats(
                [[window[self.sequence_length].Close]],
                &self.device,
            ));
        }

        let sequence = Tensor::cat(sequences.clone(), 0)
            .reshape([sequences.len(), self.sequence_length, 5]);  // 특성 수를 5로 변경
        let targets = Tensor::cat(targets, 0);

        TimeSeriesBatch { sequence, targets }
    }
}
// 모델 설정
#[derive(Module, Debug)]
pub struct LstmModel2<B: Backend> {
    lstm: Lstm<B>,
    linear: Linear<B>,
    relu: Relu,
}
// #[derive(Config, Debug)]
// pub struct ModelConfig {
//     #[config(default = 1)]
//     input_size: usize,
//     #[config(default = 64)]
//     hidden_size: usize,
//     #[config(default = 1)]
//     output_size: usize,
//     #[config(default = "0.5")]
//     dropout: f64,
// }
impl<B: Backend> LstmModel2<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            lstm: LstmConfig::new(5, HIDDEN_SIZE, true).init(device),  // 5개 특성 입력
            linear: LinearConfig::new(SEQUENCE_LENGTH * HIDDEN_SIZE, 1).init(device),  // 수정된 부분
            relu: Relu::new(),
        }
    }
    // pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
    //     let (output, _) = self.lstm.forward(x, None);

    //     // 마지막 시점의 출력만 선택
    //     // let last_output = output.select(batch_size, seq_len - 1); // [batch_size, hidden_size]
    //     let output = self.linear.forward(output).squeeze(0); // [batch_size, 1]
    //     self.relu.forward(output) // [batch_size, 1]
    // }
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (output, _) = self.lstm.forward(x, None);
        
        // 차원 정보 가져오기
        let batch_size = output.dims()[0];
        let total_features = SEQUENCE_LENGTH * HIDDEN_SIZE;
        
        // 명시적으로 크기 지정
        let output = output.reshape([batch_size, total_features]);
        let output = self.linear.forward(output);
        
        self.relu.forward(output)
    }
    pub fn forward_regression(
        &self,
        x: Tensor<B, 3>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(x);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);
        RegressionOutput::new(loss, output, targets)
    }
}

// 학습 스텝 구현

impl<B: AutodiffBackend> TrainStep<TimeSeriesBatch<B>, RegressionOutput<B>> for LstmModel2<B> {
    fn step(&self, batch: TimeSeriesBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.sequence, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}
impl<B: Backend> ValidStep<TimeSeriesBatch<B>, RegressionOutput<B>> for LstmModel2<B> {
    fn step(&self, batch: TimeSeriesBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.sequence);
        let loss = MseLoss::new().forward(
            output.clone(),
            batch.targets.clone(), // targets clone 추가
            Reduction::Mean,
        );

        RegressionOutput::new(loss, output, batch.targets)
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]

pub struct LstmConfig2 {
    pub d_input: usize,
    pub d_hidden: usize,
    pub bias: bool,
    // pub initializer: Initializer,
}
#[derive(Config)]
pub struct TrainingConfig {
    pub model: LstmConfig2,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}
impl LstmConfig2 {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LstmModel2<B> {
        LstmModel2 {
            lstm: LstmConfig::new(1, HIDDEN_SIZE, true).init(device),
            linear: LinearConfig::new(HIDDEN_SIZE, 1).init(device),
            relu: Relu::new(),
        }
    }
}
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(config.seed);

    let batcher_train = TimeSeriesBatcher::new(device.clone(), SEQUENCE_LENGTH);
    let batcher_test = TimeSeriesBatcher::<B::InnerBackend>::new(device.clone(), SEQUENCE_LENGTH);

    let dataloader = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(1)
        .build(StockDataset::new().unwrap());
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(1)
        .build(StockDataset::new().unwrap());

    // Explicitly specify the type for model
    let model: LstmModel2<B> = LstmModel2::new(&device);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), config.learning_rate);

    let model_trained = learner.fit(dataloader, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

pub fn run() {
    let artifact_dir = "./models/lstm";

    type MyBackend = Wgpu<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let config = TrainingConfig {
        num_epochs: 10,
        batch_size: 64,
        num_workers: 4,
        seed: 42,
        learning_rate: 1.0e-4,
        model: LstmConfig2 {
            d_input: 1,
            d_hidden: HIDDEN_SIZE,
            bias: true,
        },
        optimizer: AdamConfig::new(),
    };
    // 학습 실행
    train::<MyAutodiffBackend>(artifact_dir, config, device.clone());

    println!("Training completed successfully!");
}
