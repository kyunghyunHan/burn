use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::data::dataset::InMemDataset;
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::Lstm;
use burn::nn::LstmConfig;
use burn::nn::Relu;
use burn::prelude::Backend;
use burn::prelude::Module;
use burn::prelude::Tensor;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::tensor::cast::ToElement;
use burn::tensor::ElementConversion;
use burn::train::metric::LossMetric;
use burn::train::RegressionOutput;
use burn::train::TrainOutput;
use burn::train::TrainStep;
use burn::train::ValidStep;
use burn::{
    config::Config, data::dataloader::DataLoaderBuilder, optim::AdamConfig,
    tensor::backend::AutodiffBackend, train::LearnerBuilder,
};

/*LSTM */
//7일의 정보를 활용하므로 Sequence 7 output = 1
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::simple::DiabetesDataset;
// 상수 정의
const SEQUENCE_LENGTH: usize = 7; // 시퀀스 길이
const LEARNING_RATE: f64 = 0.001; // 학습률
const BATCH_SIZE: usize = 100;
const HIDDEN_STATE: usize = 10;
const OUTPUT_DIM: usize = 1;
const HIDDEN_DIM: usize = 10; // 은닉층 차원
const INPUT_DIM: usize = 5; // 입력 차원 (특성 수)
const NUM_LAYERS: usize = 1; // LSTM 층 수
const NUM_EPOCHS: usize = 100;

// 데이터 구조체
#[derive(Clone, Debug, Deserialize)]
struct TimeSeriesData {
    Open: f32, // i64에서 f32로 변경
    High: f32,
    Low: f32,
    Volume: f32, // i32에서 f32로 변경
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
                seq_array[i][0] = item.Open as f32; // i64를 f32로 변환
                seq_array[i][1] = item.High;
                seq_array[i][2] = item.Low;
                seq_array[i][3] = item.Volume as f32; // i32를 f32로 변환
                seq_array[i][4] = item.Close;
            }

            sequences.push(Tensor::<B, 2>::from_floats(seq_array, &self.device));

            // 다음 날의 종가(Close)를 타겟으로 사용
            targets.push(Tensor::<B, 2>::from_floats(
                [[window[self.sequence_length].Close]],
                &self.device,
            ));
        }

        let sequence =
            Tensor::cat(sequences.clone(), 0).reshape([sequences.len(), self.sequence_length, 5]); // 특성 수를 5로 변경
        let targets = Tensor::cat(targets, 0);

        TimeSeriesBatch { sequence, targets }
    }
}
// 모델 설정
#[derive(Module, Debug)]
pub struct LstmModel2<B: Backend> {
    lstm: Lstm<B>,
    linear: Linear<B>,
    // relu: Relu,
}

impl<B: Backend> LstmModel2<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // 입력 차원, 은닉층 차원, bias 설정
            lstm: LstmConfig::new(INPUT_DIM, HIDDEN_DIM, true).init(device),
            // 은닉층 차원에서 출력 차원으로
            linear: LinearConfig::new(HIDDEN_DIM, OUTPUT_DIM).init(device),
            // relu: Relu::new(),
        }
    }
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (output, _) = self.lstm.forward(x, None);

        // 차원 정보 가져오기
        let batch_size = output.dims()[0]; // 93
        let seq_len = output.dims()[1]; // 7
        let hidden_size = output.dims()[2]; // 10

        // 마지막 시퀀스 선택하기 위해 모든 차원을 명시적으로 조작
        let output = output.reshape([batch_size * seq_len, hidden_size]); // [93*7, 10]
        let output = output.reshape([batch_size, seq_len, hidden_size]); // [93, 7, 10]
        let last_seq = output.narrow(1, seq_len - 1, 1); // 마지막 시퀀스만 선택 [93, 1, 10]
        let last_seq = last_seq.squeeze(1); // 중간 차원 제거 [93, 10]

        // 선형 레이어 통과
        let output = self.linear.forward(last_seq);

        output
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
    // #[config(default = 10)]
    pub num_epochs: usize,
    // #[config(default = 64)]
    pub batch_size: usize,
    // #[config(default = 4)]
    pub num_workers: usize,
    // #[config(default = 42)]
    pub seed: u64,
    // #[config(default = 0.001)]
    pub learning_rate: f64,
}
impl LstmConfig2 {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LstmModel2<B> {
        LstmModel2 {
            lstm: LstmConfig::new(INPUT_DIM, HIDDEN_DIM, true).init(device), // 5 -> HIDDEN_DIM
            linear: LinearConfig::new(HIDDEN_DIM, OUTPUT_DIM).init(device),  // HIDDEN_DIM -> 1
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

fn infer<B: Backend>(artifact_dir: &str, device: B::Device, data: Vec<TimeSeriesData>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    // 데이터 통계 출력
    println!("Data length: {}", data.len());

    // 데이터 정규화를 위한 통계 계산
    let mut close_min = f32::INFINITY;
    let mut close_max = f32::NEG_INFINITY;
    for item in &data {
        close_min = close_min.min(item.Close);
        close_max = close_max.max(item.Close);
    }
    println!("Close price range: min={}, max={}", close_min, close_max);

    // 정규화 함수
    let normalize = |x: f32| -> f32 {
        if close_max == close_min {
            println!("Warning: max equals min in normalization");
            return 0.0;
        }
        (x - close_min) / (close_max - close_min)
    };

    // 역정규화 함수
    let denormalize = |x: f32| -> f32 { x * (close_max - close_min) + close_min };

    let mut predictions = Vec::new();
    let mut actual_values = Vec::new();

    // 예측 수행
    for i in 0..data.len() - SEQUENCE_LENGTH {
        let mut input_data = [[[0.0f32; 5]; SEQUENCE_LENGTH]; 1];

        // 입력 데이터 준비 및 정규화
        for j in 0..SEQUENCE_LENGTH {
            let idx = i + j;
            input_data[0][j][0] = normalize(data[idx].Open);
            input_data[0][j][1] = normalize(data[idx].High);
            input_data[0][j][2] = normalize(data[idx].Low);
            input_data[0][j][3] = normalize(data[idx].Volume);
            input_data[0][j][4] = normalize(data[idx].Close);
        }

        // 실제값 저장
        actual_values.push(data[i + SEQUENCE_LENGTH].Close);

        // 예측 수행
        let sequence = Tensor::<B, 3>::from_floats(input_data, &device);
        let output = model.forward(sequence);
        let predicted = output.into_scalar().to_f32();

        println!("Raw prediction: {}", predicted);

        // 예측값 역정규화
        let predicted_denorm = denormalize(predicted);
        println!("Denormalized prediction: {}", predicted_denorm);

        predictions.push(predicted_denorm);
    }

    // 예측 결과 확인
    println!("\nPredictions length: {}", predictions.len());
    println!("Actual values length: {}", actual_values.len());

    if !predictions.is_empty() && !actual_values.is_empty() {
        // MAE 계산
        let mae: f32 = predictions
            .iter()
            .zip(actual_values.iter())
            .map(|(pred, actual)| {
                let diff = (pred - actual).abs();
                println!("Prediction: {}, Actual: {}, Diff: {}", pred, actual, diff);
                diff
            })
            .sum::<f32>()
            / predictions.len() as f32;

        println!("\nMean Absolute Error (MAE): {}", mae);
    } else {
        println!("Error: No predictions or actual values available");
    }
}

pub fn run() {
    let artifact_dir = "./models/lstm";

    type MyBackend = Wgpu<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let config = TrainingConfig {
        num_epochs: NUM_EPOCHS,
        batch_size: BATCH_SIZE,
        num_workers: 4,
        seed: 42,
        learning_rate: LEARNING_RATE,
        model: LstmConfig2 {
            d_input: INPUT_DIM,
            d_hidden: HIDDEN_DIM,
            bias: true,
        },
        optimizer: AdamConfig::new(),
    };
    // 학습 실행
    // train::<MyAutodiffBackend>(artifact_dir, config, device.clone());
    // run_inference();
    // let data= DiabetesDataset::new2().unwrap().
    let test_data = vec![
        TimeSeriesData {
            Open: 828.659973,
            High: 833.450012,
            Low: 828.349976,
            Volume: 1247700.0,
            Close: 831.659973,
        },
        TimeSeriesData {
            Open: 823.02002,
            High: 828.070007,
            Low: 821.655029,
            Volume: 1597800.0,
            Close: 828.070007,
        },
        TimeSeriesData {
            Open: 819.929993,
            High: 824.400024,
            Low: 818.97998,
            Volume: 1281700.0,
            Close: 824.159973,
        },
        TimeSeriesData {
            Open: 819.359985,
            High: 823.0,
            Low: 818.469971,
            Volume: 1304000.0,
            Close: 818.97998,
        },
        TimeSeriesData {
            Open: 819.0,
            High: 823.0,
            Low: 816.0,
            Volume: 1053600.0,
            Close: 820.450012,
        },
        TimeSeriesData {
            Open: 816.0,
            High: 820.958984,
            Low: 815.48999,
            Volume: 1198100.0,
            Close: 819.23999,
        },
        TimeSeriesData {
            Open: 811.700012,
            High: 815.25,
            Low: 809.780029,
            Volume: 1129100.0,
            Close: 813.669983,
        },
        TimeSeriesData {
            Open: 828.659973,
            High: 833.450012,
            Low: 828.349976,
            Volume: 1247700.0,
            Close: 831.659973,
        },
        TimeSeriesData {
            Open: 823.02002,
            High: 828.070007,
            Low: 821.655029,
            Volume: 1597800.0,
            Close: 828.070007,
        },
        TimeSeriesData {
            Open: 819.929993,
            High: 824.400024,
            Low: 818.97998,
            Volume: 1281700.0,
            Close: 824.159973,
        },
        TimeSeriesData {
            Open: 819.359985,
            High: 823.0,
            Low: 818.469971,
            Volume: 1304000.0,
            Close: 818.97998,
        },
        TimeSeriesData {
            Open: 819.0,
            High: 823.0,
            Low: 816.0,
            Volume: 1053600.0,
            Close: 820.450012,
        },
        TimeSeriesData {
            Open: 816.0,
            High: 820.958984,
            Low: 815.48999,
            Volume: 1198100.0,
            Close: 819.23999,
        },
        TimeSeriesData {
            Open: 811.700012,
            High: 815.25,
            Low: 809.780029,
            Volume: 1129100.0,
            Close: 813.669983,
        },
        TimeSeriesData {
            Open: 828.659973,
            High: 833.450012,
            Low: 828.349976,
            Volume: 1247700.0,
            Close: 831.659973,
        },
        TimeSeriesData {
            Open: 823.02002,
            High: 828.070007,
            Low: 821.655029,
            Volume: 1597800.0,
            Close: 828.070007,
        },
        TimeSeriesData {
            Open: 819.929993,
            High: 824.400024,
            Low: 818.97998,
            Volume: 1281700.0,
            Close: 824.159973,
        },
        TimeSeriesData {
            Open: 819.359985,
            High: 823.0,
            Low: 818.469971,
            Volume: 1304000.0,
            Close: 818.97998,
        },
        TimeSeriesData {
            Open: 819.0,
            High: 823.0,
            Low: 816.0,
            Volume: 1053600.0,
            Close: 820.450012,
        },
        TimeSeriesData {
            Open: 816.0,
            High: 820.958984,
            Low: 815.48999,
            Volume: 1198100.0,
            Close: 819.23999,
        },
        TimeSeriesData {
            Open: 811.700012,
            High: 815.25,
            Low: 809.780029,
            Volume: 1129100.0,
            Close: 813.669983,
        },
        TimeSeriesData {
            Open: 819.359985,
            High: 823.0,
            Low: 818.469971,
            Volume: 1304000.0,
            Close: 818.97998,
        },
        TimeSeriesData {
            Open: 819.0,
            High: 823.0,
            Low: 816.0,
            Volume: 1053600.0,
            Close: 820.450012,
        },
        TimeSeriesData {
            Open: 816.0,
            High: 820.958984,
            Low: 815.48999,
            Volume: 1198100.0,
            Close: 819.23999,
        },
        TimeSeriesData {
            Open: 811.700012,
            High: 815.25,
            Low: 809.780029,
            Volume: 1129100.0,
            Close: 813.669983,
        },
    ];

    infer::<MyAutodiffBackend>(artifact_dir, device, test_data);

    println!("Training completed successfully!");
}
