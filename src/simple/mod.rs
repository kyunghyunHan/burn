use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::data::dataset::InMemDataset;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::Relu;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::Config;
use burn::prelude::Int;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::serde::de::value;
use burn::tensor::activation::relu;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Float;
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::{Data, Device};
use burn::train::metric::AccuracyMetric;
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use csv::ReaderBuilder;
use serde::Deserialize;
use std::path::Path;
// 간단한 데이터셋 구조체
#[derive(Clone, Debug, Deserialize)]
struct SimpleDataset {
    data: [i64; 2],
    labels: i64,
}
pub struct DiabetesDataset {
    dataset: InMemDataset<SimpleDataset>,
}
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize, //
    hidden_size: usize, //
    #[config(default = "0.5")]
    dropout: f64, //dropout
}
impl Dataset<SimpleDataset> for DiabetesDataset {
    // type Item = (f32, i32);
    fn get(&self, index: usize) -> Option<SimpleDataset> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
impl DiabetesDataset {
    pub fn new() -> Result<Self, std::io::Error> {
        // Download dataset csv file
        let path = Path::new(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("../dataset/test.csv");

        // Build dataset from csv with tab ('\t') delimiter
        let dataset = InMemDataset::from_csv(path, &ReaderBuilder::new()).unwrap();
        let dataset = Self { dataset };
        Ok(dataset)
    }
}
// 배치 구조체
#[derive(Clone, Debug)]
struct SimpleBatch<B: Backend> {
    values: Tensor<B, 1>,
    targets: Tensor<B, 1, Int>,
}

// 배처 구조체
#[derive(Clone)] // Clone trait 추가

struct SimpleBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SimpleBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<SimpleDataset, SimpleBatch<B>> for SimpleBatcher<B> {
    fn batch(&self, items: Vec<SimpleDataset>) -> SimpleBatch<B> {
        let values = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([(item.data)], &self.device))
            .collect();
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints([(item.labels)], &self.device))
            .collect();

        let values = Tensor::cat(values, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        SimpleBatch { values, targets }
    }
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimpleModel<B> {
        SimpleModel {
            //커널 크기 3사용
            //채널 1에서 8로 확장
            // conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
            // //8에서 16으로 확장
            // conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
            //적응형 평균 폴링 모듈을 사용 이미지의 차원을 8x8으로 축소
            // pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            // activation: ReLU::new(),
            linear1: LinearConfig::new(2, 4).init(device),
            linear2: LinearConfig::new(4, 2).init(device),
            linear3: LinearConfig::new(2, 2).init(device),
            // dropout: DropoutConfig::new(self.dropout).init(),
            // activation: Relu::new(),
        }
    }
}
#[derive(burn::module::Module, Debug)]
pub struct SimpleModel<B: Backend> {
    // dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>, // 출력이 2개가 되도록 수정 필요
    linear3: Linear<B>, // 출력이 2개가 되도록 수정 필요

                        // activation: Relu,
}
impl<B: Backend> SimpleModel<B> {
    // fn init(device: &B::Device) -> Self {
    //     Self {
    //         // dropout: DropoutConfig::new(0.1).init(),
    //         linear1: LinearConfig::new(1, 4).init(device), // 입력 1, 은닉층 4
    //         linear2: LinearConfig::new(4, 2).init(device), // 은닉층 4, 출력 2 (이진분류)
    //         linear3: LinearConfig::new(4, 2).init(device), // 은닉층 4, 출력 2 (이진분류)

    //                                                        // activation: Relu::new(),
    //     }
    // }

    pub fn forward(&self, images: Tensor<B, 1>) -> Tensor<B, 2> {
        let [batch_size] = images.dims();
        let xs: Tensor<B, 2> = images.reshape([batch_size, 1]);
        let xs = self.linear1.forward(xs);
        let xs = relu(xs);
        let xs = self.linear2.forward(xs);
        let xs = relu(xs);
        self.linear3.forward(xs) // [batch_size, 2]
    }
    fn forward_classification(
        &self,
        x: Tensor<B, 1>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(x);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}
#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
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

impl<B: AutodiffBackend> TrainStep<SimpleBatch<B>, ClassificationOutput<B>> for SimpleModel<B> {
    fn step(&self, batch: SimpleBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.values, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<SimpleBatch<B>, ClassificationOutput<B>> for SimpleModel<B> {
    fn step(&self, batch: SimpleBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.values, batch.targets)
    }
}
fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);
    let batcher_train = SimpleBatcher::<B>::new(device.clone());
    let batcher_valid = SimpleBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DiabetesDataset::new().unwrap());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DiabetesDataset::new().unwrap());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: SimpleDataset) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.labels;
    let batcher = SimpleBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.values);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}
pub fn run() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    train::<MyAutodiffBackend>(
        "./models/simple",
        TrainingConfig::new(ModelConfig::new(10, 1), AdamConfig::new()),
        device.clone(),
    );
    // infer::<MyBackend>(
    //     "./models/simple",
    //     device,
    //     SimpleDataset {
    //         data: [13, 9],
    //         labels: 0,
    //     },
    // );
    // 여기서 실제 학습 로직을 구현할 수 있습니다
    println!("Model created successfully!");
}
