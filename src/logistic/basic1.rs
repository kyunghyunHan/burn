use burn::backend::autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::data::dataloader::{batcher::Batcher, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::nn::{
    loss::{MseLoss, Reduction},
    Linear, LinearConfig, Lstm, LstmConfig,
};
use burn::tensor::Int; // 꼭 추가!

pub fn example(){
    println!("logistic");

}