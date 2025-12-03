#![recursion_limit = "256"] // ✅ 크레이트 전체에 적용

mod linear;
mod logistic;
mod lstm;
// mod transformer;
fn main() {
    // lstm::example();
    // lstm::infer_example();
    // lstm::basic::evaluate();
    // lstm::basic3::example();
    // linear::basic1::example();
    // logistic::basic1::example();
    linear::basic2::example()
}
