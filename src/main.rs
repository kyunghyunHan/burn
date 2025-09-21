#![recursion_limit = "256"] // ✅ 크레이트 전체에 적용

mod lstm;
// mod transformer;
fn main() {
    // lstm::example();
    // lstm::infer_example();
    lstm::basic::evaluate();
}
