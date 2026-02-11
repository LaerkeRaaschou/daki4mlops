#include "ort_classifier.hpp"

#include <stdexcept>

OrtClassifier::OrtClassifier(const std::wstring& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "resnet18_infer"),
      session_options_(),
      session_(nullptr)
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
        throw std::runtime_error("This example expects exactly 1 input and 1 output.");
    }

    // Cache names (nice for transparency/debugging)
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = session_.GetInputNameAllocated(0, allocator);
    auto out_name = session_.GetOutputNameAllocated(0, allocator);

    input_name_ = in_name.get();
    output_name_ = out_name.get();
}

std::vector<float> OrtClassifier::run_one(const std::vector<float>& chw_3x64x64) {
    if (chw_3x64x64.size() != INPUT_ELEMS) {
        throw std::runtime_error("Input must be exactly 3*64*64 floats (CHW).");
    }

    // Hardcoded runtime shape: 1x3x64x64
    const int64_t input_shape[4] = {1, C, H, W};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    // ORT wraps your buffer (no copy).
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(chw_3x64x64.data()),
        chw_3x64x64.size(),
        input_shape,
        4
    );

    const char* input_names[]  = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    Ort::Value& out0 = output_tensors[0];
    float* logits_ptr = out0.GetTensorMutableData<float>();

    // We *expect* 200 logits. If you want to be strict, you can validate once here.
    // For beginner clarity, we just read the output shape and copy that many.
    auto out_shape = out0.GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 2 || out_shape[0] != 1) {
        throw std::runtime_error("Unexpected output shape. Expected [1, num_classes].");
    }

    const size_t num_classes = static_cast<size_t>(out_shape[1]);
    std::vector<float> logits(num_classes);
    for (size_t i = 0; i < num_classes; ++i) {
        logits[i] = logits_ptr[i];
    }
    return logits;
}