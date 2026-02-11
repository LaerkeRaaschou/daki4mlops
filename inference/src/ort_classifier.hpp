#pragma once

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class OrtClassifier {
public:
    // classifier exported to ONNX that expects 1x3x64x64 input and outputs 200 logits.
    explicit OrtClassifier(const std::wstring& model_path);

    // Run inference on ONE image.
    // Input layout: NCHW with N=1 implied, so you pass C*H*W floats = 3*64*64.
    std::vector<float> run_one(const std::vector<float>& chw_3x64x64);

    const std::string& input_name() const { return input_name_; }
    const std::string& output_name() const { return output_name_; }

    static constexpr int64_t C = 3;
    static constexpr int64_t H = 64;
    static constexpr int64_t W = 64;
    static constexpr size_t INPUT_ELEMS = static_cast<size_t>(C * H * W);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;

    std::string input_name_;
    std::string output_name_;
};
