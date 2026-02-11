#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "cli.hpp"
#include "ort_classifier.hpp"
#include "preprocess.hpp"

namespace {
void print_logits_preview(const std::vector<float>& logits, size_t count) {
    std::cout << "First " << count << " logits: ";
    for (size_t i = 0; i < count && i < logits.size(); ++i) {
        std::cout << logits[i] << " ";
    }
    std::cout << "\n";
}

size_t argmax(const std::vector<float>& values) {
    auto it = std::max_element(values.begin(), values.end());
    return static_cast<size_t>(std::distance(values.begin(), it));
}
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "Starting ORT inference (single image)...\n";

        // Minimal flow: load model -> preprocess image -> run -> print top-1
        const std::wstring model_path = L"..\\..\\onnx\\resnet18_tinyimagenet_64.onnx";
        OrtClassifier clf(model_path);

        std::cout << "Model input name : " << clf.input_name() << "\n";
        std::cout << "Model output name: " << clf.output_name() << "\n";

        CliOptions options;
        if (!parse_cli(argc, argv, options)) {
            return 1;
        }

        // Preprocess: load -> resize -> RGB -> float32 -> normalize -> CHW
        std::vector<float> input_data = preprocess_image_to_chw_3x64x64(options.image_path);

        // Time the inference call only
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<float> logits = clf.run_one(input_data);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

        std::cout << "Output shape: 1 " << logits.size() << "\n";
        print_logits_preview(logits, 5);

        size_t top1_idx = argmax(logits);
        std::cout << "Top-1 class index: " << top1_idx << "\n";
        std::cout << "Top-1 logit: " << logits[top1_idx] << "\n";


        std::cout << "Inference time: "
                  << duration.count()
                  << " microseconds ("
                  << duration.count() / 1000.0
                  << " ms)\n";

        std::cout << "Done.\n";
        return 0;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "std::exception: " << e.what() << "\n";
        return 1;
    }
}
