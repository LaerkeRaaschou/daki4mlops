#include "preprocess.hpp"

#include <stdexcept>
#include <opencv2/opencv.hpp>

namespace {
constexpr int W = 64;
constexpr int H = 64;
constexpr int C = 3;
}

std::vector<float> preprocess_image_to_chw_3x64x64(const std::string& image_path) {
    // 1) Load (OpenCV loads as BGR by default)
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    std::cout << "Loaded image: " << bgr.cols << "x" << bgr.rows << "\n";

    // 2) Resize to 64x64
    cv::Mat bgr_resized;
    cv::resize(bgr, bgr_resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

    // 3) Convert BGR -> RGB (common convention for PyTorch-trained models)
    cv::Mat rgb;
    cv::cvtColor(bgr_resized, rgb, cv::COLOR_BGR2RGB);

    // 4) Convert uint8 [0..255] -> float32 [0..1]
    cv::Mat rgb_f32;
    rgb.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);

    // 5) Convert HWC -> CHW in a flat vector
    std::vector<float> chw(static_cast<size_t>(C * H * W));

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            cv::Vec3f pix = rgb_f32.at<cv::Vec3f>(y, x); // [R,G,B] floats

            // CHW indexing:
            // channel-major, then row-major inside each channel
            const size_t idx = static_cast<size_t>(y * W + x);
            
            // ImageNet normalization (must match how the model was trained)
            constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
            constexpr float stdv[3] = {0.229f, 0.224f, 0.225f};

            chw[0 * H * W + idx] = (pix[0] - mean[0]) / stdv[0]; // R
            chw[1 * H * W + idx] = (pix[1] - mean[1]) / stdv[1]; // G
            chw[2 * H * W + idx] = (pix[2] - mean[2]) / stdv[2]; // B

        }
    }

    return chw;
}
