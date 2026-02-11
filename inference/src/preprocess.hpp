#pragma once

#include <string>
#include <vector>

// Loads an image from disk and returns a CHW float vector:
// shape = 3 x 64 x 64, values in [0, 1].
// Assumes single image inference.
std::vector<float> preprocess_image_to_chw_3x64x64(const std::string& image_path);
