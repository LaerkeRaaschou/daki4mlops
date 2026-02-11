#include "cli.hpp"

#include <iostream>

bool parse_cli(int argc, char** argv, CliOptions& out) {
    if (argc != 2) {
        std::cerr << "Usage: resnet18_infer <path_to_image>\n";
        std::cerr << "Example: resnet18_infer ..\\..\\images\\test.jpeg\n";
        return false;
    }

    out.image_path = argv[1];
    return true;
}
