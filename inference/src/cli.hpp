#pragma once

#include <string>

struct CliOptions {
    std::string image_path;
};

bool parse_cli(int argc, char** argv, CliOptions& out);
