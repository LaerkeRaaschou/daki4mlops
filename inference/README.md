This project is a minimal, beginner-friendly C++ inference example using ONNX Runtime.
It is intentionally small and explicit so you can see the entire flow end-to-end.

Goal
- Load a ResNet-18 ONNX model
- Preprocess one image (resize, normalize)
- Run inference and print top-1 prediction

Why C++ for inference
- Lower latency and smaller deployable footprint (no Python runtime needed)
- Easier integration with production services and embedded systems
- Predictable performance and memory behavior

Model expectations
- Input: 1x3x64x64 float32 (NCHW)
- Output: 1x200 float32 logits
- Preprocessing matches ImageNet-style normalization

Requirements
- CMake 3.20+
- A C++17 compiler (MSVC on Windows is fine)
- OpenCV (C++ libraries)
- ONNX Runtime (CPU build)

Build (Windows)
1) Download ONNX Runtime (CPU) and unzip it. Example folder: C:\onnxruntime-win-x64-1.16.3
2) Download OpenCV for Windows (the prebuilt package) and unzip it. Example folder: C:\opencv
3) Find the OpenCV CMake folder. It is usually:
	C:\opencv\build\x64\vc17\lib
4) Configure with CMake (set both ONNX Runtime and OpenCV):
	cmake -S . -B build -DONNXRUNTIME_DIR=C:\onnxruntime-win-x64-1.16.3 -DOpenCV_DIR=C:\opencv\build\x64\vc17\lib
5) Build:
	cmake --build build --config Release

Run
1) Put a test image somewhere in the repo (for example: inference/images/test.jpeg)
2) Run the executable with the image path:
	.\build\Release\resnet18_infer.exe ..\..\images\test.jpeg

If your model or image paths are different, edit them in src/main.cpp.

Files and their purpose
- src/main.cpp: Minimal pipeline (load model, preprocess, run, print results)
- src/cli.hpp/cpp: Tiny CLI parser for the image path
- src/ort_classifier.hpp/cpp: ONNX Runtime wrapper for a single-image classifier
- src/preprocess.hpp/cpp: OpenCV-based preprocessing (resize, RGB, normalize)

Notes
- The preprocessing uses ImageNet mean/std; this must match how the model was trained.
- Paths are intentionally hardcoded for clarity in a first project.
