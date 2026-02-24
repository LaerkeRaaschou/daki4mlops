cmake_minimum_required(VERSION 3.20)
project(resnet18_ort_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this to: .../onnxruntime-win-x64-<version>
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime root directory")

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNXRUNTIME_DIR is not set correctly. "
                      "Expected to find include/onnxruntime_cxx_api.h under it.")
endif()

add_executable(resnet18_infer
  src/cli.cpp
  src/main.cpp
  src/ort_classifier.cpp
  src/preprocess.cpp
)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)
target_link_libraries(resnet18_infer PRIVATE ${OpenCV_LIBS})
target_include_directories(resnet18_infer PRIVATE ${OpenCV_INCLUDE_DIRS})

# ---- ONNX Runtime ----
target_include_directories(resnet18_infer PRIVATE
  "${ONNXRUNTIME_DIR}/include"
)

target_link_directories(resnet18_infer PRIVATE
  "${ONNXRUNTIME_DIR}/lib"
)

target_link_libraries(resnet18_infer PRIVATE
  onnxruntime
)

# ---- Copy DLLs next to the exe (Windows convenience) ----
# ORT
file(GLOB ORT_DLLS
  "${ONNXRUNTIME_DIR}/bin/*.dll"
  "${ONNXRUNTIME_DIR}/lib/*.dll"
)
foreach(dll ${ORT_DLLS})
  add_custom_command(TARGET resnet18_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll}"
            "$<TARGET_FILE_DIR:resnet18_infer>"
  )
endforeach()

# OpenCV (only if OpenCV was found with DLLs on Windows; harmless otherwise)
if(WIN32)
  get_filename_component(_opencv_dir "${OpenCV_DIR}" DIRECTORY)
  file(GLOB OPENCV_DLLS
    "${_opencv_dir}/bin/*.dll"
    "${_opencv_dir}/x64/vc*/bin/*.dll"
    "${_opencv_dir}/../bin/*.dll"
  )
  foreach(dll ${OPENCV_DLLS})
    add_custom_command(TARGET resnet18_infer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${dll}"
              "$<TARGET_FILE_DIR:resnet18_infer>"
    )
  endforeach()
endif()
