cmake_minimum_required(VERSION 3.20)
project(Music2EmoNative VERSION 0.1.0 LANGUAGES CXX)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime C/C++ SDK")
find_path(ONNX_INCLUDE onnxruntime_cxx_api.h HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
find_library(ONNX_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)

add_executable(revelation-music2emo src/main.cpp src/audio_features.cpp)
target_compile_features(revelation-music2emo PRIVATE cxx_std_17)
target_include_directories(revelation-music2emo PRIVATE "${ONNX_INCLUDE}")
target_link_libraries(revelation-music2emo PRIVATE "${ONNX_LIBRARY}")
if(MSVC)
  target_compile_options(revelation-music2emo PRIVATE /EHsc /utf-8)
endif()
if(WIN32)
  find_file(ONNX_DLL onnxruntime.dll HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/bin" REQUIRED)
  add_custom_command(TARGET revelation-music2emo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNX_DLL}" "$<TARGET_FILE_DIR:revelation-music2emo>")
endif()
if(REVELATION_EDITOR_BIN)
  add_custom_command(TARGET revelation-music2emo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:revelation-music2emo>" "${REVELATION_EDITOR_BIN}")
endif()
