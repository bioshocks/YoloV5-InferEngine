#pragma once

#ifdef MYDLL_EXPORTS
#define INFERENGINE_API __declspec(dllexport)
#else
#define INFERENGINE_API __declspec(dllimport)
#endif
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
INFERENGINE_API class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity != nvinfer1::ILogger::Severity::kINFO)
            std::cout << msg << std::endl;
    }
};
 
INFERENGINE_API struct Result {
    int label = 0;
    float confidence = 0;
    float bbox[4];
};
 
INFERENGINE_API class YOLOv5Inference {
public:
    INFERENGINE_API YOLOv5Inference(const std::string& engine_name);
    INFERENGINE_API ~YOLOv5Inference();
    INFERENGINE_API std::vector<Result> infer(cv::Mat& src,float iou_thresh,float class_thresh,float NMS_thresh);
 
private:
    std::vector<char> readFile(const std::string& filepath);
    size_t calculateVolume(const nvinfer1::Dims& dims);
    std::vector<float> preprocessImage(cv::Mat&, const nvinfer1::Dims& inputDims, int& x_offset, int& y_offset, float& ratio);
    float Area(float* bbox);
    float InterSection(float* bbox1, float* bbox2);
    float IOU(float* bbox1, float* bbox2, int ref);
    std::vector<Result> NMS(std::vector<Result>& proposals, float NMS_thresh);
    std::vector<double> softmax(std::vector<float> input);
    std::vector<Result> PostProcess(std::vector<Result>& proposals, int imgHeight, int imgWidth,int xoffset,int yoffset, float ratio);
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    cudaStream_t cudastream;
    void* input_mem = nullptr;
    void* output_mem = nullptr;
    nvinfer1::Dims input_dim;
    nvinfer1::Dims output_dim;
    int num_class;
    size_t input_size;
    size_t output_size;
    int image_size;
};