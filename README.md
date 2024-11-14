# YoloV5-InferEngine
This repository contains a YOLOv5 inference engine implemented in C++ using TensorRT. The engine is designed to perform efficient object detection with YOLOv5 models.

This File will create an InferEngine.dll and InferEngine.lib

## **Enviroment:**
**opencv**:3.2.0

**TensorRT**:10.4.0.26

**CUDA**:12.6



## Features
**Efficient Inference:** Utilizes TensorRT for high-performance inference.

**Preprocessing:** Includes image preprocessing to prepare inputs for the model.

**Postprocessing:** Handles postprocessing steps such as Non-Maximum Suppression (NMS) and bounding box adjustments.

**CUDA Support:** Leverages CUDA for GPU acceleration.

## Installation
Clone the repository:

git clone https://github.com/bioshocks/YOLOV5-InferEngine.git




## Usage
Before you begin, ensure that your trained **yourmodel.pt** model has been exported to the **ONNX** format and converted to a **TensorRT engine file**.

**Install dependencies:** Ensure you have CUDA and TensorRT installed on your system.

**Include**


1.Include Tensorrt\include to the Additional Include Directories , For example, add C:\Program Files\TensorRT-10.4.0.26\include to the Additional Include Directories.

2.Include CUDA\V12.6\include

**Initialization**
Create an instance of the YOLOv5Inference class by providing the path to the TensorRT engine file:

```cpp
#include "InferEngineClass.h"
YOLOv5Inference yolo("path/to/your/engine.trt");
```

**Inference**
Perform inference on an input image:

```cpp
cv::Mat image = cv::imread("path/to/your/image.jpg");
std::vector<Result> results = yolo.infer(image, 0.5, 0.5, 0.5);
```

**Results**
The infer method returns a vector of Result objects, each containing the bounding box coordinates, confidence score, and class label.

```cpp
results = infer_engine->infer(src, 0.1, 0.1, 0.4);
for (auto result : results)
{	
        cv::rectangle(src, cv::Point(result.bbox[0], result.bbox[1]), cv::Point(result.bbox[2], result.bbox[3]), cv::Scalar(0, 0, 255), 1);
}
cv::imshow("Rectangle", src);
cv::waitKey(0);
```
You can see 'example.cpp' for detail usages.

**Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

**License**
This project is licensed under the MIT License. See the LICENSE file for details.

**Acknowledgements**
YOLOv5 by Ultralytics
TensorRT by NVIDIA
