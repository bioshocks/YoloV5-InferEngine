#include "InferEngineClass.h"

 YOLOv5Inference::YOLOv5Inference(const std::string& engine_name) {
    Logger logger;
    mRuntime.reset(nvinfer1::createInferRuntime(logger));
    std::vector<char> engineData = readFile(engine_name);
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
    mContext.reset(mEngine->createExecutionContext());
    

    const char* input_name = mEngine->getIOTensorName(0);
    const char* output_name = mEngine->getIOTensorName(1);

    input_dim = mEngine->getTensorShape(input_name);
    output_dim = mEngine->getTensorShape(output_name);

    num_class = output_dim.d[2] - 5;
    image_size = input_dim.d[2];
    input_size = calculateVolume(input_dim);
    output_size = calculateVolume(output_dim);

    cudaStreamCreate(&cudastream);
    cudaMallocAsync(&input_mem, input_size * sizeof(float), cudastream);
    cudaMallocAsync(&output_mem, output_size * sizeof(float), cudastream);
}

 YOLOv5Inference::~YOLOv5Inference() {
    cudaFree(input_mem);
    cudaFree(output_mem);
    cudaStreamDestroy(cudastream);
}
size_t  YOLOv5Inference::calculateVolume(const nvinfer1::Dims& dims) {
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        volume *= dims.d[i];
    }
    return volume;
}


INFERENGINE_API std::vector<Result> YOLOv5Inference::infer(cv::Mat& src,float iou_thresh,float class_thresh,float NMS_thresh){
    int x_offset = 0;
    int y_offset = 0;
    float ratio = 0;
    std::vector<float> inputData = preprocessImage(src, input_dim, x_offset, y_offset, ratio);
    cudaMemcpyAsync(input_mem, inputData.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, cudastream);

    void* binding[] = { input_mem, output_mem };
    mContext->executeV2(binding);

    std::vector<float> result_mem(output_size);
    cudaMemcpyAsync(result_mem.data(), output_mem, output_size * sizeof(float), cudaMemcpyDeviceToHost, cudastream);
    cudaStreamSynchronize(cudastream);

    auto ptr = result_mem.begin();
    std::vector<Result> results;
    int num_bboxes = output_size / (5 + num_class);

    for (int i = 0; i < num_bboxes; ++i) {
        const float objectness = ptr[4];
        if (objectness > iou_thresh) {
            const int label = std::max_element(ptr + 5, ptr + 5 + num_class) - (ptr + 5);
            const float confidence = ptr[5 + label] * objectness;
            if (confidence >= class_thresh) {
                Result result;
                result.bbox[0] = ptr[0] - ptr[2] * 0.5f;
                result.bbox[1] = ptr[1] - ptr[3] * 0.5f;
                result.bbox[2] = ptr[0] + ptr[2] * 0.5f;
                result.bbox[3] = ptr[1] + ptr[3] * 0.5f;
                result.confidence = confidence;
                result.label = label;
                results.emplace_back(result);
            }
        }
        ptr += (5 + num_class);
    }

    
    results = NMS(results, NMS_thresh);
    return PostProcess(results,src.rows,src.cols,x_offset,y_offset,ratio);
}
std::vector<char> YOLOv5Inference::readFile(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        
        throw"Failed to open file model!";
        abort();
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

std::vector<float> YOLOv5Inference::preprocessImage(cv::Mat& src, const nvinfer1::Dims& inputDims, int& x_offset, int& y_offset, float& ratio) {
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    const int model_width = image_size;
    const int model_height = image_size;
    cv::Mat resizedImg;
    ratio = std::min(model_width / (src.cols * 1.0f),
        model_height / (src.rows * 1.0f));
    // 等比例缩放
    const int border_width = std::round(src.cols * ratio);
    const int border_height = std::round(src.rows * ratio);
    // 计算偏移值
    x_offset = (model_width - border_width) / 2;
    y_offset = (model_height - border_height) / 2;
    cv::resize(src, resizedImg, cv::Size(border_width, border_height));
    cv::copyMakeBorder(resizedImg, resizedImg, y_offset, y_offset, x_offset,
        x_offset, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    resizedImg.convertTo(resizedImg, CV_32F, 1.0 / 255.0);
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB);

    std::vector<float> input_blob(model_height * model_width * 3);
    const int channels = resizedImg.channels();
    const int width = resizedImg.cols;
    const int height = resizedImg.rows;
    cv::Mat chw[3];
    cv::split(resizedImg, chw); 
    for (int i = 0; i < 3; i++) {
        memcpy(input_blob.data() + i * width * height, chw[i].data, width * height * sizeof(float));
    }
    return input_blob;
}
float YOLOv5Inference::Area(float* bbox)
{
    float width = bbox[2] - bbox[0];
    float height = bbox[3] - bbox[1];
    if (width <= 0 || height <= 0)
    {
        return 0;
    }
    return width * height;
}

float YOLOv5Inference::InterSection(float* bbox1, float* bbox2)
{
    float bbox[4];
    bbox[0] = (std::max)(bbox1[0], bbox2[0]);
    bbox[1] = (std::max)(bbox1[1], bbox2[1]);
    bbox[2] = (std::min)(bbox1[2], bbox2[2]);
    bbox[3] = (std::min)(bbox1[3], bbox2[3]);
    return Area(bbox);
}

float YOLOv5Inference::IOU(float* bbox1, float* bbox2, int ref)
{
    float area1 = Area(bbox1);
    float area2 = Area(bbox2);
    float inter_section_area = InterSection(bbox1, bbox2);
    float area = area1 + area2 - inter_section_area;
    switch (ref)
    {
    case 1:
        area = area1;
        break;
    case 2:
        area = area2;
        break;
    case 3:
        area = (std::min)(area1, area2);
    default:
        break;
    }
    return inter_section_area / area;
}
static bool PredCompare(Result p1, Result p2)
{
    return p1.confidence > p2.confidence;
}
std::vector<Result> YOLOv5Inference::NMS(std::vector<Result>& proposals, float NMS_thresh)
{
    std::stable_sort(proposals.begin(), proposals.end(), PredCompare);

    std::vector<int> indices;
    for (int i = 0; i < proposals.size(); ++i)
    {
        Result& p = proposals[i];
        bool keep = 1;
        for (std::vector<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter)
        {
            Result& r_p = proposals[*iter];
            if (keep)
            {
                if (IOU(p.bbox, r_p.bbox, 0) > NMS_thresh)
                {
                    keep = false;
                }
            }
            else
            {
                break;
            }
        }
        if (keep)
        {
            indices.push_back(i);
        }
    }
    std::vector<Result> result;
    for (int i = 0; i < indices.size(); ++i)
    {
        result.push_back(proposals[indices[i]]);
    }
    return result;
}
std::vector<Result> YOLOv5Inference::PostProcess(
    std::vector<Result>& proposals,
    int imgHeight,
    int imgWidth,
    int xoffset,
    int yoffset,
    float ratio)
{
    std::vector<Result> output;
    if (ratio <= 1e-6f)
        return output;

    for (auto& p : proposals)
    {
        Result r = p;


        float x1 = (r.bbox[0] - xoffset) / ratio;
        float y1 = (r.bbox[1] - yoffset) / ratio;
        float x2 = (r.bbox[2] - xoffset) / ratio;
        float y2 = (r.bbox[3] - yoffset) / ratio;


        r.bbox[0] = std::clamp(std::min(x1, x2), 0.f, (float)(imgWidth - 1));
        r.bbox[1] = std::clamp(std::min(y1, y2), 0.f, (float)(imgHeight - 1));
        r.bbox[2] = std::clamp(std::max(x1, x2), 0.f, (float)(imgWidth - 1));
        r.bbox[3] = std::clamp(std::max(y1, y2), 0.f, (float)(imgHeight - 1));

        if (r.bbox[2] > r.bbox[0] + 1 && r.bbox[3] > r.bbox[1] + 1)
        {
            output.push_back(r);
        }
    }

    return output;
}
