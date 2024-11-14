#include "InferEngineClass.h"
#include <chrono>
int main()
{
	auto infer_engine = new YOLOv5Inference("svd-yolo");
	cv::Mat src = cv::imread("multi.jpg");
	std::vector<Result> results;
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < 100; i++)
	{
		results = infer_engine->infer(src, 0.1, 0.1, 0.4);

	}
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "»¨·ÑÁË"
		<< double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
		<< "Ãë" <<std:: endl;

	/*for (auto result : results)
	{
		
		cv::rectangle(src, cv::Point(result.bbox[0], result.bbox[1]), cv::Point(result.bbox[2], result.bbox[3]), cv::Scalar(0, 0, 255), 1);
	}
	cv::imshow("Rectangle", src);
	cv::waitKey(0);*/
	delete infer_engine;
	return 0;
}