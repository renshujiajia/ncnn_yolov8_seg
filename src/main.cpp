#include <iostream>
#include "yolov8_seg.h"

using namespace yolov8personal;


using namespace std;

int main(int argc, char* argv[])
{
    cout << "Hello, World!" << endl;
    
    // char* modelfile = "../models/best_sim-sim-opt.bin";
    // char* paramfile = "../models/best_sim-sim-opt.param";

    // char* modelfile = "../models/best-sim-opt-fp16.bin";
    // char* paramfile = "../models/best-sim-opt-fp16.param";
    char* modelfile = "../anothermodel/yolov8n-seg-sim-opt-fp16.bin";
    char* paramfile = "../anothermodel/yolov8n-seg-sim-opt-fp16.param";
    // char* modelfile = "/home/renshujia/AiStudy/deploy/ncnn_ubuntu/ncnn-20240102-ubuntu-2204-shared/example/yolov72/yolov7-mask-ncnn/models/yolov7-mask.bin";
    // char* paramfile = "/home/renshujia/AiStudy/deploy/ncnn_ubuntu/ncnn-20240102-ubuntu-2204-shared/example/yolov72/yolov7-mask-ncnn/models/yolov7-mask.param";
    // char* imagefile = "../images/pingti.png";
    char* imagefile = "../images/horses.jpg";


    // yolov8_seg yolov8;
    // auto result = yolov8.getString();
    // std::cout << result << std::endl;
    Yolo ayolo;


    // 加载模型
    ayolo.loadmodel(modelfile, paramfile, false);
    
    // 执行推理
    cv::Mat img = cv::imread(imagefile, 1);
    // 判断是否读取成功
    if (img.empty())
    {
        std::cout << "read image failed!" << std::endl;
        return -1;
    }
    // 进行推理

    std::vector<Object> objects;
    ayolo.inference(img, objects);
    ayolo.drawresult(img, objects);

    //保存img的result
    cv::imwrite("result.jpg", img);
    return 0;

}