#ifndef YOLOV8_SEG_H
#define YOLOV8_SEG_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "net.h"

namespace yolov8personal
{
    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
        cv::Mat mask;
        std::vector<float> mask_feat;
    };
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };


    class Yolo
    {
        public:
            // 构造函数
            Yolo();
            // 析构函数
            ~Yolo();

            // 加载模型
            bool loadmodel(char* modelpath, char* parampath, bool use_gpu);

            // 执行推理
            bool inference(cv::Mat& img, std::vector<Object>& objects);

            // 结果绘制
            void drawresult(cv::Mat& img, std::vector<Object>& objects);


        private:
            // 网络对象
            ncnn::Net yolonet;

            // 推理结果
            std::vector<Object> resultobjects;
        
    };
}

#endif // YOLOV8_SEG_H