//
//  Created by Tyson YU on 3/29/2019
//

#ifndef BF_H
#define BF_H

#include "common_include.h"
#include "data_loader.h"

class BF
{
public:
    typedef std::shared_ptr<BF> Ptr;
    Calibration::Ptr calibration_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud_;
    double max_depth_ = 0;
    cv::Mat raw_image_;         //相机的彩色RGB图像
    cv::Mat ground_truth_;      //基准深度图
    cv::Mat raw_image_gray_;    //相机图像的灰度图
    cv::Mat raw_projection_image_;  //激光雷达点云投影到相机图像上
    cv::Mat mask_;  //对每一个像素处理时候的掩膜
    cv::Mat depth_image_;
    cv::Mat result_depth_image_;    //结果
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud_;
    struct point_
    {
        int x;
        int y;
        double depth;
        double distance;
        bool iscore = false;
    };
public:
    BF(){};
    BF(Calibration::Ptr Calibration, pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud, cv::Mat Image):calibration_(Calibration), raw_cloud_(Cloud), raw_image_(Image){}
    void BFProcess();
    void Projection();
    float LocalBF(cv::Mat mask);
    void Evaluation();
    void Image2cloud();
};






#endif //BF_H