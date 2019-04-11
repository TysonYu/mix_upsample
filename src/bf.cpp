//
//  Created by Tyson YU on 3/29/2019
//

#include "bf.h"
#define INF 300


void BF::Projection()
{
    pcl::transformPointCloud (*raw_cloud_, *raw_cloud_, calibration_->Rt_);//相机坐标系
    //  过滤掉点云中后面的点
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (raw_cloud_);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, 200);//delete all the point that z<0 && z>200
    pass.filter (*raw_cloud_);
    pcl::transformPointCloud (*raw_cloud_, *raw_cloud_, calibration_->intrisic_);//image，Z上未归一化的像素坐标系
    for(int i = 0; i < raw_cloud_->points.size(); i++)
    {
        raw_cloud_->points[i].x = raw_cloud_->points[i].x / raw_cloud_->points[i].z;
        raw_cloud_->points[i].y = raw_cloud_->points[i].y / raw_cloud_->points[i].z;
        if (raw_cloud_->points[i].z > max_depth_)
                max_depth_ = raw_cloud_->points[i].z;
    }
    // 把点云投影到图像上
    cv::Mat M(raw_image_.rows, raw_image_.cols, CV_32F);//把点投影到M上
    cv::MatIterator_<float>Mbegin,Mend;//遍历所有像素，初始化像素值
	for (Mbegin=M.begin<float>(),Mend=M.end<float>();Mbegin!=Mend;++Mbegin)
		*Mbegin=INF;
    for(int i = 0; i < raw_cloud_->points.size(); i++)//把深度值投影到图像M上
    {
        if(raw_cloud_->points[i].x>=0  && raw_cloud_->points[i].x<raw_image_.cols && raw_cloud_->points[i].y>=0 && raw_cloud_->points[i].y<raw_image_.rows)
        {
            if( raw_cloud_->points[i].z < M.at<float>(raw_cloud_->points[i].y,raw_cloud_->points[i].x))
                M.at<float>(raw_cloud_->points[i].y,raw_cloud_->points[i].x) = raw_cloud_->points[i].z;
        }
    }
    raw_projection_image_ = M;
    result_depth_image_ = M.clone();
    
}

float BF::LocalBF(cv::Mat mask)
{
    float sum = INF;
    float result = INF;
    float r0 = INF;
    float W = INF;
    std::vector<point_> v_point;
    for(int i = 0; i < mask.rows; i++)
        for(int j = 0; j < mask.cols; j++)
        {
            if(mask.at<float>(i,j) != INF)
            {
                struct point_ point;
                point.y = i;
                point.x = j;
                point.depth =  mask.at<float>(i,j);
                point.distance = sqrt(pow(i-6, 2)+pow(j-6,2));
                if(point.depth < r0)    r0 = point.depth;
                v_point.emplace_back(point);
            }
        }
    //  --------- DBSCAN ---------------------------------------------------
    // std::vector< std::vector<point_> > Cluster;
    // std::vector<point_> total_core;
    // std::vector<point_> cluster_1;
    // std::vector<point_> cluster_2;
    //  找到所有的核
    // for(int i = 0; i < v_point.size(); i++)
    // {
    //     int nearpoints = 0;
    //     for(int j = 0; j < v_point.size(); j++)
    //     {
    //         if(j != i)
    //         {
    //             if(abs(v_point.at(i).depth - v_point.at(j).depth)/(v_point.at(i).depth + v_point.at(j).depth) < 0.2)   
    //                 nearpoints ++;
    //         }
    //     }
    //     if(nearpoints >= 2)
    //         total_core.emplace_back(v_point.at(i));
    // }
    
    // for(int i = 0; i < total_core.size(); i ++)
    // {
    //     if(abs(total_core.at(0).depth - total_core.at(i).depth) / (total_core.at(0).depth + total_core.at(i).depth) > 0.2)
    //     {
    //         // cout << "is edge" << endl;
    //         return 0;
    //     }
    // }

    //  --------- calculate ------------------------------------------------
    if(v_point.size() > 1)
    {
        sum = 0;
        W = 0;
        for(int i = 0; i < v_point.size(); i++)
        {
            float temp = (v_point.at(i).distance * abs(r0 - v_point.at(i).depth))/((1 + v_point.at(i).distance)*(1 + abs(r0 - v_point.at(i).depth)));
            W = W + temp;
            sum = sum + v_point.at(i).depth * temp;
        }
        result = sum / W;
    }
    return result;
}

void BF::Image2cloud()
{
    // 深度图返回点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr result_cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
    for(int i = 160; i < result_depth_image_.rows - 6; i ++)
    {
        for(int j = 6; j < result_depth_image_.cols - 6; j++)
        {
            if(result_depth_image_.at<float>(i,j) != INF && result_depth_image_.at<float>(i,j) != 0)
            {
                pcl::PointXYZRGB point;
                point.z = result_depth_image_.at<float>(i,j);
                // point.z = 1;
                point.x = j * point.z;
                point.y = i * point.z;
                point.r = raw_image_.at<cv::Vec3b>(i,j)[2];//round() most closed number
                point.g = raw_image_.at<cv::Vec3b>(i,j)[1];
                point.b = raw_image_.at<cv::Vec3b>(i,j)[0];
                result_cloud_rgb->points.emplace_back(point);
            }
        }
    }

    pcl::transformPointCloud (*result_cloud_rgb, *result_cloud_rgb, calibration_->intrisic_.inverse());//image，Z上未归一化的像素坐标系
    result_cloud_ = result_cloud_rgb;
}

void BF::Evaluation()
{
    int count = 0;
    float sum = 0;
    float result = 0;
    for(int i = 160; i < raw_image_.rows - 6; i++)
        for(int j = 6; j < raw_image_.cols - 6; j++)
        {
            if(ground_truth_.at<ushort>(i,j) != 0 && raw_projection_image_.at<float>(i,j) != 0 && raw_projection_image_.at<float>(i,j) != INF)
            {
                sum += abs((float)ground_truth_.at<ushort>(i,j)/256.0 - raw_projection_image_.at<float>(i,j));
                count ++;
            }
        }
    result = sum / count;
    cout << "average different = " << result << endl;
}

void BF::BFProcess()
{
    Projection();

    for(int i = 160; i < raw_image_.rows - 6; i++)
        for(int j = 6; j < raw_image_.cols - 6; j++)
        {
            if(raw_projection_image_.at<float>(i,j) == INF)
            {
                cv::Rect rect(j-6, i-6, 13, 13);  //x、y、width、height
                mask_ = raw_projection_image_(rect);
                result_depth_image_.at<float>(i,j) = LocalBF(mask_);
            }
                
        }
    Image2cloud();
    Evaluation();
    // cv::Mat temp_1(raw_image_.rows, raw_image_.cols, CV_8U);//把点投影到M上
    // for(int i = 0; i < result_depth_image_.rows; i++)
    //     for(int j = 0; j < result_depth_image_.cols; j++)
    //     {
    //         if(result_depth_image_.at<float>(i,j) == INF)
    //             temp_1.at<char>(i,j) = 255;
    //         else
    //             temp_1.at<char>(i,j) =  result_depth_image_.at<float>(i,j)/max_depth_ *255;
    //     }

    // cv::imshow("depthmap_1", temp_1);
    // cv::waitKey(0);
    // cv::destroyWindow("depthmap_1");
}
