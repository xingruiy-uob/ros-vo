#ifndef FRAME_H
#define FRAME_H

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

class Frame
{
    Eigen::Affine3d frame_pose;
    Eigen::Matrix3d cam_param;

public:
    Frame();
    Frame(const Frame &);
    Frame(Frame &&);
    Frame &operator=(Frame);
    Frame &operator=(Frame &&);
    friend void swap(Frame &, Frame &);
    void copy_to(Frame &);
    void clear_pose();

    Eigen::Affine3d inline get_pose() const
    {
        return frame_pose;
    }

    void inline set_pose(const Eigen::Affine3d &pose)
    {
        frame_pose = pose;
    }

    cv::Mat depth;
    cv::Mat image;
};

#endif