#include "simple_vo/frame.h"

Frame::Frame()
{
}

Frame::Frame(const Frame &other)
{
    cam_param = other.cam_param;
    frame_pose = other.frame_pose;
    depth = other.depth;
    image = other.image;
}

Frame &Frame::operator=(Frame other)
{
    if (this != &other)
    {
        swap(*this, other);
    }
}

void swap(Frame &lhs, Frame &rhs)
{
    if (&lhs != &rhs)
    {
        using std::swap;
        swap(lhs.cam_param, rhs.cam_param);
        swap(lhs.frame_pose, rhs.frame_pose);
        swap(lhs.image, rhs.image);
        swap(lhs.depth, rhs.depth);
    }
}

void Frame::copy_to(Frame &other)
{
    other.frame_pose = frame_pose;
    other.cam_param = cam_param;
    image.copyTo(other.image);
    depth.copyTo(other.depth);
}

void Frame::clear_pose()
{
    frame_pose.setIdentity();
}