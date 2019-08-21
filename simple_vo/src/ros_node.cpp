#include <ros/ros.h>
#include "simple_vo/simple_vo.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "simple_vo");

    SimpleVO vo(640, 480);

    ros::spin();
    return 0;
}