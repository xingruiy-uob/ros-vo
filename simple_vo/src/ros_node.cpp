#include <ros/ros.h>
#include "simple_vo/simple_vo.h"

int main(int argc, char **argv)
{
    bool enable_debug_output = false;

    if (argc >= 2)
    {
        if (strcpy(argv[1], "debug"))
        {
            enable_debug_output = true;
        }
    }

    ros::init(argc, argv, "simple_vo");

    SimpleVO vo(
        640, 480,
        enable_debug_output,
        "/camera/depth/image",
        "/camera/rgb/image_color");

    ros::spin();
    return 0;
}