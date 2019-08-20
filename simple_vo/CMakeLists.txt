cmake_minimum_required(VERSION 2.8)
project(simple_vo CXX CUDA)

find_package(OpenCV 3.4 REQUIRED)

find_package(catkin 
REQUIRED COMPONENTS
    roscpp 
    cuda_icp
    cuda_ros
    sensor_msgs 
    geometry_msgs
    message_filters
    tf2_ros
)

catkin_package(
    CATKIN_DEPENDS 
        cuda_icp
        cuda_ros
        sensor_msgs
        geometry_msgs
        message_filters
        tf2_ros
)

add_executable(simple_vo)

target_sources(simple_vo
PRIVATE
    src/frame.cpp
    src/ros_node.cpp
    src/simple_vo.cpp
)

target_include_directories(simple_vo
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(simple_vo
PUBLIC
    cxx_std_11
)

target_link_libraries(simple_vo
PUBLIC
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)