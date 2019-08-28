cmake_minimum_required(VERSION 2.8)
project(cuda_icp LANGUAGES CXX CUDA)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(catkin 
REQUIRED
    roscpp
    cuda_ros
)

catkin_package(
    INCLUDE_DIRS 
        include
    LIBRARIES 
        cuda_icp
    # CATKIN_DEPENDS 
    # DEPENDS 
)

add_library(cuda_icp STATIC)

target_sources(cuda_icp
PRIVATE
    src/icp_tracker.cpp
    src/pyr_down.cu
    src/se3_estimator.cu
    src/se3_experiment.cu
)

target_compile_features(cuda_icp
PUBLIC
    cxx_std_11
)

target_include_directories(cuda_icp
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(cuda_icp
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
)

add_subdirectory(example)