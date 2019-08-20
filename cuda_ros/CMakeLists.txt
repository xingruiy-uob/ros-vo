cmake_minimum_required(VERSION 2.8)
project(cuda_ros CXX CUDA)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

find_package(catkin 
REQUIRED
)

catkin_package(
    INCLUDE_DIRS 
        include
    LIBRARIES 
        cuda_ros
    # CATKIN_DEPENDS 
    # DEPENDS 
)

add_library(cuda_ros STATIC)

target_sources(cuda_ros
PRIVATE
    src/debug_image.cu
)

target_compile_features(cuda_ros
PUBLIC
    cxx_std_11
)

target_include_directories(cuda_ros
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)

target_link_libraries(cuda_ros
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
)
