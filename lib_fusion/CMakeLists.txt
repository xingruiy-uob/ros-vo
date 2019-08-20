cmake_minimum_required(VERSION 2.8)
project(lib_fusion CXX CUDA)

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
        lib_fusion
    # CATKIN_DEPENDS 
    # DEPENDS 
)

add_library(lib_fusion STATIC)

target_sources(lib_fusion
PRIVATE
    src/voxel_mapper.cpp
)

target_compile_features(lib_fusion
PUBLIC
    cxx_std_11
)

target_include_directories(lib_fusion
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(lib_fusion
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
)
