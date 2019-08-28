add_executable(test_icp)

target_sources(test_icp
PRIVATE
    main.cpp
)

target_link_libraries(test_icp
PRIVATE
    cuda_icp
)