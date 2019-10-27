cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)

project(PCLKeypoint)

add_subdirectory(pybind11)
find_package(Eigen3)
find_package(PCL 1.8)

set(MY_HEADERS
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    src/utils.hpp
    src/keypoints.hpp
)

INCLUDE_DIRECTORIES(
    ${MY_HEADERS}
)

#add_executable(test_pcl_keypoint
#    src/main.cpp
#    src/keypoints.cpp
#    ${MY_HEADERS}
#)

#target_link_libraries(test_pcl_keypoint
#    ${PCL_LIBRARIES}
#)

pybind11_add_module(${PROJECT_NAME}
    src/keypoints.cpp
    ${MY_HEADERS}
)
target_link_libraries(${PROJECT_NAME}
    ${PCL_LIBRARIES}
)
