cmake_minimum_required(VERSION 3.10)

project(pcl_write)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 11)
find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (${CMAKE_PROJECT_NAME} main.cpp)
target_link_libraries (${CMAKE_PROJECT_NAME} ${PCL_LIBRARIES})
