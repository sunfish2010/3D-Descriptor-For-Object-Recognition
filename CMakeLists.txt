cmake_minimum_required(VERSION 3.10)

project(object_detection C CXX)

#set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CONFIGURATION_TYPES Debug;RelWithDebInfo;Release
        CACHE STRING "Set configuration types" FORCE)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS_DEBUG                 "-O0 -g ")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO        "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE               "-O3    -DNDEBUG")

SET(BOOST_INCLUDEDIR "/usr/include")
SET(BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu")
SET (BOOST_MIN_VERSION "1.55.0")
set (Boost_NO_BOOST_CMAKE ON)
set(Boost_DEBUG ON)

find_package(Threads REQUIRED)


list(APPEND CUDA_NVCC_FLAGS_DEBUG          -O0 -g -G)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO -O2 -g -lineinfo)
list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3 -DNDEBUG)

list(APPEND CMAKE_EXE_LINKER_FLAGS "-lX11 -lXxf86vm -lXi")

find_package(PCL 1.9 REQUIRED)

find_package(CUDA 8.0 REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
        list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(util)
add_subdirectory(src)



cuda_add_executable (${CMAKE_PROJECT_NAME} main.cpp main.hpp)
#add_executable(${CMAKE_PROJECT_NAME} test.cpp)
target_link_libraries (${CMAKE_PROJECT_NAME}
        ${PCL_LIBRARIES}
        util
        src
        )
