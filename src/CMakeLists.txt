
set(SOURCE_FILES
        "detection.cu"
        "detection.h"
        "sample.cu"
        "sample.h"
        "shot.cu"
        "shot.h"
        "kdtree.cpp"
        "kdtree.hpp"
        )

cuda_add_library(src
        ${SOURCE_FILES}
        OPTIONS -arch=sm_50 -std=c++11
        )