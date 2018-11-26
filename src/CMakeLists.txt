
set(SOURCE_FILES
        "detection.cu"
        "detection.h"
        "sample.cu"
        "sample.h"
        "shot.cu"
        "shot.h"
        "shot_lrf.cu"
        "shot_lrf.h"
        "search.cu"
        "search.h"
        "kdtree.cpp"
        "kdtree.hpp"
        )

cuda_add_library(src
        ${SOURCE_FILES}
        OPTIONS -arch=sm_50 -std=c++11
        )