
set(SOURCE_FILES
        "detection.cu"
        "detection.h"
        "sample.cu"
        "sample.h"
        "kdtree.cpp"
        "kdtree.hpp"
        )

cuda_add_library(src
        ${SOURCE_FILES}
        OPTIONS -arch=sm_20
        )