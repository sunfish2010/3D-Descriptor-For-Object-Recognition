
set(SOURCE_FILES
        "detection.cu"
        "detection.h"
        "kdtree.cpp"
        "kdtree.hpp"
        )

cuda_add_library(src
        ${SOURCE_FILES}
        OPTIONS -arch=sm_20
        )