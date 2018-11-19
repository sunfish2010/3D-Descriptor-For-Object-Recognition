
set(SOURCE_FILES
        "detection.cu"
        "detection.h"
        "kdtree.cpp"
        "kdtree.h"
        )

cuda_add_library(src
        ${SOURCE_FILES}
        OPTIONS -arch=sm_20
        )