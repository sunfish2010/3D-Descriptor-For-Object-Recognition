
set(SOURCE_FILES
        "sample.cu"
        "sample.h"
        "grid.cu"
        "grid.h"
        "shot.cu"
        "shot.h"
        "shot_lrf.cu"
        "shot_lrf.h"
        "search.cu"
        "search.h"
        )

cuda_add_library(src
        ${SOURCE_FILES}
        OPTIONS -arch=sm_50 -std=c++11
        )