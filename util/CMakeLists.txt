set(SOURCE_FILES
        "utilityCore.cpp"
        "utilityCore.hpp"
        )

cuda_add_library(util
        ${SOURCE_FILES}
        OPTIONS -arch=sm_50 -std=c++11
        )