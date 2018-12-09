#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/system_error.h>

// check cuda error
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) _checkCUDAErrorHelper(msg, FILENAME, __LINE__)

static
void _checkCUDAErrorHelper(const char *msg, const char *filename, int line) {
#if !defined(NDEBUG)
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (filename) {
        fprintf(stderr, " (%s:%d)", filename, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

// for random number generation
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}