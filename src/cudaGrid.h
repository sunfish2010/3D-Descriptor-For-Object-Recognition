#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"


/*
 * Atomic functions for float
 * https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
 */
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val){
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__ void getMinMax(int N ,const PointType *pts_in, Eigen::Vector4f *min_pt, Eigen::Vector4f  *max_pt){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N){

        PointType pt = pts_in[index] ;
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
            //float pt_x = pt.x, pt_y = pt.y, pt_z = pt.z;
            atomicMin(&(*min_pt)[0], pt.x);
            atomicMin(&(*min_pt)[1], pt.y);
            atomicMin(&(*min_pt)[2], pt.z);
            atomicMax(&(*max_pt)[0], pt.x);
            atomicMax(&(*max_pt)[1], pt.y);
            atomicMax(&(*max_pt)[2], pt.z);
        }
    }
}


__device__ int kernComputeIndices(Eigen::Vector4i pos, Eigen::Vector4i grid_res){
    return pos[0] + pos[1] * grid_res[0] + pos[2] * grid_res[1] * grid_res[2];
}

__global__ void kernComputeIndices(int N, Eigen::Vector4i grid_res, Eigen::Vector4i grid_min,
                                   Eigen::Vector4f inv_radius, PointType *pos, int *indices, int *grid_indices){
    int index = threadIdx.x + (blockIdx.x *blockDim.x);
    if (index < N){
        PointType pt = pos[index] ;
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
            Eigen::Vector4i ijk(static_cast<int>(floor(pt.x * inv_radius[0])),
                                static_cast<int>(floor(pt.y * inv_radius[1])), static_cast<int>(floor(pt.z * inv_radius[2])), 0);


            Eigen::Vector4i offset = ijk - grid_min;
//            printf("offset is %d, %d, %d \n",offset[0], offset[1], offset[2]);
//            printf("grid res is %d, %d, %d \n", grid_res[0], grid_res[1], grid_res[2]);
            grid_indices[index] = kernComputeIndices(offset, grid_res);
//            printf("indice is %d \n", grid_indices[index] );
            indices[index] = index;
        }

    }
}
