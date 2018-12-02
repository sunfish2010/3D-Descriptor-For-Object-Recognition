
#include "grid.h"

static cudaEvent_t start, stop;
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
    __shared__ float min_max[6];
    for (int i = 0; i < 3; ++i)
        min_max[i] = FLT_MAX;
    for (int i = 3; i < 6; ++i)
        min_max[i] = FLT_MIN;
    __syncthreads();
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N){
        PointType pt = pts_in[index] ;
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
            atomicMin(&min_max[0], pt.x);
            atomicMin(&min_max[1], pt.y);
            atomicMin(&min_max[2], pt.z);
            atomicMax(&min_max[3], pt.x);
            atomicMax(&min_max[4], pt.y);
            atomicMax(&min_max[5], pt.z);
        }
    }
    __syncthreads();
    atomicMin(&(*min_pt)[0], min_max[0]);
    atomicMin(&(*min_pt)[1], min_max[1]);
    atomicMin(&(*min_pt)[2], min_max[2]);
    atomicMax(&(*max_pt)[0], min_max[3]);
    atomicMax(&(*max_pt)[1], min_max[4]);
    atomicMax(&(*max_pt)[2], min_max[5]);
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
            grid_indices[index] = kernComputeIndices(offset, grid_res);
            indices[index] = index;
        }

    }
}



void Grid::computeSceneProperty(const pcl::PointCloud<PointType>::ConstPtr &input, const IndicesPtr &grid_indices,
        const IndicesPtr &array_indices) {
    if (!input || radius <= 0 || !grid_indices || !array_indices){
        std::cerr <<  "ComputeSceneProperty input not set correctly " << std::endl;
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float miliseconds = 0;

    N = (int)(*input).size();
    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));

    cudaEventRecord(start);
    cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
    cudaMemcpy(dev_pc, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy pc");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    std::cout << "allocate and memcpy pt takes: " << miliseconds << std::endl;
    // calculate min max for the pc

    Eigen::Vector4f min_p, max_p;

    min_p.setConstant(FLT_MAX);
    max_p.setConstant(-FLT_MAX);
    cudaEventRecord(start);
    cudaMalloc((void**)&dev_min, sizeof(Eigen::Vector4f));
    checkCUDAError("cudaMalloc min");
    cudaMalloc((void**)&dev_max, sizeof(Eigen::Vector4f));
    checkCUDAError("cudaMalloc max");
    cudaMemcpy(dev_min, &min_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_max, &max_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy min,max");

    getMinMax <<< fullBlockPerGrid_points, blockSize>>>(N, dev_pc, dev_min, dev_max);
    checkCUDAError("getMinMax error");
    cudaMemcpy(&min_p, dev_min, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy min  error");
    cudaMemcpy(&max_p, dev_max, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy max error");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    std::cout << "calculating min max takes  " << miliseconds << std::endl;
    // device the pc into cells

    inv_radius = Eigen::Array4f::Ones()/ (Eigen::Vector4f(radius, radius, radius, 1.0f).array());
    max_pi = Eigen::Vector4i (static_cast<int>(floor(max_p[0] * inv_radius[0])),
                           static_cast<int>(floor(max_p[1] * inv_radius[1])), static_cast<int>(floor(max_p[2] * inv_radius[2])), 0);
    min_pi = Eigen::Vector4i (static_cast<int>(floor(min_p[0] * inv_radius[0])),
                           static_cast<int>(floor(min_p[1] * inv_radius[1])), static_cast<int>(floor(inv_radius[2] * min_p[2])), 0);


    pc_dimension = max_pi - min_pi + Eigen::Vector4i::Ones();
    pc_dimension[3] = 0;

    cudaEventRecord(start);
    cudaMalloc((void**)&dev_grid_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMalloc((void**)&dev_array_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");

    kernComputeIndices <<< fullBlockPerGrid_points, blockSize >>> (N, pc_dimension, min_pi,
            inv_radius, dev_pc, dev_array_indices, dev_grid_indices);
    checkCUDAError("kernComputeIndices Failed");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    std::cout << "calculating array & grid indices takes  " <<miliseconds << std::endl;

    if (grid_indices){
        cudaMemcpy(&(*grid_indices)[0], dev_grid_indices, N * sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError("kernCopy grid indices failed");
    }

    if (array_indices){
        cudaMemcpy(&(*array_indices)[0], dev_array_indices, N * sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError("kernCopy array indices failed");
    }


}


Grid::~Grid() {
    cudaFree(dev_array_indices);
    cudaFree(dev_grid_indices);
    cudaFree(dev_min);
    cudaFree(dev_max);
    cudaFree(dev_pc);
}