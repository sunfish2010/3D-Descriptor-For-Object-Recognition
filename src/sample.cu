#include "detection.h"
#define VERBOSE 1
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>




/*
 * Atomic functions for float
 * https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
 */

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

struct keep {
    __host__ __device__ bool operator()(const int x) {
        return (x != -1);
    }
};
//
//struct keep_indice {
//    __host__ __device__ bool operator()(const PointType p) {
//        return (x != -1);
//    }
//};


//
//global__ void kernDownSample(int N, const PointType *pos_in, PointType *pos_out, int* keep){
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    if (index < N ){
//        if (keep[index] > 0){
//            pos_out
//        }
//    }
//
//}


__global__ void kernComputeMinDist(int N, const PointType *pos,const int*indices,
        int*min_indices, Eigen::Vector4f inv_radius){
    extern __shared__ float dist[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < N){
        for (int offset = 0; offset * 8192 < N; offset++){
            if (index < 8192 * (offset + 1)&& index >= 8192 * offset){
                PointType pt = pos[index];
                if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
                    Eigen::Vector4i ijk(static_cast<int>(floor(pt.x * inv_radius[0])),
                                        static_cast<int>(floor(pt.y * inv_radius[1])), static_cast<int>(floor(pt.z * inv_radius[2])), 0);
                    float curr_dist = (pt.x - ijk[0]) * (pt.x - ijk[0]) + (pt.y - ijk[1]) * (pt.y - ijk[1])
                                      + (pt.z - ijk[2]) * (pt.z - ijk[2]);
                    //if (indices[index] > 8700 || indices[index] < 0)printf("indice is %d", indices[index]);

//            __syncthreads();
                    int bin = indices[index] - (indices[index] / 8192) * 8192 ;
                    if (bin > 4000)
                        printf("bin is %d, dist is : %f", bin, curr_dist);
                    atomicMin(&dist[bin], curr_dist);

                    if (curr_dist == dist[bin])
                        min_indices[index] = index;
                }
            }
        }
    }

}


//__global__ void kernDownSample(int N, const PointType *pos_in, PointType *pos_out, const int *indices){
//    int index = threadIdx.x + blockDim.x * blockIdx.x;
//    if (index < N){
//        if (indices[index] != -1){
//            pos_out[indices[index]] = pos_in[index];
//        }
//    }
//}

UniformDownSample::~UniformDownSample() {

    cudaFree(dev_kept_indices);
    cudaFree(dev_pc);

    dev_kept_indices = NULL;
    dev_pc = NULL;
}


void UniformDownSample::downSample(const pcl::PointCloud<PointType >::ConstPtr &input,
         Search& tool ) {
    if (input != tool._surface) {
        std::cout << "The input must be the same as the search surface" << std::endl;
    }
    _input = input;
    int grid_N = tool._grid_count;
    std::cout << grid_N << std::endl;
    N = tool._N_surface;
    dim3 fullBlockPerGrid(static_cast<u_int32_t >((N + blockSize - 1) / blockSize));

    cudaMalloc((void **) &dev_kept_indices, N * sizeof(int));
    cudaMemset(dev_kept_indices, -1, N * sizeof(int));

    kernComputeMinDist << < fullBlockPerGrid, blockSize, 8192 * sizeof(float) >> > (N, tool.dev_pos_surface,
            tool.dev_grid_indices, dev_kept_indices, tool.inv_radius);

    checkCUDAError("kernComputeMinDist error");

//    thrust::device_ptr<int> dev_thrust_tmp = thrust::device_ptr<int>(dev_kept_indices);
    int *new_end = thrust::partition(thrust::device, dev_kept_indices, dev_kept_indices + N, keep());

    N_new = static_cast<int>(new_end - dev_kept_indices);

//    cudaMalloc((void**)&dev_new_pc, N * sizeof(PointType));
//    checkCUDAError("malloc dev_coherent_pc error");
//    thrust::copy_if(dev_kept_indices, dev_kept_indices + N, dev_new_pc, keep());
//    checkCUDAError("thrust copy error");
//#if VERBOSE
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "Num unique element is " << num_unique << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "Min is " << min_pi << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "Max is " << max_pi << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "The inverse radius is " << inv_radius << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "The grid count is " << total_grid_count << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "The point cloud dimension is " << pc_dimension << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//
//#endif


}

void UniformDownSample::fillOutput(pcl::PointCloud<PointType>::Ptr &output) {
    if (!output ) {
        std::cout << "" << std::endl;
    }
    std::cout << "---------------------------------------------------------" << std::endl;
    output->height = 1;
    output->is_dense = true;
    output->width = static_cast<uint32_t>(N_new);
    output->points.resize (static_cast<uint32_t>(N_new));
    int count = 0;
    for (auto &i: (*kept_indices)){
        output->points[count] = _input->points[i];
        count ++;
    }
}

IndicesPtr UniformDownSample::getKeptIndices() {
    kept_indices = IndicesPtr(new std::vector<int>(N_new)) ;
    std::cout << "---------------------------------------------------------" << std::endl;
    cudaMemcpy(&(*kept_indices)[0], dev_kept_indices, sizeof(int) * N_new, cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy kept_indices error");
    return kept_indices;
}






//    if (!input || !output || !kept_indices ){
//        cudaFree(dev_min);
//        cudaFree(dev_max);
//        std::cerr << "function not properly initialized " << std::endl;
//        exit(1);
//    }
//
//
//
//    if (dev_grid_indices){
//        cudaMemcpy(&(*grid_indices)[0], dev_grid_indices, N * sizeof(int), cudaMemcpyDeviceToHost);
//        checkCUDAError("kernCopy grid indices failed");
//    }
//
//    thrust::device_ptr<int> dev_thrust_grid_indices =  thrust::device_ptr<int>(dev_grid_indices);
//    thrust::device_ptr<int> dev_thrust_array_indices = thrust::device_ptr<int>(dev_array_indices);
//    thrust::sort_by_key(dev_thrust_grid_indices, dev_thrust_grid_indices + N, dev_thrust_array_indices);
//
//    // get the coherent val for original entry
//
//    kernUniformDownSample <<<fullBlockPerGrid_points, blockSize >>> (N, dev_pc, dev_new_pc, dev_array_indices);
//    checkCUDAError("kernGetCoherentVal Failed");
//
//    // get the first occurance of unique indices
//    cudaMalloc((void**)&dev_tmp, N * sizeof(int));
//    checkCUDAError("cudaMalloc dev_tmp failed");
//
//    isfirst_indices<<< fullBlockPerGrid_points, blockSize>>> (N, dev_grid_indices, dev_tmp);
//    checkCUDAError("isfirst indices error");
//
//    //thrust::device_ptr<int> dev_thrust_tmp = thrust::device_ptr<int>(dev_tmp);
//
//    int * new_end = thrust::partition(thrust::device, dev_tmp, dev_tmp + N, isFirst());
//
//    int num_unique = static_cast<int>(new_end - dev_tmp);
////
//    std::vector<int>unique_indices(num_unique, 0);
//    cudaMemcpy(&unique_indices[0], dev_tmp, num_unique  * sizeof(int), cudaMemcpyDeviceToHost);
//    checkCUDAError("Memcpy device to host Failed");
////    cudaMalloc((void**)&dev_dist, N * sizeof(int));
////    checkCUDAError("malloc dev_dist error");
//    kernComputeDist <<< fullBlockPerGrid_points, blockSize >>>(N, dev_new_pc, dev_tmp, inv_radius);
//    checkCUDAError("KernComputeDist failed");
//
//




//
//
//
//    // sort inplace
//    thrust::sort_by_key(dev_thrust_dist, dev_thrust_dist + N, dev_thrust_array_indices);
//    kernUniformDownSample <<<fullBlockPerGrid_points, blockSize >>> (N, dev_pc, dev_new_pc, dev_array_indices);
//    checkCUDAError("kernGetCoherentVal Failed");

//    std::vector<PointType>new_pts(num_unique);
//    cudaMemcpy(&new_pts[0], dev_new_pc, num_unique * sizeof(PointType), cudaMemcpyDeviceToHost);




    //thrust::copy(dev_tmp, new_end, unique_indices.begin());



//        cudaMalloc((void**)&dev_grid_start, total_grid_count * sizeof(int));
//        cudaMalloc((void**)&dev_grid_end, total_grid_count * sizeof(int))
//        cudaMemset(dev_grid_start, -1, total_grid_count * sizeof(int));
//        cudaMemset(dev_grid_end, -1, total_grid_count * sizeof(int));
//
//        kernIdentifyCellStartEnd <<<fullBlockPerGrid_points, blockSize >>> (N, dev_grid_indices,
//                dev_grid_start, dev_grid_end);
//        checkCUDAErrorWithLine("kernIdentifyCellStartEnd Failed");



    // this is rather unefficeint
    //cudaMalloc((void**)&dev_distance, N * sizeof(int));
//    std::vector<int> grid_indices(N, 0);
//    cudaMemcpy(&grid_indices[0], dev_grid_indices, N  * sizeof(int), cudaMemcpyDeviceToHost);
//    for (auto i:grid_indices)
//        std::cout << i << std::endl;
//    checkCUDAError("Memcpy device to host Failed");


//    thrust::counting_iterator
//    for (int i = 0; i < total_grid_count; i++){
//
//    }


//
    //for (int i = 0; i < 100; i++) std::cout << grid_indices[i] << ",";

//
//


    //kernUniformDownSample <<<fullBlockPerGrid_points, blockSize>>> (n_model, radius, dev_ss_pc_model, dev_ss_pc_model);

