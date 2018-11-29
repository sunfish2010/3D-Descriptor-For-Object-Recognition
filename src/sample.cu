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


/** \brief common functions for search  */
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
__global__ void kernDownSample(int N, const float *dist, const float *dist_min, int *grid_indices,  int* keep){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N ){
        if (dist[index] == dist_min[grid_indices[index]]){
            keep[index] = index;
        }
    }

}


__global__ void kernComputeIndicesDistances(int N, Eigen::Vector4i grid_res, Eigen::Vector4i grid_min,
        Eigen::Vector4f inv_radius, PointType *pos, int *grid_indices, float*min_dist, float*dist){
//    extern __shared__ float dist[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < N){
        PointType pt = pos[index];
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
            Eigen::Vector4i ijk(static_cast<int>(floor(pt.x * inv_radius[0])),
                                static_cast<int>(floor(pt.y * inv_radius[1])), static_cast<int>(floor(pt.z * inv_radius[2])), 0);
            float curr_dist = (pt.x - ijk[0]) * (pt.x - ijk[0]) + (pt.y - ijk[1]) * (pt.y - ijk[1])
                              + (pt.z - ijk[2]) * (pt.z - ijk[2]);
            Eigen::Vector4i offset = ijk - grid_min;
            int i = offset[0] + offset[1] * grid_res[0] + offset[2] * grid_res[0] * grid_res[1];
            grid_indices[index] = i;
            dist[index] = curr_dist;
            atomicMin(&min_dist[i], curr_dist);

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
    cudaFree(dev_grid_indices);
    cudaFree(dev_min);
    cudaFree(dev_max);
    dev_kept_indices = NULL;
}


void UniformDownSample::downSample(const pcl::PointCloud<PointType >::ConstPtr &input) {
    if (radius == 0)
        std::cerr << "error" << std::endl;
    _input = input;
    N = static_cast<int>((*input).points.size());
    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));
    cudaMalloc((void**) &dev_pos_surface, N * sizeof(PointType));
    cudaMemcpy(dev_pos_surface, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy pc");

    // calculate min max for the pc
    Eigen::Vector4f min_p, max_p;
    min_p.setConstant(FLT_MAX);
    max_p.setConstant(-FLT_MAX);
    cudaMemcpy(dev_min, &min_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_max, &max_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy min,max");

    getMinMax <<< fullBlockPerGrid_points, blockSize>>>(N, dev_pos_surface, dev_min, dev_max);
    checkCUDAError("getMinMax error");
    cudaMemcpy(&min_p, dev_min, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy min  error");
    cudaMemcpy(&max_p, dev_max, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy max error");

    // devide the pc into cells
    inv_radius = Eigen::Array4f::Ones()/ (Eigen::Vector4f(radius, radius, radius, 1.0f).array());
    max_pi = Eigen::Vector4i(static_cast<int>(floor(max_p[0] * inv_radius[0])),
                             static_cast<int>(floor(max_p[1] * inv_radius[1])), static_cast<int>(floor(max_p[2] * inv_radius[2])), 0);
    min_pi = Eigen::Vector4i(static_cast<int>(floor(min_p[0] * inv_radius[0])),
                             static_cast<int>(floor(min_p[1] * inv_radius[1])), static_cast<int>(floor(inv_radius[2] * min_p[2])), 0);


    pc_dimension = max_pi - min_pi + Eigen::Vector4i::Ones();
    pc_dimension[3] = 0;

    _grid_count_max = pc_dimension[0] + pc_dimension[0] * pc_dimension[1] + pc_dimension[0] * pc_dimension[1] * pc_dimension[2];

    //Eigen::Vector4i grid_res = Eigen::Vector4i(1, pc_dimension[0], pc_dimension[0] * pc_dimension[1], 0);

    cudaMalloc((void**)&dev_grid_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMalloc((void**)&dev_kept_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMemset(dev_kept_indices, -1, N * sizeof(int));

    cudaMalloc((void**)&dev_dist, N * sizeof(float));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMalloc((void**)&dev_min_dist, _grid_count_max * sizeof(float));
    checkCUDAError("cudaMalloc dev_indices error");
    thrust::device_ptr<float> dev_ptr(dev_min_dist);
    thrust::fill(dev_ptr, dev_ptr + N, FLT_MAX);

    kernComputeIndicesDistances <<< fullBlockPerGrid_points, blockSize >>> (N, pc_dimension, min_pi,
            inv_radius, dev_pos_surface, dev_grid_indices,  dev_min_dist, dev_dist);
    checkCUDAError("kernComputeIndices Failed");

    kernDownSample<<< fullBlockPerGrid_points, blockSize>>>(N, dev_dist, dev_min_dist, dev_grid_indices,dev_kept_indices);

    int * new_end = thrust::partition(thrust::device, dev_kept_indices, dev_kept_indices + N, keep());
    N_new = static_cast<int>(new_end - dev_kept_indices);

    std::cout << N_new << std::endl;

}

void UniformDownSample::fillOutput(pcl::PointCloud<PointType>::Ptr &output) {
    if (!output || !_input ) {
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

