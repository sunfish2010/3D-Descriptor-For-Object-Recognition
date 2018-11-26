#include "detection.h"
#define VERBOSE 1
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include "cudaGrid.h"

//Copy index
__global__ void isfirst_indices(int N, int *input, int *res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        if (tid == 0) {
            res[0] = 0;
        } else if (input[tid] != input[tid - 1]) {
            res[tid] = tid;
        } else {
            res[tid] = -1;
        }
    }
}


struct isFirst {
    __host__ __device__ bool operator()(const int x) {
        return (x != -1);
    }
};


__device__ float kernComputeDist(PointType pos, Eigen::Vector4i ijk){
    return (pos.x - ijk[0]) * (pos.x - ijk[0]) + (pos.y - ijk[1]) * (pos.y - ijk[1])
            + (pos.z - ijk[2]) * (pos.z - ijk[2]);
}


// uniform downsampling the points
__global__ void kernComputeDist(int N, const PointType *pts_in, int *dist, Eigen::Vector4f inv_radius){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < N){
        PointType pt = pts_in[index] ;
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
            Eigen::Vector4i ijk(static_cast<int>(floor(pt.x * inv_radius[0])),
                                static_cast<int>(floor(pt.y * inv_radius[1])), static_cast<int>(floor(pt.z * inv_radius[2])), 0);
            dist[index] = kernComputeDist(pts_in[index], ijk);
        }
    }
}


__global__ void kernUniformDownSample(int N, PointType *pts_in, PointType *pts_out, int *indices){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N){
        pts_out[index] = pts_in[indices[index]];
    }
}

UniformDownSample::UniformDownSample(float radius): radius(radius), N_new(0), N(0){
    cudaMalloc((void**)&dev_min, sizeof(Eigen::Vector4f));
    cudaMalloc((void**)&dev_max, sizeof(Eigen::Vector4f));
    checkCUDAError("cudaMalloc min,max");
}

UniformDownSample::~UniformDownSample() {
    cudaFree(dev_min);
    cudaFree(dev_max);
    cudaFree(dev_grid_indices);
    cudaFree(dev_array_indices);
    cudaFree(dev_new_pc);
    cudaFree(dev_pc);
    cudaFree(dev_tmp);
    dev_min = NULL;
    dev_max = NULL;
    dev_grid_indices = NULL;
    dev_array_indices = NULL;
    dev_new_pc = NULL;
    dev_tmp = NULL;
    dev_pc = NULL;
}

void UniformDownSample::setRadius(float radius) {this->radius = radius;}

void UniformDownSample::downSample(const pcl::PointCloud<PointType >::ConstPtr &input,
        pcl::PointCloud<PointType>::Ptr &output) {
    if (!input || !output || !kept_indices ){
        cudaFree(dev_min);
        cudaFree(dev_max);
        std::cerr << "function not properly initialized " << std::endl;
        exit(1);
    }

    N = (int)(*input).size();
    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));
    cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
    cudaMemcpy(dev_pc, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy pc");

    // calculate min max for the pc

    Eigen::Vector4f min_p, max_p;

    min_p.setConstant(FLT_MAX);
    max_p.setConstant(-FLT_MAX);
    cudaMemcpy(dev_min, &min_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_max, &max_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy min,max");

    getMinMax <<< fullBlockPerGrid_points, blockSize>>>(N, dev_pc, dev_min, dev_max);
    checkCUDAError("getMinMax error");
    cudaMemcpy(&min_p, dev_min, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy min  error");
    cudaMemcpy(&max_p, dev_max, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy max error");
    // device the pc into cells

    Eigen::Vector4f inv_radius = Eigen::Array4f::Ones()/ (Eigen::Vector4f(radius, radius, radius, 1.0f).array());
    Eigen::Vector4i max_pi(static_cast<int>(floor(max_p[0] * inv_radius[0])),
            static_cast<int>(floor(max_p[1] * inv_radius[1])), static_cast<int>(floor(max_p[2] * inv_radius[2])), 0);
    Eigen::Vector4i min_pi(static_cast<int>(floor(min_p[0] * inv_radius[0])),
            static_cast<int>(floor(min_p[1] * inv_radius[1])), static_cast<int>(floor(inv_radius[2] * min_p[2])), 0);


    Eigen::Vector4i pc_dimension = max_pi - min_pi + Eigen::Vector4i::Ones();
    pc_dimension[3] = 0;

    int total_grid_count = pc_dimension[0] * pc_dimension[1] * pc_dimension[2];

    //Eigen::Vector4i grid_res = Eigen::Vector4i(1, pc_dimension[0], pc_dimension[0] * pc_dimension[1], 0);

    cudaMalloc((void**)&dev_grid_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMalloc((void**)&dev_array_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");

    kernComputeIndices <<< fullBlockPerGrid_points, blockSize >>> (N, pc_dimension, min_pi,
        inv_radius, dev_pc, dev_array_indices, dev_grid_indices);
    checkCUDAError("kernComputeIndices Failed");

    if (dev_grid_indices){
        cudaMemcpy(&(*grid_indices)[0], dev_grid_indices, N * sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError("kernCopy grid indices failed");
    }

    thrust::device_ptr<int> dev_thrust_grid_indices =  thrust::device_ptr<int>(dev_grid_indices);
    thrust::device_ptr<int> dev_thrust_array_indices = thrust::device_ptr<int>(dev_array_indices);
    thrust::sort_by_key(dev_thrust_grid_indices, dev_thrust_grid_indices + N, dev_thrust_array_indices);

    // get the coherent val for original entry
    cudaMalloc((void**)&dev_new_pc, N * sizeof(PointType));
    checkCUDAError("malloc dev_coherent_pc error");
    kernUniformDownSample <<<fullBlockPerGrid_points, blockSize >>> (N, dev_pc, dev_new_pc, dev_array_indices);
    checkCUDAError("kernGetCoherentVal Failed");

    // get the first occurance of unique indices
    cudaMalloc((void**)&dev_tmp, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_tmp failed");

    isfirst_indices<<< fullBlockPerGrid_points, blockSize>>> (N, dev_grid_indices, dev_tmp);
    checkCUDAError("isfirst indices error");

    //thrust::device_ptr<int> dev_thrust_tmp = thrust::device_ptr<int>(dev_tmp);

    int * new_end = thrust::partition(thrust::device, dev_tmp, dev_tmp + N, isFirst());

    int num_unique = static_cast<int>(new_end - dev_tmp);
//
    std::vector<int>unique_indices(num_unique, 0);
    cudaMemcpy(&unique_indices[0], dev_tmp, num_unique  * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("Memcpy device to host Failed");
//    cudaMalloc((void**)&dev_dist, N * sizeof(int));
//    checkCUDAError("malloc dev_dist error");
    kernComputeDist <<< fullBlockPerGrid_points, blockSize >>>(N, dev_new_pc, dev_tmp, inv_radius);
    checkCUDAError("KernComputeDist failed");

    std::cout << "---------------------------------------------------------" << std::endl;
    (*output).height = 1;
    (*output).is_dense = true;
    (*output).width = static_cast<uint32_t>(num_unique);
    (*output).points.resize (static_cast<uint32_t>(num_unique));

    //for (auto & i : unique_indices) std::cout << i << std::endl;


    std::vector<int>dist(N, 0);
    std::vector<int>indices(N, 0);
    cudaMemcpy(&dist[0], dev_tmp, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&indices[0], dev_array_indices, N * sizeof(int), cudaMemcpyDeviceToHost);

    int cell_start = unique_indices[0];
    int cell_end = unique_indices[1];
    for (int i = 2; i < unique_indices.size(); i++){
       std::sort(indices.begin() + cell_start, indices.begin() + cell_end, [&](const int& a, const int& b) {
           return (dist[a] < dist[b]);
       });

       //std::cout << "min is " << indices[cell_start] << std::endl;
       (*output).points[i] = (*input).points[indices[cell_start]];
       kept_indices->emplace_back(indices[cell_start]);
       cell_start = cell_end;
       cell_end = unique_indices[i];
    }
    (*output).points[unique_indices.size()-1] = (*input).points[unique_indices[unique_indices.size()- 1]];

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



#if VERBOSE
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Num unique element is " << num_unique << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Min is " << min_pi << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Max is " << max_pi << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "The inverse radius is " << inv_radius << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "The grid count is " << total_grid_count << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "The point cloud dimension is " << pc_dimension << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

#endif

    //kernUniformDownSample <<<fullBlockPerGrid_points, blockSize>>> (n_model, radius, dev_ss_pc_model, dev_ss_pc_model);
}
