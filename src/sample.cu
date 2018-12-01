#include "detection.h"
#define VERBOSE 1
#include <cstdio>

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

struct keep {
    __host__ __device__ bool operator()(const PointType p) {
        return (p.x != NAN);
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


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__global__ void kernRandomDownSample(int N, float p, PointType* pos){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N){
        thrust::default_random_engine rng = makeSeededRandomEngine(0, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) < p)
            pos[index].x = NAN;
    }
}


UniformDownSample::~UniformDownSample() {
    cudaFree(dev_grid_indices);
    cudaFree(dev_array_indices);
    cudaFree(dev_new_pc);
    cudaFree(dev_pc);
    cudaFree(dev_tmp);
    dev_grid_indices = NULL;
    dev_array_indices = NULL;
    dev_new_pc = NULL;
    dev_tmp = NULL;
    dev_pc = NULL;
}

void UniformDownSample::setRadius(float radius) {this->radius = radius;}

void UniformDownSample::randDownSample(const pcl::PointCloud<PointType>::ConstPtr &input,
                                       pcl::PointCloud<PointType>::Ptr &output) {
    if (!input || !output ){
        std::cerr << "function not properly initialized " << std::endl;
        exit(1);
    }
    N = (int)(*input).size();
    dev_pc = NULL;
    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));
    cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
    cudaMemcpy(dev_pc, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy pc");

//    // get the first occurance of unique indices
//    cudaMalloc((void**)&dev_tmp, N * sizeof(int));
//    checkCUDAError("cudaMalloc dev_tmp failed");
//    cudaMemset(dev_tmp, -1, N* sizeof(int));
//    checkCUDAError("memset error");

    kernRandomDownSample<<<fullBlockPerGrid_points, blockSize>>>(N, 0.98, dev_pc);

//    thrust::device_ptr<PointType> dev_thrust_pc =  thrust::device_ptr<PointType>(dev_grid_indices);
//    int * new_end = thrust::partition(dev_thrust_pc, dev_tpc + N, keep());
//    int num_unique = static_cast<int>(new_end - dev_tmp);
//
//    std::vector<int>unique_indices(num_unique, 0);


    std::cout << "---------------------------------------------------------" << std::endl;
    (*output).height = 1;
    (*output).is_dense = false;
    (*output).width = static_cast<uint32_t>(N);
    (*output).points.resize (static_cast<uint32_t>(N));
    cudaMemcpy(&(*output).points[0], dev_pc, N  * sizeof(PointType), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < num_unique;i++){
//        (*output).points[i] = (*input).points[unique_indices[i]];
//    }

    cudaFree(dev_pc);


}

void UniformDownSample::downSample(const pcl::PointCloud<PointType >::ConstPtr &input,
        pcl::PointCloud<PointType>::Ptr &output, const IndicesPtr &grid_indices, const IndicesPtr &array_indices,
        const Eigen::Vector4f &inv_radius ) {
    if (!input || !output || !kept_indices ){
        std::cerr << "function not properly initialized " << std::endl;
        exit(1);
    }

    N = (int)(*input).size();
    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));

    //Eigen::Vector4i grid_res = Eigen::Vector4i(1, pc_dimension[0], pc_dimension[0] * pc_dimension[1], 0);

    cudaMalloc((void**)&dev_grid_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMemcpy(dev_grid_indices,& (*grid_indices)[0], N * sizeof(int));
    checkCUDAError("cudaMemcpy dev_indices error");
    cudaMalloc((void**)&dev_array_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMemcpy(dev_array_indices,& (*array_indices)[0], N * sizeof(int));
    checkCUDAError("cudaMemcy dev_indices error");

    cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
    cudaMemcpy(dev_pc, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy pc");

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


//#if VERBOSE
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "Num unique element is " << num_unique << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "The grid count is " << total_grid_count << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::cout << "The point cloud dimension is " << pc_dimension << std::endl;
//    std::cout << "---------------------------------------------------------" << std::endl;
//
//#endif

}



