#include "detection.h"
#define VERBOSE 1
#include <cstdio>

static cudaEvent_t start, stop;
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




/** \brief downsampling the point cloud by using global memory for performance improvement
 * **/

__global__ void kernComputeIndicesDistances(int N, Eigen::Vector4i grid_res, Eigen::Vector4i grid_min,
                                            const Eigen::Vector4f inv_radius, PointType *pos, int *grid_indices,
                                            float*min_dist, float*dist){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < N){
//        printf("index is %d \n", index);
        PointType pt = pos[index];
//        printf("1111111 \n");
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
            Eigen::Vector4i ijk(static_cast<int>(floor(pt.x * inv_radius[0])),
                                static_cast<int>(floor(pt.y * inv_radius[1])), static_cast<int>(floor(pt.z * inv_radius[2])), 0);
            float curr_dist = (pt.x - ijk[0]) * (pt.x - ijk[0]) + (pt.y - ijk[1]) * (pt.y - ijk[1])
                              + (pt.z - ijk[2]) * (pt.z - ijk[2]);
            Eigen::Vector4i offset = ijk - grid_min;
            int i = offset[0] + offset[1] * grid_res[0] + offset[2] * grid_res[0] * grid_res[1];
//            printf("index i is %d \n", i);
            grid_indices[index] = i;
            dist[index] = curr_dist;
            atomicMin(&min_dist[i], curr_dist);
//            printf("min dist is %f, curr dist is %f, index is %d \n", min_dist[i], curr_dist, grid_indices[index]);
        }
    }
}

__global__ void kernDownSample(int N, const float *dist, const float *dist_min, int *grid_indices,  int* keep){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N ){
        int grid_i = grid_indices[index];
//        if (grid_i != 0)
//            printf("index is %d, kernDOwnsample grid_i is %d\n", index, grid_indices[index]);
        if (dist[index] == dist_min[grid_i]){
            keep[grid_i] = index;
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

    dev_grid_indices = NULL;
    dev_array_indices = NULL;
    dev_new_pc = NULL;
    dev_tmp = NULL;
    dev_pc = NULL;
    kept_indices.clear();
}



/** \brief  randoming selecting indices to keep
 *  would lead weird matching behavior in the end
 * **/
//void UniformDownSample::randDownSample(const pcl::PointCloud<PointType>::ConstPtr &input,
//                                       pcl::PointCloud<PointType>::Ptr &output) {
//    if (!input || !output ){
//        std::cerr << "function not properly initialized " << std::endl;
//        exit(1);
//    }
//    N = (int)(*input).size();
//    dev_pc = NULL;
//    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));
//    cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
//    cudaMemcpy(dev_pc, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
//    checkCUDAError("cudaMemcpy pc");
//
//    kernRandomDownSample<<<fullBlockPerGrid_points, blockSize>>>(N, 0.98, dev_pc);
//
//    std::cout << "---------------------------------------------------------" << std::endl;
//    (*output).height = 1;
//    (*output).is_dense = false;
//    (*output).width = static_cast<uint32_t>(N);
//    (*output).points.resize (static_cast<uint32_t>(N));
//    cudaMemcpy(&(*output).points[0], dev_pc, N  * sizeof(PointType), cudaMemcpyDeviceToHost);
//
//    cudaFree(dev_pc);
//}
//
//
///** \brief downsampling the point cloud by computing unique indices and
// *  resampling using cpu, this method is slower and use more memory than the atomic version
// * **/
//
//void UniformDownSample::downSample(const pcl::PointCloud<PointType >::ConstPtr &input,
//                                   pcl::PointCloud<PointType>::Ptr &output, IndicesPtr &kept_indices,const  IndicesPtr &grid_indices, const IndicesPtr &array_indices,
//                                   const Eigen::Vector4f &inv_radius ) {
//    if (!input || !output || !kept_indices ){
//        std::cerr << "function not properly initialized " << std::endl;
//        exit(1);
//    }
//
//    N = (int)(*input).size();
//    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));
//
//    //Eigen::Vector4i grid_res = Eigen::Vector4i(1, pc_dimension[0], pc_dimension[0] * pc_dimension[1], 0);
//
//    cudaMalloc((void**)&dev_grid_indices, N * sizeof(int));
//    checkCUDAError("cudaMalloc dev_indices error");
//    cudaMemcpy(dev_grid_indices,& (*grid_indices)[0], N * sizeof(int), cudaMemcpyHostToDevice);
//    checkCUDAError("cudaMemcpy dev_indices error");
//    cudaMalloc((void**)&dev_array_indices, N * sizeof(int));
//    checkCUDAError("cudaMalloc dev_indices error");
//    cudaMemcpy(dev_array_indices,& (*array_indices)[0], N * sizeof(int), cudaMemcpyHostToDevice);
//    checkCUDAError("cudaMemcy dev_indices error");
//
//    cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
//    cudaMemcpy(dev_pc, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
//    checkCUDAError("cudaMemcpy pc");
//
//    // computation begin
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    float miliseconds = 0;
//
//    cudaEventRecord(start);
//    thrust::device_ptr<int> dev_thrust_grid_indices =  thrust::device_ptr<int>(dev_grid_indices);
//    thrust::device_ptr<int> dev_thrust_array_indices = thrust::device_ptr<int>(dev_array_indices);
//    thrust::sort_by_key(dev_thrust_grid_indices, dev_thrust_grid_indices + N, dev_thrust_array_indices);
//
//    // get the coherent val for original entry
//    cudaMalloc((void**)&dev_new_pc, N * sizeof(PointType));
//    checkCUDAError("malloc dev_coherent_pc error");
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
//
//
//    std::vector<int>unique_indices(num_unique, 0);
//    cudaMemcpy(&unique_indices[0], dev_tmp, num_unique  * sizeof(int), cudaMemcpyDeviceToHost);
//    checkCUDAError("Memcpy device to host Failed");
////    cudaMalloc((void**)&dev_dist, N * sizeof(int));
////    checkCUDAError("malloc dev_dist error");
//    kernComputeDist <<< fullBlockPerGrid_points, blockSize >>>(N, dev_new_pc, dev_tmp, inv_radius);
//    checkCUDAError("KernComputeDist failed");
//
//
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&miliseconds, start, stop);
//    std::cout << "uniform subsampling takes: " << miliseconds << std::endl;
//
//    std::cout << "---------------------------------------------------------" << std::endl;
//    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
//    (*output).height = 1;
//    (*output).is_dense = true;
//    (*output).width = static_cast<uint32_t>(num_unique);
//    (*output).points.resize (static_cast<uint32_t>(num_unique));
//
//    //for (auto & i : unique_indices) std::cout << i << std::endl;
//
//
//    std::vector<int>dist(N, 0);
//    std::vector<int>indices(N, 0);
//    cudaMemcpy(&dist[0], dev_tmp, N * sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&indices[0], dev_array_indices, N * sizeof(int), cudaMemcpyDeviceToHost);
//
//    int cell_start = unique_indices[0];
//    int cell_end = unique_indices[1];
//    for (int i = 2; i < unique_indices.size(); i++){
//        std::sort(indices.begin() + cell_start, indices.begin() + cell_end, [&](const int& a, const int& b) {
//            return (dist[a] < dist[b]);
//        });
//
//        //std::cout << "min is " << indices[cell_start] << std::endl;
//        (*output).points[i] = (*input).points[indices[cell_start]];
//        kept_indices->emplace_back(indices[cell_start]);
//        cell_start = cell_end;
//        cell_end = unique_indices[i];
//    }
//    (*output).points[unique_indices.size()-1] = (*input).points[unique_indices[unique_indices.size()- 1]];
//    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
//    std::cout << "cpu copying points after computation takes: " << duration << std::endl;
//
////#if VERBOSE
////    std::cout << "---------------------------------------------------------" << std::endl;
////    std::cout << "Num unique element is " << num_unique << std::endl;
////    std::cout << "---------------------------------------------------------" << std::endl;
////    std::cout << "The grid count is " << total_grid_count << std::endl;
////    std::cout << "---------------------------------------------------------" << std::endl;
////    std::cout << "The point cloud dimension is " << pc_dimension << std::endl;
////    std::cout << "---------------------------------------------------------" << std::endl;
////
////#endif
//
//}


void UniformDownSample::downSampleAtomic(const pcl::PointCloud<PointType >::ConstPtr &input,
                                         const Eigen::Vector4f &inv_radius, const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi ) {

    if (!input ){
        std::cerr << "function not properly initialized " << std::endl;
        exit(1);
    }


    N = (int)(*input).size();
    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));

    int _grid_count_max = pc_dimension[0] + pc_dimension[0] * pc_dimension[1] + pc_dimension[0] * pc_dimension[1] * pc_dimension[2];
    std::cout << "Max possible grid index is " << _grid_count_max << std::endl;
    assert(_grid_count_max < N);

    //Eigen::Vector4i grid_res = Eigen::Vector4i(1, pc_dimension[0], pc_dimension[0] * pc_dimension[1], 0);
    cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
    cudaMemcpy(dev_pc, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy pc");

    // computation begin
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float miliseconds = 0;

    cudaEventRecord(start);

    cudaMalloc((void**)&dev_grid_indices, N * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");

    cudaMalloc((void**)&dev_dist, N * sizeof(float));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMalloc((void**)&dev_min_dist, _grid_count_max * sizeof(float));
    checkCUDAError("cudaMalloc dev_indices error");
    thrust::device_ptr<float> dev_ptr(dev_min_dist);
    thrust::fill(dev_ptr, dev_ptr + _grid_count_max , FLT_MAX);
    checkCUDAError("thrust error");

    kernComputeIndicesDistances <<< fullBlockPerGrid_points, blockSize >>> (N, pc_dimension, min_pi,
            inv_radius, dev_pc, dev_grid_indices,  dev_min_dist, dev_dist);
    checkCUDAError("kernComputeIndices Failed");

//    cudaFree(dev_pc);

    cudaMalloc((void**)&dev_kept_indices, _grid_count_max * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMemset(dev_kept_indices, -1, _grid_count_max * sizeof(int));
    checkCUDAError("memset error Failed");

    kernDownSample<<< fullBlockPerGrid_points, blockSize>>>(N, dev_dist, dev_min_dist, dev_grid_indices,dev_kept_indices);
    checkCUDAError("kernDownSample Failed");
    std::cout << "KernDOwnsample finished " << std::endl;
    try{
        int * new_end = thrust::partition(thrust::device, dev_kept_indices, dev_kept_indices + _grid_count_max, isFirst());
        N_new = static_cast<int>(new_end - dev_kept_indices);
    }
    catch (thrust::system_error e){
        std::cerr << "Error inside sort: " << e.what() << std::endl;
        exit(1);
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    std::cout << "uniform subsampling takes: " << miliseconds << std::endl;

    std::cout << "---------Saving results useful for later computation ----------------------------------" << std::endl;
    kept_indices.resize(N_new);
    cudaMemcpy(&kept_indices[0], dev_kept_indices, sizeof(int) * N_new, cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy kept_indices error");

//    grid_indices.resize(N);
//    cudaMemcpy(&grid_indices[0], dev_grid_indices, sizeof(int) * N, cudaMemcpyDeviceToHost);
//    checkCUDAError("cudamemcpy grid_indices error");

//    for (auto &i:grid_indices)
//        std::cout << i << std::endl;

    std::cout << N_new << std::endl;

    cudaFree(dev_grid_indices);
    cudaFree(dev_pc);
    cudaFree(dev_kept_indices);
    cudaFree(dev_dist);
    cudaFree(dev_min_dist);
    checkCUDAError("cuda free");

}


void UniformDownSample::display(const pcl::PointCloud<PointType>::ConstPtr &input, const pcl::PointCloud<PointType>::Ptr &output) {
    (*output).height = 1;
    (*output).is_dense = true;
    (*output).width = static_cast<uint32_t>(N_new);
    (*output).points.resize (static_cast<uint32_t>(N_new));


//    IndicesPtr kept_indices = IndicesPtr(new std::vector<int>(N_new));
    std::cout << "initialization" << std::endl;


    for (int i = 0; i < N_new; i++){
        (*output).points[i] = (*input).points[kept_indices[i]];
    }
    std::cout << "copy is done" << std::endl;

}


