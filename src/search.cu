#include "search.h"


__global__ void kernSearchRadius(int N, int *feature_indices, int * output, PointType *surface) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < N){

    }
}

Search::~Search() {
    cudaFree(dev_pos_surface);
    cudaFree(dev_neighbor_indices);
    cudaFree(dev_features_indices);
    cudaFree(dev_grid_indices);
    dev_neighbor_indices = NULL;
    dev_features_indices = NULL;
    dev_pos_surface = NULL;
}

void Search::search(const pcl::PointCloud<PointType>::Ptr &output) {
    if (_method == SearchMethod::KDTree) {
        std::cout << "Function not implemented yet" << std::endl;
        exit(1);
    }
    if (_method == SearchMethod::Radius){
        cudaMalloc((void**)&dev_features_indices, _N_features * sizeof(int));
        checkCUDAError("mallod dev_features_indices error");
        cudaMemcpy(dev_features_indices, &(*_feature_indices)[0], _N_features * sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError("memcpy dev_features_indices error");
        cudaMalloc((void**)&dev_neighbor_indices, _N_features * _n * sizeof(int));
        checkCUDAError("malloc dev_neighbor indices error");
        cudaMemset(dev_neighbor_indices, -1, _N_features * _n * sizeof(int));
        checkCUDAError("memset ni error");
        cudaMalloc((void**)&dev_pos_surface, _N_surface * sizeof(PointType));
        checkCUDAError("malloc dps error");
        cudaMemcpy(dev_pos_surface, &(_surface->points[0]), _N_surface * sizeof(PointType), cudaMemcpyHostToDevice);
        checkCUDAError("memcpy ps error");
        cudaMalloc((void**)&dev_grid_indices,_N_surface * sizeof(int));
        checkCUDAError("malloc gi failed");
        cudaMemcpy(dev_grid_indices, &(*_grid_indices)[0], _N_surface * sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError("memcpy gi failed");


    }

}