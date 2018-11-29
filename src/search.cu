#include "search.h"



/** \brief 3d indices to 1d  */
__device__ int kernComputeIndices(Eigen::Vector4i pos, Eigen::Vector4i grid_res){
    return pos[0] + pos[1] * grid_res[0] + pos[2] * grid_res[1] * grid_res[2];
}


/** \brief common functions for search  */
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



__global__ void kernSearchRadius(int N, int *feature_indices, int * output, PointType *surface) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < N){

    }
}


Search::~Search() {

    cudaFree(dev_neighbor_indices);
    cudaFree(dev_features_indices);
//
    dev_neighbor_indices = NULL;
    dev_features_indices = NULL;
//    dev_pos_surface = NULL;
//    dev_array_indices = NULL;
//    dev_grid_indices = NULL;
//    dev_max = NULL;
//    dev_min = NULL;
}


void Search::initSearch(float radius) {
    if (!_surface){
        std::cerr << "Must set up surface input "  << std::endl;
        exit(1);
    }




}


//void get


void Search::search(const pcl::PointCloud<PointType>::Ptr &output) {
    if (_method == SearchMethod::KDTree) {
        std::cout << "Function not implemented yet" << std::endl;
        exit(1);
    }
    if (_method == SearchMethod::Radius){
//        cudaMalloc((void**)&dev_features_indices, _N_features * sizeof(int));
//        checkCUDAError("mallod dev_features_indices error");
//        cudaMemcpy(dev_features_indices, &(*_feature_indices)[0], _N_features * sizeof(int), cudaMemcpyHostToDevice);
//        checkCUDAError("memcpy dev_features_indices error");
//        cudaMalloc((void**)&dev_neighbor_indices, _N_features * _n * sizeof(int));
//        checkCUDAError("malloc dev_neighbor indices error");
//        cudaMemset(dev_neighbor_indices, -1, _N_features * _n * sizeof(int));
//        checkCUDAError("memset ni error");
//        cudaMalloc((void**)&dev_pos_surface, _N_surface * sizeof(PointType));
//        checkCUDAError("malloc dps error");
//        cudaMemcpy(dev_pos_surface, &(_surface->points[0]), _N_surface * sizeof(PointType), cudaMemcpyHostToDevice);
//        checkCUDAError("memcpy ps error");
//        cudaMalloc((void**)&dev_grid_indices,_N_surface * sizeof(int));
//        checkCUDAError("malloc gi failed");
//        cudaMemcpy(dev_grid_indices, &(*_grid_indices)[0], _N_surface * sizeof(int), cudaMemcpyHostToDevice);
//        checkCUDAError("memcpy gi failed");

    }

}