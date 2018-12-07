#include "search.h"


__global__ void kernSearchRadius(int N, int n, const int max_neighbor, const PointType *surface, float radius,
        const int *feature_indices, const Eigen::Vector4f inv_radius, const Eigen::Vector4i min_pi,
        int * neighbor_indices, int* num_neighbors, float* dist){

}

Search::~Search() {

//    cudaFree(dev_grid_indices);
    dev_neighbor_indices = NULL;
    dev_features_indices = NULL;
    dev_pos_surface = NULL;
}

void Search::search(const Eigen::Vector4f &inv_radius,
        const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_p) {
    if (!_surface || !_input || !_feature_indices || _N_features > _N_surface){
        std::cerr << "Search function not properly setup" << std::endl;
        exit(1);
    }

    if (_method == SearchMethod::KDTree) {
        std::cout << "Function not implemented yet" << std::endl;
        exit(1);
    }
    if (_method == SearchMethod::Radius){
        if(_radius == 0) {
            std::cerr << "Search function not properly setup" << std::endl;
            exit(1);
        }

        // computing indices to search

        cudaMalloc((void**)&dev_features_indices, _N_features * sizeof(int));
        checkCUDAError("mallod dev_features_indices error");
        cudaMemcpy(dev_features_indices, &(*_feature_indices)[0], _N_features * sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError("memcpy dev_features_indices error");
        cudaMalloc((void**)&dev_neighbor_indices, _N_features * _n * sizeof(int));
        checkCUDAError("malloc dev_neighbor indices error");
        cudaMemset(dev_neighbor_indices, -1, _N_features * _n * sizeof(int));
        checkCUDAError("memset ni error");
        cudaMalloc((void**)&dev_distances, _N_features * _n * sizeof(int));
        checkCUDAError("malloc dev_neighbor distances error");

        cudaMalloc((void**)&dev_pos_surface, _N_surface * sizeof(PointType));
        checkCUDAError("malloc dps error");
        cudaMemcpy(dev_pos_surface, &(_surface->points[0]), _N_surface * sizeof(PointType), cudaMemcpyHostToDevice);
        checkCUDAError("memcpy ps error");
//        cudaMalloc((void**)&dev_grid_indices,_N_surface * sizeof(int));
//        checkCUDAError("malloc gi failed");
//        cudaMemcpy(dev_grid_indices, &(*_grid_indices)[0], _N_surface * sizeof(int), cudaMemcpyHostToDevice);
//        checkCUDAError("memcpy gi failed");
        cudaMalloc((void**)&dev_num_neighbors, _N_features * sizeof(int));
        checkCUDAError("malloc num neighbors error");
        cudaMemset(dev_num_neighbors, 0, sizeof(int) * _N_features);
        checkCUDAError("memset num neighbors error");

        dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((_N_surface + blockSize - 1)/blockSize));

        kernSearchRadius<<<fullBlockPerGrid_points, blockSize, _N_features * sizeof(u_int8_t) * 6>>> (_N_surface, _N_features,
                _n, dev_pos_surface, _radius, dev_features_indices, inv_radius, min_p, dev_neighbor_indices, dev_num_neighbors,
                dev_distances);
        checkCUDAError("KernSearchRadius error");

        _num_neighbors.resize(_N_features);
        cudaMemcpy(&_num_neighbors[0], dev_num_neighbors, sizeof(int) * _N_features, cudaMemcpyDeviceToHost);
        checkCUDAError("cudamemcpy  num neigbors issue");

        _neighbor_indices.resize(_n * _N_features);
        cudaMemcpy(&_neighbor_indices[0], dev_neighbor_indices, sizeof(int) * _N_features * _n, cudaMemcpyDeviceToHost);
        checkCUDAError("cudamemcpy  num neigbors issue");

        _neighbor_distances.resize(_n * _N_features);
        cudaMemcpy(&_neighbor_distances[0], dev_distances, sizeof(float) * _N_features * _n, cudaMemcpyDeviceToHost);
        checkCUDAError("cudamemcpy  distances issue");

        for(auto&n: _num_neighbors)
            std::cout << n << std::endl;

        cudaFree(dev_pos_surface);
        cudaFree(dev_neighbor_indices);
        cudaFree(dev_features_indices);
        cudaFree(dev_num_neighbors);

    }

}