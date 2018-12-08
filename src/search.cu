#include "search.h"



Search::~Search() {

//    cudaFree(dev_grid_indices);
    dev_neighbor_indices = NULL;

}


void Search::setInputCloud(const pcl::PointCloud<pcl::SHOT352>::Ptr &input) {
    _input = input;
    _N_input = static_cast<int>(input->points.size());
}

void Search::search(const pcl::CorrespondencesPtr &model_scene_corrs) {
    if (!_surface || !_input || _N_input > _N_surface){
        std::cerr << "Search function not properly setup" << std::endl;
        exit(1);
    }

    cudaMalloc((void**)&dev_neighbor_indices, _N_surface * sizeof(int));
    checkCUDAError("malloc dev_neighbor indices error");
    cudaMemset(dev_neighbor_indices, -1, _N_surface * sizeof(int));
    checkCUDAError("memset ni error");

    cudaMalloc((void**)&dev_input, _N_input * sizeof(pcl::SHOT352));
    checkCUDAError("malloc dev_neighbor distances error");
    cudaMemcpy(dev_input, &(_input->points[0]), _N_input * sizeof(pcl::SHOT352), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_surface, _N_surface * sizeof(pcl::SHOT352));
    checkCUDAError("malloc dps error");
    cudaMemcpy(dev_surface, &(_surface->points[0]), _N_surface * sizeof(pcl::SHOT352), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy ps error");

    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((_N_surface + blockSize - 1)/blockSize));

//    kernSearchCorrespondence

    checkCUDAError("KernSearchCorres error");


    _neighbor_indices.resize(_N_surface);
    cudaMemcpy(&_neighbor_indices[0], dev_neighbor_indices, sizeof(int) * _N_surface, cudaMemcpyDeviceToHost);
    checkCUDAError("cudamemcpy  num neigbors issue");


    for(auto&n: _neighbor_indices)
        std::cout << n << std::endl;

    cudaFree(dev_surface);
    cudaFree(dev_input);
    cudaFree(dev_neighbor_indices);


}