#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "cudaCommon.h"
#include "kdtree.hpp"


class Search{
public:

    explicit Search(int k):_k(k){};
    Search():_k(1), dev_input(NULL),
             dev_neighbor_indices(NULL), dev_surface(NULL), _N_input(0), _N_surface(0){};
    ~Search();

    inline void setK(int k){
        _k = k;
    }

    void setInputCloud(const pcl::PointCloud<pcl::SHOT352>::Ptr &input);

    inline void setSurfaceCloud(const pcl::PointCloud<pcl::SHOT352>::Ptr &input){
        _surface = input;
        _N_surface = static_cast<int>(input->points.size());
    }


    void search(const pcl::CorrespondencesPtr &model_scene_corrs);

    inline IndicesConstPtr getNeighborIndices(){return IndicesConstPtr(&_neighbor_indices);}


private:
    int _k;
    pcl::PointCloud<pcl::SHOT352>::ConstPtr _input;
    pcl::PointCloud<pcl::SHOT352>::ConstPtr _surface;

    int *dev_neighbor_indices;
    pcl::SHOT352 *dev_surface;
    pcl::SHOT352 *dev_input;
    int _N_input;
    int _N_surface;

    std::vector<int> _neighbor_indices;

};
