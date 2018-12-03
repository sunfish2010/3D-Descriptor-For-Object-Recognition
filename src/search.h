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
    enum SearchMethod{
        KDTree,
        Radius
    };

    explicit Search(float radius):_radius(radius), _k(0), _method(SearchMethod::Radius){};
    explicit Search(int k):_k(k), _radius(0), _method(SearchMethod::KDTree){};
    Search():_radius(0), _k(0), _method(SearchMethod::Radius), dev_features_indices(NULL),
             dev_neighbor_indices(NULL), dev_pos_surface(NULL), _N_features(0), _N_surface(0){};
    ~Search();

    inline void setRadius(float radius){
        _radius = radius;
        _method = SearchMethod::Radius;
    }

    inline void setK(int k){
        _k = k;
        _method = SearchMethod ::KDTree;
    }

    inline void setFeatures(const pcl::PointCloud<PointType>::ConstPtr &input){
        _input = input;
        _N_features = static_cast<int>(input->points.size());
    }

    inline void setSurface(const pcl::PointCloud<PointType>::ConstPtr &surface){
        _surface = surface;
        _N_surface = static_cast<int >(surface->points.size());
    }

    inline void setFeaturesIndices(const IndicesPtr &input){
        _feature_indices = input;
    }

    inline void setGridIndices(const IndicesPtr &input){
        _grid_indices = input;
    }

    void search(const pcl::PointCloud<PointType>::Ptr &output);

private:
    float _radius;
    int _k;
    SearchMethod _method;
    pcl::PointCloud<PointType>::ConstPtr _input;
    pcl::PointCloud<PointType>::ConstPtr _surface;
    IndicesPtr _grid_indices;
    IndicesPtr _feature_indices;
    int *dev_neighbor_indices;
    int *dev_features_indices;
    int *dev_grid_indices;
    int *dev_pos_surface;
    int _N_features;
    int _N_surface;
    /** \brief inner max neighbor to keep */
    const int _n = 15;

};
