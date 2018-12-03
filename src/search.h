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

    inline void setFeaturesIndices(const IndicesConstPtr &input){
        _feature_indices = input;
    }

//    inline void setGridIndices(const IndicesPtr &input){
//        _grid_indices = input;
//    }

    void search( const Eigen::Vector4f &inv_radius,
            const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_p);

    inline IndicesConstPtr getNumNeighbors(){return IndicesConstPtr(&_num_neighbors);}
    inline IndicesConstPtr getNeighborIndices(){return IndicesConstPtr(&_neighbor_indices);}
    inline boost::shared_ptr<const std::vector<float>> getNeighborDistance(){
        return boost::shared_ptr<const std::vector<float>>(&_neighbor_distances);}

private:
    float _radius;
    int _k;
    SearchMethod _method;
    pcl::PointCloud<PointType>::ConstPtr _input;
    pcl::PointCloud<PointType>::ConstPtr _surface;
//    IndicesPtr _grid_indices;
    IndicesConstPtr _feature_indices;
    int *dev_neighbor_indices;
    int *dev_features_indices;
//    int *dev_grid_indices;
    PointType *dev_pos_surface;
    int *dev_num_neighbors;
    float *dev_distances;
    int _N_features;
    int _N_surface;

    std::vector<int> _num_neighbors;
    std::vector<int> _neighbor_indices;
    std::vector<float> _neighbor_distances;
    /** \brief inner max neighbor to keep */
    const int _n = 16;

};
