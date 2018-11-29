#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "cuda_common.h"


class Search;
class UniformDownSample{
public:
    UniformDownSample(){
        cudaMalloc((void**)&dev_min, sizeof(Eigen::Vector4f));
        cudaMalloc((void**)&dev_max, sizeof(Eigen::Vector4f));
        checkCUDAError("cudaMalloc min,max");
    }
    ~UniformDownSample();
    inline void setRadius(float radius){this->radius = radius;}
    //void downSample(const pcl::PointCloud<PointType >::ConstPtr input);
    void downSample(const pcl::PointCloud<PointType >::ConstPtr &input);
    IndicesPtr getKeptIndices();
    //void getGridIndices(IndicesPtr &indices);
    void fillOutput(pcl::PointCloud<PointType>::Ptr &output);
//    inline void setOutput(const pcl::PointCloud<PointType>::Ptr &output){_output  = output;}

private:
    /** \brief for grid set up */
    int N;
    int N_new;
    float radius;
    Eigen::Vector4f *dev_min;
    Eigen::Vector4f *dev_max;
    Eigen::Vector4i min_pi, max_pi;
    Eigen::Vector4f inv_radius;
    Eigen::Vector4i pc_dimension;

    int *dev_grid_indices;
    int *dev_kept_indices;
    PointType *dev_pos_surface;
    int _grid_count_max;
    float *dev_min_dist;
    float *dev_dist;

    IndicesPtr kept_indices;
//    IndicesPtr grid_indices;
//    pcl::PointCloud<PointType>::Ptr _output;
    pcl::PointCloud<PointType>::ConstPtr _input;

};

//namespace UniformSample{
//
//    void UniformDownSample(int& N, const pcl::PointCloud<PointType >::ConstPtr input, bool mode);
//}