#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

#include "checkCUDAError.h"



class UniformDownSample{
public:
    explicit UniformDownSample(float radius);
    ~UniformDownSample();
    void setRadius(float radius);
    //void downSample(const pcl::PointCloud<PointType >::ConstPtr input);
    void downSample(const pcl::PointCloud<PointType >::ConstPtr &input, pcl::PointCloud<PointType>::Ptr &output);
    inline void setKeptIndicesPtr(const IndicesPtr &indices){ kept_indices = indices; }
    inline void setGridIndicesPtr(const IndicesPtr &indices){ grid_indices = indices; }

private:
    float radius;
    int N_new;
    int N;
    Eigen::Vector4f *dev_min;
    Eigen::Vector4f *dev_max;
    int *dev_grid_indices;
    int *dev_array_indices;
    PointType *dev_new_pc;
    int *dev_tmp;
    PointType *dev_pc;

    IndicesPtr kept_indices;
    IndicesPtr grid_indices;

};

//namespace UniformSample{
//
//    void UniformDownSample(int& N, const pcl::PointCloud<PointType >::ConstPtr input, bool mode);
//}