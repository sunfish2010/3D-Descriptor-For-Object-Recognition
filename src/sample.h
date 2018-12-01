#pragma once

#include "common.h"

#include "cudaCommon.h"



class UniformDownSample{
public:
    UniformDownSample()= default;
    ~UniformDownSample();
//    void setRadius(float radius);
    //void downSample(const pcl::PointCloud<PointType >::ConstPtr input);
    void downSample(const pcl::PointCloud<PointType >::ConstPtr &input, pcl::PointCloud<PointType>::Ptr &output,
                    const IndicesPtr &grid_indices, const IndicesPtr &array_indices,
                    const Eigen::Vector4f &inv_radius);
    void randDownSample(const pcl::PointCloud<PointType >::ConstPtr &input, pcl::PointCloud<PointType>::Ptr &output);
    inline void setKeptIndicesPtr(const IndicesPtr &indices){ kept_indices = indices; }

private:
//    float radius=0.f;
//    int N_new=0;
    int N=0;
    int *dev_grid_indices=NULL;
    int *dev_array_indices=NULL;
    PointType *dev_new_pc=NULL;
    int *dev_tmp=NULL;
    PointType *dev_pc=NULL;

    IndicesPtr kept_indices;

};

//namespace UniformSample{
//
//    void UniformDownSample(int& N, const pcl::PointCloud<PointType >::ConstPtr input, bool mode);
//}