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
    UniformDownSample()= default;
    ~UniformDownSample();
    //void setRadius(float radius);
    //void downSample(const pcl::PointCloud<PointType >::ConstPtr input);
    void downSample(const pcl::PointCloud<PointType >::ConstPtr &input, Search& tool);
    IndicesPtr getKeptIndices();
    //void getGridIndices(IndicesPtr &indices);
    void fillOutput(pcl::PointCloud<PointType>::Ptr &output);
//    inline void setOutput(const pcl::PointCloud<PointType>::Ptr &output){_output  = output;}

private:
    int N_new;
    int N;

    int *dev_kept_indices;
    PointType *dev_new_pc;
    PointType *dev_pc;

    IndicesPtr kept_indices;
//    IndicesPtr grid_indices;
//    pcl::PointCloud<PointType>::Ptr _output;
    pcl::PointCloud<PointType>::ConstPtr _input;

};

//namespace UniformSample{
//
//    void UniformDownSample(int& N, const pcl::PointCloud<PointType >::ConstPtr input, bool mode);
//}