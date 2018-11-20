#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

#include <boost/version.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/sync/file_lock.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include "checkCUDAError.h"

typedef pcl::PointXYZRGB PointType;

class UniformDownSample{
public:
    UniformDownSample(float radius);
    ~UniformDownSample();
    void setRadius(float radius);
    void downSample(const pcl::PointCloud<PointType >::ConstPtr input);
    void downSample(const pcl::PointCloud<PointType >::ConstPtr input, pcl::PointCloud<PointType>::Ptr output);

private:
    float radius;
    int N_new;
    int N;
    Eigen::Vector4f *dev_min;
    Eigen::Vector4f *dev_max;
};

//namespace UniformSample{
//
//    void UniformDownSample(int& N, const pcl::PointCloud<PointType >::ConstPtr input, bool mode);
//}