#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "descriptor.h"
#include "cuda_common.h"

class SHOT_LRF:public Descriptor<pcl::ReferenceFrame>{
public:
    //typedef pcl::PointCloud<OutType> PointCloudOut;
    //typedef typename PointCloudOut::Ptr PointCloudOutPtr;

    using Descriptor<pcl::ReferenceFrame>::_radius;
    using Descriptor<pcl::ReferenceFrame>::_normals;
    using Descriptor<pcl::ReferenceFrame>::_input;
    using Descriptor<pcl::ReferenceFrame>::_surface;
    using Descriptor<pcl::ReferenceFrame>::_feature_indices;
    using Descriptor<pcl::ReferenceFrame>::_grid_indices;

    SHOT_LRF()=default;
    ~SHOT_LRF()override= default;

protected:

    void computeDescriptor(const pcl::PointCloud<pcl::ReferenceFrame> &output);





};