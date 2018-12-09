#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "descriptor.h"

#include "cudaCommon.h"

class SHOT_LRF:public Descriptor<pcl::ReferenceFrame>{
public:
    //typedef pcl::PointCloud<OutType> PointCloudOut;
    //typedef typename PointCloudOut::Ptr PointCloudOutPtr;

    using Descriptor<pcl::ReferenceFrame>::_radius;
    using Descriptor<pcl::ReferenceFrame>::_k;
    using Descriptor<pcl::ReferenceFrame>::_normals;
    using Descriptor<pcl::ReferenceFrame>::_input;
    using Descriptor<pcl::ReferenceFrame>::_surface;
//    using Descriptor<pcl::ReferenceFrame>::_neighbor_indices;
    using Descriptor<pcl::ReferenceFrame>::_kept_indices;

    SHOT_LRF()=default;
    ~SHOT_LRF()override= default;

protected:

    void computeDescriptor(pcl::PointCloud<pcl::ReferenceFrame> &output,  const Eigen::Vector4f &inv_radius,
                           const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi) override;


private:
//    int *dev_neighbor_indices;
//    int *dev_grid_indices;


    int _N_features;
    int _N_surface;

    std::vector<int> _num_neighbors;
//    std::vector<double> _sum;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> _covs;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> _vij;
    std::vector<int> _neighbor_indices;
    std::vector<float> _neighbor_distances;
    /** \brief inner max neighbor to keep */
    const int _n = 128;

};