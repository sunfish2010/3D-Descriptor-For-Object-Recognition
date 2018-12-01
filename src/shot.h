#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "descriptor.h"
#include "cudaCommon.h"

class SHOT:public Descriptor<pcl::SHOT352>{
public:
    //typedef pcl::PointCloud<OutType> PointCloudOut;
    //typedef typename PointCloudOut::Ptr PointCloudOutPtr;

    using Descriptor<pcl::SHOT352>::_radius;
    using Descriptor<pcl::SHOT352>::_normals;
    using Descriptor<pcl::SHOT352>::_input;
    using Descriptor<pcl::SHOT352>::_surface;
    using Descriptor<pcl::SHOT352>::_neighbor_indices;
    using Descriptor<pcl::SHOT352>::_kept_indices;

    SHOT(int nr_shape_bins = 10) :Descriptor<pcl::SHOT352>(0.01),
                                  nr_shape_bins_ (nr_shape_bins),
                                  shot_ (), lrf_radius_ (0),
                                  sqradius_ (0), radius3_4_ (0), radius1_4_ (0), radius1_2_ (0),
                                  nr_grid_sector_ (32),
                                  maxAngularSectors_ (32),
                                  descLength_ (0){};
    virtual ~SHOT()override= default;

    inline void setLRFPtr(const pcl::PointCloud<pcl::ReferenceFrame> &lrf){_lrf = lrf;}

private:
    pcl::PointCloud<pcl::ReferenceFrame> _lrf;

protected:

    void computeDescriptor(const pcl::PointCloud<pcl::SHOT352> &output);



    int nr_shape_bins_;
    /** \brief Placeholder for a point's SHOT. */
    Eigen::VectorXf shot_;

    float lrf_radius_;
    /** \brief The radius used for the LRF computation */

    double sqradius_;
    /** \brief 3/4 of the search radius. */
    double radius3_4_;

    /** \brief 1/4 of the search radius. */
    double radius1_4_;

    /** \brief 1/2 of the search radius. */
    double radius1_2_;

    /** \brief Number of azimuthal sectors. */
    const int nr_grid_sector_;

    /** \brief ... */
    const int maxAngularSectors_;

    /** \brief One SHOT length. */
    int descLength_;

};