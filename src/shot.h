#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "descriptor.h"
#include <pcl/features/shot_lrf_omp.h>
#include "cudaCommon.h"

class SHOT352:public Descriptor<pcl::SHOT352>{
public:

    using Descriptor<pcl::SHOT352>::_radius;
    using Descriptor<pcl::SHOT352>::_normals;
    using Descriptor<pcl::SHOT352>::_input;
    using Descriptor<pcl::SHOT352>::_surface;
    using Descriptor<pcl::SHOT352>::_kept_indices;

    explicit SHOT352(int nr_shape_bins = 10, int nr_color_bins = 30) :Descriptor<pcl::SHOT352>(0.01),
                                  nr_shape_bins_ (nr_shape_bins),nr_color_bins_(nr_color_bins),
                                  lrf_radius_ (0),
                                  nr_grid_sector_ (32),
                                  maxAngularSectors_ (32),
                                  descLength_ (0){};

    ~SHOT352()override {
        _input.reset();
        _kept_indices.reset();
        _normals.reset();
        _surface.reset();
    }


protected:

    void computeDescriptor(pcl::PointCloud<pcl::SHOT352> &output,  const Eigen::Vector4f &inv_radius,
                           const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi) override;


    /** \brief size of color & shape histogram  */
    int nr_shape_bins_, nr_color_bins_;

    /** \brief The radius used for the LRF computation */
    float lrf_radius_;

    /** \brief Number of azimuthal sectors. */
    const int nr_grid_sector_;

    /** \brief ... division */
    const int maxAngularSectors_;

    /** \brief One SHOT length. */
    int descLength_;

};
