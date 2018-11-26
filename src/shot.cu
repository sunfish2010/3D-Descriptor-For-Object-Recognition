#include "shot.h"


void SHOT::computeDescriptor(const pcl::PointCloud<pcl::SHOT352> &output) {
    descLength_ = nr_grid_sector_ * (nr_shape_bins_ + 1);

    sqradius_ = _radius * _radius;
    radius3_4_ = (_radius * 3) / 4;
    radius1_4_ = _radius / 4;
    radius1_2_ = _radius / 2;

    assert(descLength_ == 352);
}
