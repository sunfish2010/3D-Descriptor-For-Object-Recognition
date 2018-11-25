#include "descriptor.h"


template <typename OutType>
void Descriptor<OutType>::compute(PointCloudOutPtr &output) {
    if (_normals == nullptr) {
        std::cerr << "normals has not been set yet, abort" <<std::endl;
        exit(1);
    }

    if (_input == nullptr){
        std::cerr << "input cloud has not been set yet" << std::endl;
        exit(1);
    }

    if(_surface == nullptr){
        std::cerr << "cloud surface has not been set yet" << std::endl;
        exit(1);
    }
    output->header = _input->header;
    output->points.resize(_input->points.size());
    output->width= _input->width;
    output->height= _input->height;
    output->is_dense = _input->is_dense;

}

template <typename OutType>
void SHOT<OutType>::computeDescriptor(PointCloudOutPtr &output) {
    descLength_ = nr_grid_sector_ * (nr_shape_bins_ + 1);

    sqradius_ = _radius * _radius;
    radius3_4_ = (_radius * 3) / 4;
    radius1_4_ = _radius / 4;
    radius1_2_ = _radius / 2;

    assert(descLength_ == 352);
}

