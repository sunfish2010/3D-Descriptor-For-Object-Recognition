#pragma once

#include "common.h"

#include "cudaCommon.h"



class Grid{
public:
    Grid()= default;
    ~Grid()= default;
    void setRadius(float radius){this->radius = radius;}
    void computeSceneProperty(const pcl::PointCloud<PointType >::ConstPtr &input,
            const IndicesPtr &grid_indices, const IndicesPtr &array_indices, Eigen::Vector4f &inv_radius,
            Eigen::Vector4i &pc_dimension, Eigen::Vector4i &min_pi);
    // property to grab


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    float radius=0.f;
    int N=0;



};