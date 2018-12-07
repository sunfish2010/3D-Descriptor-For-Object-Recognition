#pragma once

#include "common.h"

#include "cudaCommon.h"



class Grid{
public:
    Grid()= default;
    ~Grid();
    void setRadius(float radius){this->radius = radius;}
    void computeSceneProperty(const pcl::PointCloud<PointType >::ConstPtr &input,
            const IndicesPtr &grid_indices, const IndicesPtr &array_indices);
    inline Eigen::Vector4i getSceneMin() const { return min_pi; }
    inline Eigen::Vector4i getSceneMax()const {return max_pi;}
    inline Eigen::Vector4f getInverseRadius() const {return inv_radius;}
    inline Eigen::Vector4i getDimension()const {return pc_dimension;}

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    float radius=0.f;
    int N=0;
    PointType *dev_pc=NULL;
    int *dev_grid_indices=NULL;
    int *dev_array_indices=NULL;

    // property to grab
    Eigen::Vector4i min_pi;
    Eigen::Vector4i max_pi;
    Eigen::Vector4f inv_radius;
    Eigen::Vector4i pc_dimension;

};