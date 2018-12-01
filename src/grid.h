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
    inline Eigen::Vector4i getSceneMin() const { return min; }
    inline Eigen::Vector4i getSceneMax()const {return max;}
    inline Eigen::Vector4f getInverseRadius() const {return inv_radius;}


private:
    float radius=0.f;
    int N=0;
    PointType *dev_pc=NULL;
    Eigen::Vector4f *dev_min=NULL;
    Eigen::Vector4f *dev_max=NULL;
    int *dev_grid_indices=NULL;
    int *dev_array_indices=NULL;;

    // property to grab
    Eigen::Vector4i min;
    Eigen::Vector4i max;
    Eigen::Vector4f inv_radius;
    Eigen::Vector4i pc_dimension;

};