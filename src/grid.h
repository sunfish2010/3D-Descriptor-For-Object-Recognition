#pragma once

#include "common.h"
#include "boost/make_shared.hpp"
#include "cudaCommon.h"



class Grid{
public:
    Grid()= default;
    ~Grid()= default;
    void setRadius(float radius){this->radius = radius;}
    void computeSceneProperty(const pcl::PointCloud<PointType >::ConstPtr &input,
            const IndicesPtr &grid_indices, const IndicesPtr &array_indices, Eigen::Vector4f &inv_radius,
            Eigen::Vector4i &pc_dimension, Eigen::Vector4i &min_pi, Eigen::Vector4i &max_pi);
    IndicesConstPtr getCellStart(){return boost::make_shared<const std::vector<int>>(cellStartIndices);}
    IndicesConstPtr getCellEnd(){return boost::make_shared<const std::vector<int>>(cellEndIndices);}
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    float radius=0.f;
    int N=0;
    std::vector<int> cellStartIndices;
    std::vector<int> cellEndIndices;


};