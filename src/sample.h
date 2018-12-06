#pragma once

#include "common.h"
#include "cudaCommon.h"


class Search;
class UniformDownSample{
public:
    UniformDownSample()= default;
    ~UniformDownSample();
//    void downSample(const pcl::PointCloud<PointType >::ConstPtr &input, pcl::PointCloud<PointType>::Ptr &output,
//                    IndicesPtr &kept_indices, const IndicesPtr &grid_indices, const IndicesPtr &array_indices,
//                    const Eigen::Vector4f &inv_radius);
//    void randDownSample(const pcl::PointCloud<PointType >::ConstPtr &input, pcl::PointCloud<PointType>::Ptr &output);
    void downSampleAtomic(const pcl::PointCloud<PointType >::ConstPtr &input, const Eigen::Vector4f &inv_radius,
            const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi);

    void display(const pcl::PointCloud<PointType >::ConstPtr &input, const pcl::PointCloud<PointType>::Ptr &output);

    inline IndicesConstPtr getKeptIndice() {return boost::make_shared<const std::vector<int>>(kept_indices);}
    inline IndicesConstPtr getGridIndices(){return IndicesConstPtr(&grid_indices);}


private:
//    float radius=0.f;
    int N_new=0;
    int N=0;
    int *dev_grid_indices=NULL;
    int *dev_array_indices=NULL;
    int *dev_kept_indices = NULL;
    float *dev_min_dist = NULL;
    float *dev_dist = NULL;
    PointType *dev_new_pc=NULL;
    int *dev_tmp=NULL;
    PointType *dev_pc=NULL;
    std::vector<int> kept_indices;
    std::vector<int> grid_indices;

};

//namespace UniformSample{
//
//    void UniformDownSample(int& N, const pcl::PointCloud<PointType >::ConstPtr input, bool mode);
//}