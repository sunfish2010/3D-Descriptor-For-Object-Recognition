#pragma once
#include "common.h"


// descriptor template class to allow different types of descriptors

template <typename OutType>
class Descriptor{
public:
    typedef pcl::PointCloud<OutType> PointCloudOut;
    //typedef typename PointCloudOut::Ptr PointCloudOutPtr;
    Descriptor()= default;
    explicit Descriptor(float radius): _radius(radius){};
    virtual ~Descriptor(){
        _input.reset();
        _kept_indices.reset();
        _array_indices.reset();
        _normals.reset();
        _surface.reset();
    };
    inline void setRadius(float radius){_radius = radius;}
    inline void setNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr &normals){_normals = normals;}
    inline void setSurface(const pcl::PointCloud<PointType>::ConstPtr &surface){_surface = surface;}
    inline void setInputCloud(const pcl::PointCloud<PointType>::ConstPtr &input){_input = input;}
//    inline void setFeatureNeighborsIndices(const IndicesPtr &neighbor_indices){_neighbor_indices = neighbor_indices;}
    inline void setKeptIndices(const IndicesConstPtr &kept_indices){_kept_indices = kept_indices;}
    inline void setArrayIndices(const IndicesConstPtr &array_indices){_array_indices = array_indices;}

    void compute(PointCloudOut &output,  const Eigen::Vector4f &inv_radius, const Eigen::Vector4i &pc_dimension,
            const Eigen::Vector4i &min_pi,  const Eigen::Vector4i &max_pi,
            const IndicesConstPtr &grid_start_indices, const IndicesConstPtr &grid_end_indices);

protected:
    float _radius;
    const int _k = 15;
    pcl::PointCloud<PointType>::ConstPtr _input;
    pcl::PointCloud<pcl::Normal>::ConstPtr _normals;
    pcl::PointCloud<PointType>::ConstPtr _surface;
    IndicesConstPtr _array_indices;
    IndicesConstPtr _kept_indices;
private:
    bool initialized();
    virtual void computeDescriptor(PointCloudOut &output,  const Eigen::Vector4f &inv_radius,
                                   const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi, const Eigen::Vector4i &max_pi,
                                   const IndicesConstPtr &grid_start_indices, const IndicesConstPtr &grid_end_indices) = 0;
};

template <typename OutType>
void Descriptor<OutType>::compute(PointCloudOut &output,  const Eigen::Vector4f &inv_radius,
                                  const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi,const Eigen::Vector4i &max_pi,
                                  const IndicesConstPtr &grid_start_indices, const IndicesConstPtr &grid_end_indices) {
   if (!initialized()){
       std::cerr << "descriptor has not been correctly initialized " <<std::endl;
       exit(1);
   }
//    output.header = _input->header;
    output.points.resize(_input->points.size());
    output.width= _input->width;
    output.height= _input->height;
    output.is_dense = _input->is_dense;
    computeDescriptor(output, inv_radius, pc_dimension, min_pi, max_pi, grid_start_indices, grid_end_indices);

}

template <typename OutType>
bool Descriptor<OutType>::initialized() {
    return (_input != nullptr && _normals != nullptr && _surface != nullptr && _radius >0 && _kept_indices!= nullptr);
}

