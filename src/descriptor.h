#pragma once
#include "common.h"


// similar design to pcl structure for compatibility

template <typename OutType>
class Descriptor{
public:
    typedef pcl::PointCloud<OutType> PointCloudOut;
    //typedef typename PointCloudOut::Ptr PointCloudOutPtr;
    Descriptor()= default;
    explicit Descriptor(float radius): _radius(radius){};
    virtual ~Descriptor(){};
    inline void setRadius(float radius){_radius = radius;}
    inline void setNormals(pcl::PointCloud<pcl::Normal>::ConstPtr normals){_normals = normals;}
    inline void setSurface(pcl::PointCloud<PointType>::ConstPtr surface){_surface = surface;}
    inline void setInputCloud(pcl::PointCloud<PointType>::ConstPtr input){_input = input;}
    inline void setFeatureNeighborsIndices(const IndicesPtr &neighbor_indices){_neighbor_indices = neighbor_indices;}
    inline void setKeptIndices(const IndicesPtr &kept_indices){_kept_indices = kept_indices;}

    void compute(const PointCloudOut &output);

protected:
    float _radius;
    const int _k = 15;
    pcl::PointCloud<PointType>::ConstPtr _input;
    pcl::PointCloud<pcl::Normal>::ConstPtr _normals;
    pcl::PointCloud<PointType>::ConstPtr _surface;
    IndicesPtr _neighbor_indices;
    IndicesPtr _kept_indices;
private:
    bool initialized();
    virtual void computeDescriptor(const PointCloudOut &output) = 0;
};

template <typename OutType>
void Descriptor<OutType>::compute(const PointCloudOut &output) {
   if (!initialized()){
       std::cerr << "descriptor has not been correctly initialized " <<std::endl;
       exit(1);
   }
    output.header = _input->header;
    output.points.resize(_input->points.size());
    output.width= _input->width;
    output.height= _input->height;
    output.is_dense = _input->is_dense;

}

template <typename OutType>
bool Descriptor<OutType>::initialized() {
    return (_input != nullptr && _normals != nullptr && _surface != nullptr);
}

