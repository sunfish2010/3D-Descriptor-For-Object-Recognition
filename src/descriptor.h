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

    void compute(PointCloudOut &output);

protected:
    float _radius;
    pcl::PointCloud<PointType>::ConstPtr _input;
    pcl::PointCloud<pcl::Normal>::ConstPtr _normals;
    pcl::PointCloud<PointType>::ConstPtr _surface;
private:
    virtual void computeDescriptor(PointCloudOut &output) = 0;
};

template <typename OutType>
void Descriptor<OutType>::compute(PointCloudOut &output) {
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
    output.header = _input->header;
    output.points.resize(_input->points.size());
    output.width= _input->width;
    output.height= _input->height;
    output.is_dense = _input->is_dense;

}

