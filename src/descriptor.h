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
    inline void setFeatureIndices(const IndicesPtr &feature_indices){_feature_indices = feature_indices;}
    inline void setGridIndices(const IndicesPtr &grid_indices){_grid_indices = grid_indices;}

    void compute(const PointCloudOut &output);

protected:
    float _radius;
    pcl::PointCloud<PointType>::ConstPtr _input;
    pcl::PointCloud<pcl::Normal>::ConstPtr _normals;
    pcl::PointCloud<PointType>::ConstPtr _surface;
    IndicesPtr _feature_indices;
    IndicesPtr _grid_indices;
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
    return (_input != nullptr && _normals != nullptr && _surface != nullptr
    && _feature_indices != nullptr && _grid_indices != nullptr);
}

