#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

#include "checkCUDAError.h"

// similar design to pcl structure for compatibility

template <typename OutType>
class Descriptor{
public:
    typedef pcl::PointCloud<OutType> PointCloudOut;
    typedef typename PointCloudOut::Ptr PointCloudOutPtr;

    explicit Descriptor(float radius): _radius(radius){};
    virtual ~Descriptor(){};
    inline void setRadius(float radius){_radius = radius;}
    inline void setNormals(pcl::PointCloud<pcl::Normal>::ConstPtr &normals){_normals = normals;}
    inline void setSurface(pcl::PointCloud<PointType>::ConstPtr &surface){_surface = surface;}
    inline void setInputCloud(pcl::PointCloud<PointType>::ConstPtr &input){_input = input;}

    void compute(PointCloudOutPtr &output);

protected:
    float _radius;
    pcl::PointCloud<PointType>::ConstPtr _input;
    pcl::PointCloud<pcl::Normal>::ConstPtr _normals;
    pcl::PointCloud<PointType>::ConstPtr _surface;

    virtual void computeDescriptor(PointCloudOutPtr &output) = 0;
};

template <typename OutType = pcl::SHOT352>
class SHOT:public Descriptor<OutType>{
public:
    typedef pcl::PointCloud<OutType> PointCloudOut;
    typedef typename PointCloudOut::Ptr PointCloudOutPtr;

    using Descriptor<OutType>::_radius;
    using Descriptor<OutType>::_normals;
    using Descriptor<OutType>::_input;
    using Descriptor<OutType>::_surface;

protected:
    SHOT(int nr_shape_bins = 10) :
            nr_shape_bins_ (nr_shape_bins),
            shot_ (), lrf_radius_ (0),
            sqradius_ (0), radius3_4_ (0), radius1_4_ (0), radius1_2_ (0),
            nr_grid_sector_ (32),
            maxAngularSectors_ (32),
            descLength_ (0){};

    void computeDescriptor(PointCloudOutPtr &output);

    int nr_shape_bins_;
    /** \brief Placeholder for a point's SHOT. */
    Eigen::VectorXf shot_;

    float lrf_radius_;
    /** \brief The radius used for the LRF computation */

    double sqradius_;
    /** \brief 3/4 of the search radius. */
    double radius3_4_;

    /** \brief 1/4 of the search radius. */
    double radius1_4_;

    /** \brief 1/2 of the search radius. */
    double radius1_2_;

    /** \brief Number of azimuthal sectors. */
    const int nr_grid_sector_;

    /** \brief ... */
    const int maxAngularSectors_;

    /** \brief One SHOT length. */
    int descLength_;

};
