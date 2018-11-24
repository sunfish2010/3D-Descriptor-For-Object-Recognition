#include "detection.h"

//static int n_scene;
//static int n_model;

static PointType *dev_ss_pc_scene = NULL;
static PointType *dev_ss_pc_model = NULL;
static PointType *dev_kp_scene = NULL;
static PointType *dev_kp_model = NULL;



void detectionInit(pcl::PointCloud<PointType >::ConstPtr model,
        pcl::PointCloud<PointType >::Ptr model_keypoints){
//    n_model =
    UniformDownSample filter = UniformDownSample(0.01);

    filter.downSample(model, model_keypoints);
    std::cout<< "Valid " << std::endl;

}

void detectFree(){
//    cudaFree(dev_ss_pc_model);
//    cudaFree(dev_ss_pc_scene);
//    cudaFree(dev_kp_model);
//    cudaFree(dev_kp_scene);

    dev_ss_pc_scene = NULL;
    dev_ss_pc_model = NULL;
    dev_kp_scene = NULL;
    dev_kp_model = NULL;

    checkCUDAError("cuda Free error");

}