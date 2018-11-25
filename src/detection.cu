#include "detection.h"

//static int n_scene;
//static int n_model;

//static PointType *dev_ss_pc_scene = NULL;
//static PointType *dev_ss_pc_model = NULL;
//static PointType *dev_kp_scene = NULL;
//static PointType *dev_kp_model = NULL;



void detectionInit(pcl::PointCloud<PointType >::ConstPtr model,
        pcl::PointCloud<PointType >::Ptr model_keypoints,
        pcl::PointCloud<pcl::Normal>::ConstPtr model_normals,
        pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors){
//    n_model =
    UniformDownSample filter = UniformDownSample(0.01);

    filter.downSample(model, model_keypoints);

    SHOT descrip_shot;
    descrip_shot.setRadius(0.02);
    descrip_shot.setNormals(model_normals);
    descrip_shot.setInputCloud(model_keypoints);
    descrip_shot.setSurface(model);
    //descrip_shot.compute(*model_descriptors);


}

void detectFree(){
//    cudaFree(dev_ss_pc_model);
//    cudaFree(dev_ss_pc_scene);
//    cudaFree(dev_kp_model);
//    cudaFree(dev_kp_scene);

//    dev_ss_pc_scene = NULL;
//    dev_ss_pc_model = NULL;
//    dev_kp_scene = NULL;
//    dev_kp_model = NULL;
//
//    checkCUDAError("cuda Free error");

}