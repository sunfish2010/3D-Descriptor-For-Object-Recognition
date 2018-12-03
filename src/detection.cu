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

    //compute the common characters for the whole background
    unsigned int N = model->points.size();
    std::cout << "Num of pts is " << N << std::endl;
    float grid_res =  N > 300000? 0.03f:0.01f;
    IndicesPtr grid_indices(new std::vector<int>(N));
    IndicesPtr array_indices(new std::vector<int>(N));
//    std::vector<int> kept_indices;
    Grid grid;
    grid.setRadius(grid_res);
    grid.computeSceneProperty(model, grid_indices, array_indices);
    Eigen::Vector4i pc_dimension = grid.getDimension();
    Eigen::Vector4f inv_radius = grid.getInverseRadius();
    Eigen::Vector4i min_pi = grid.getSceneMin();

    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Min is " << min_pi << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "pc_dimension is " << pc_dimension << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "The inverse radius is " << inv_radius << std::endl;


    UniformDownSample filter;
//
//
//    filter.setKeptIndicesPtr(kept_indices);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
//    filter.downSample(model, model_keypoints, grid_indices, array_indices, inv_radius);
////    filter.randDownSample(model, model_keypoints);
    filter.downSampleAtomic(model, inv_radius, pc_dimension, min_pi);
//    filter.display(model, model_keypoints);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "GPU implementation  downsampling takes: " << duration << std::endl;


//
//    Eigen::Vector4i pc_dimension;
//
//    SHOT_LRF lrf;
//    lrf.setRadius(0.02f);
//    lrf.setInputCloud(model_keypoints);
//    lrf.setSurface(model);
//    lrf.setNormals(model_normals);

    IndicesConstPtr kept_indices = filter.getKeptIndice();
    SHOT descrip_shot;
    descrip_shot.setRadius(0.02);
    descrip_shot.setNormals(model_normals);
    descrip_shot.setInputCloud(model_keypoints);

    descrip_shot.setKeptIndices(kept_indices);
//    descrip_shot.setGridIndices(grid_indices);
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