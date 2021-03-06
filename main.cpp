//
// Created by sun on 12/9/18.
//

#include "main.hpp"

using namespace std;

#define VERBOSE 0
#define DISPLAY 1
#define SHIFT_MODEL 1

const float PI = 3.141592653589;

int main(int argc, char* argv[]){
    if (argc < 3){
        cout << "Usage: [model file] [scene file]. Press Enter to exit" << endl;
        getchar();
        return 0;
    }

    string mfn(argv[1]);
    string sfn(argv[2]);
    string m_ext = utilityCore::getFilePathExtension(mfn);
    string s_ext = utilityCore::getFilePathExtension(sfn);

    pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType>);

    if (m_ext == "pcd" && s_ext == "pcd"){
        if (pcl::io::loadPCDFile(mfn, *model) < 0){
            cout << "Error loading model cloud " << endl;
            return -1;
        }
        if (pcl::io::loadPCDFile(sfn, *scene) < 0){
            cout << "Error loading scene cloud" << endl;
            return -1;
        }

    }

#if SHIFT_MODEL
    pcl::PointCloud<PointType>::Ptr model_shifted (new pcl::PointCloud<PointType>);
    pcl::transformPointCloud (*model, *model_shifted, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::copyPointCloud(*model_shifted,*model);
#endif

#if DISPLAY
    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
//    viewer->registerKeyboardCallback(keyCallback, (void*)viewer.get());
//    viewer->registerMouseCallback(mouseCallback, (void*)viewer.get());
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(model);
    viewer->addPointCloud(model, rgb, "model");
    rgb = pcl::visualization::PointCloudColorHandlerRGBField<PointType>(scene);
    viewer->addPointCloud(scene, rgb, "scene");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene");
#endif

    pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors(new pcl::PointCloud<pcl::SHOT352>);
    pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors(new pcl::PointCloud<pcl::SHOT352>);
    pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType>);
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    detectionInit(model, model_keypoints, model_normals, model_descriptors);

#if VERBOSE

    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Model total points GPU: " << model->size() << "; Selected Keypoints: " << model_keypoints->size() << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

#endif


#if DISPLAY
    while (!viewer->wasStopped ()) {
        viewer->removeAllShapes();
        Eigen::Matrix4f randomRotranslation = Eigen::Matrix4f::Zero();
        Eigen::Matrix3f rand_rotation;
        rand_rotation << cos(PI/20),0,sin(PI/20), 0, 1, 0, -sin(PI/20), 0, cos(PI/20);
        randomRotranslation.block<3,3>(0, 0) = rand_rotation;
        pcl::transformPointCloud (*scene, *scene, randomRotranslation);
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(scene);
        viewer->updatePointCloud(scene, rgb, "scene");
//        scene_keypoints->clear();
//        scene_normals->clear();
//        scene_descriptors->clear();
        model_scene_corrs->clear();
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        detect(model_descriptors, scene, scene_keypoints, scene_normals, scene_descriptors, model_scene_corrs);
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        std::cout << "detect takes: " << duration << std::endl;

#if VERBOSE
        std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
#endif
        // using pcl's geometric consistency grouping
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
        std::vector<pcl::Correspondences> clustered_corrs;

        pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
        gc_clusterer.setGCSize (0.01);
        gc_clusterer.setGCThreshold (5);

        gc_clusterer.setInputCloud (model_keypoints);
        gc_clusterer.setSceneCloud (scene_keypoints);
        gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

        gc_clusterer.recognize (rototranslations, clustered_corrs);

        std::cout << rototranslations.size() << std::endl;
        display(model, model_keypoints, scene, scene_keypoints,
                scene_normals, model_scene_corrs, rototranslations, clustered_corrs);

        viewer->spinOnce ();
    }
#endif


}


void detectionInit(const pcl::PointCloud<PointType>::ConstPtr &model,
                   const pcl::PointCloud<PointType >::Ptr &model_keypoints,
                   const pcl::PointCloud<pcl::Normal>::Ptr &model_normals,
                   const pcl::PointCloud<pcl::SHOT352>::Ptr &model_descriptors){

    computeDescriptor(model, model_keypoints, model_normals, model_descriptors);


}


void detect(const pcl::PointCloud<pcl::SHOT352>::ConstPtr &model_descriptors,
            const pcl::PointCloud<PointType>::ConstPtr &scene,
            const pcl::PointCloud<PointType >::Ptr &scene_keypoints,
            const pcl::PointCloud<pcl::Normal>::Ptr &scene_normals,
            const pcl::PointCloud<pcl::SHOT352>::Ptr &scene_descriptors,
            const pcl::CorrespondencesPtr &model_scene_corrs){

    computeDescriptor(scene, scene_keypoints, scene_normals, scene_descriptors);

    Search corr_search;
    corr_search.setInputCloud(model_descriptors);
    corr_search.setSearchCloud(scene_descriptors);
//    corr_search.search(model_scene_corrs);
    corr_search.bruteForce(model_scene_corrs);

//    pcl::KdTreeFLANN<pcl::SHOT352> match_search;
//    match_search.setInputCloud (model_descriptors);
//    auto rep = match_search.point_representation_;
//    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
//    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
//    for (size_t i = 0; i < scene_descriptors->size (); ++i)
//    {
//        std::vector<int> neigh_indices (1);
//        std::vector<float> neigh_sqr_dists (1);
//        if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
//        {
//            continue;
//        }
//        int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
//        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
//        {
//            pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
//            model_scene_corrs->push_back (corr);
//        }
//    }
//    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
//    std::cout << "CPU implementation  search takes: " << duration << std::endl;
//
    std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;


}


void computeDescriptor(const pcl::PointCloud<PointType>::ConstPtr &model,
                       const pcl::PointCloud<PointType >::Ptr &model_keypoints,
                       const pcl::PointCloud<pcl::Normal>::Ptr &model_normals,
                       const pcl::PointCloud<pcl::SHOT352>::Ptr &model_descriptors){

    // compute normals
    pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_est;
    normal_est.setKSearch(10);
    normal_est.setInputCloud(model);
    normal_est.compute(*model_normals);

    //compute the common characters for the whole background
    unsigned int N = model->points.size();

    float grid_res =  N > 300000? 0.03f:0.01f;
    IndicesPtr grid_indices(new std::vector<int>(N));
    IndicesPtr array_indices(new std::vector<int>(N));
//    std::vector<int> kept_indices;
    Eigen::Vector4i min_pi;
    Eigen::Vector4i max_pi;
    Eigen::Vector4f inv_radius;
    Eigen::Vector4i pc_dimension;

    Grid grid;
    grid.setRadius(grid_res);
    grid.computeSceneProperty(model, grid_indices, array_indices, inv_radius, pc_dimension, min_pi, max_pi);
    IndicesConstPtr cell_start_indices = grid.getCellStart();
    IndicesConstPtr cell_end_indices = grid.getCellEnd();

#if VERBOSE
    std::cout << "Num of pts is " << N << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Min is " << min_pi << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "pc_dimension is " << pc_dimension << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "The inverse radius is " << inv_radius << std::endl;
#endif

    UniformDownSample filter;

//    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    filter.downSampleAtomic(model, inv_radius, pc_dimension, min_pi);
    filter.display(model, model_keypoints);
//    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
//    std::cout << "GPU implementation  downsampling takes: " << duration << std::endl;
    IndicesConstPtr kept_indices = filter.getKeptIndice();

//    SHOT352 descrip_shot;
//    descrip_shot.setRadius(0.02);
//    descrip_shot.setNormals(model_normals);
//    descrip_shot.setInputCloud(model_keypoints);
//    descrip_shot.setKeptIndices(kept_indices);
//    descrip_shot.setGridIndices(grid_indices);
//    descrip_shot.setSurface(model);
//    descrip_shot.compute(*model_descriptors, inv_radius, pc_dimension, min_pi);

    //descriptors
//    t1 = std::chrono::high_resolution_clock::now();
    pcl::SHOTEstimationOMP<PointType, pcl::Normal, pcl::SHOT352> descr_est;
    descr_est.setRadiusSearch (0.02f);
    descr_est.setInputCloud (model_keypoints);
    descr_est.setInputNormals (model_normals);
    descr_est.setSearchSurface (model);
    descr_est.compute (*model_descriptors);
//    t2 = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
//    std::cout << "local reference calculation takes: " << duration << std::endl;




}


void display(const pcl::PointCloud<PointType>::ConstPtr &model,
             const pcl::PointCloud<PointType >::ConstPtr &model_keypoints,
             const pcl::PointCloud<PointType>::ConstPtr &scene,
             const pcl::PointCloud<PointType >::ConstPtr &scene_keypoints,
             const pcl::PointCloud<pcl::Normal>::ConstPtr &scene_normals,
             const pcl::CorrespondencesConstPtr &model_scene_corrs,
             const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &rototranslations,
             const std::vector<pcl::Correspondences> &clustered_corrs){

    if (showKeyPoints){
        pcl::visualization::PointCloudColorHandlerCustom<PointType> color_handler(scene_keypoints, 0, 255, 0);
        viewer->addPointCloud (scene_keypoints, color_handler, "scene_keypoints");
    }else{
        if (toggled_keypoints){
            viewer->removePointCloud("scene_keypoints");
            toggled_keypoints = false;
        }
    }

    if (showNormals){
        viewer->addPointCloudNormals<PointType, pcl::Normal>(scene, scene_normals, 5, 0.05f, "model_normals");
    }else{
        if (toggled_normals){
            viewer->removePointCloud("model_normals");
            toggled_normals = false;
        }
    }


    if (showCorresp){

        for (size_t i = 0; i < rototranslations.size (); ++i) {
//            std::cout << clustered_corrs[i].size()<< ", iter" << iter << std::endl;
            if (clustered_corrs[i].size() < 10 )
                continue;
            pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
            pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

            std::stringstream ss_cloud;
            ss_cloud << "instance" << i;

            pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
            if (iter == 0)
                viewer->addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
            else
                viewer->updatePointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str ());

            for (size_t j = 0; j < clustered_corrs[i].size (); ++j) {
                std::stringstream ss_line;
                ss_line << "correspondence_line" << i << "_" << j;
                const PointType& model_point = model_keypoints->at (clustered_corrs[i][j].index_query);
                const PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);
                viewer->addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
            }

        }
        iter++;
    }else{
        if (toggled_corresp){
            viewer->removeAllShapes();
            toggled_corresp = false;
        }
    }

}


