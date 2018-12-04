#include "main.hpp"

#include <pcl/filters/uniform_sampling.h>
#include <pcl/features/shot_omp.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>

#define GPU 1
#define VERBOSE 1

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

    if (init()){
        mainLoop();
#if GPU
        detectFree();
#endif
        return 0;
    }else{
        return -1;
    }

}

void runCUDA(){

}

bool init(){
#if GPU
    cudaDeviceProp deviceProp;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 0){
        cout << "Error: GPU device not found " << endl;
        return false;
    }
#endif

    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    // model

    viewer->setBackgroundColor(0, 0, 0);

#if GPU
    //pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(model);
    //
    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(off_scene_model);
    viewer->addPointCloud(off_scene_model, rgb, "model");
    //rgb = pcl::visualization::PointCloudColorHandlerRGBField<PointType>(scene);
    //viewer->addPointCloud(scene, rgb, "scene");

    // compute normals
    pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_est;
    normal_est.setKSearch(10);
    normal_est.setInputCloud(model);
    normal_est.compute(*model_normals);
    normal_est.setInputCloud(scene);
    normal_est.compute(*scene_normals);

    pcl::UniformSampling<PointType> uniform_sampling;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    uniform_sampling.setInputCloud (scene);
    uniform_sampling.setRadiusSearch (0.03f);
    uniform_sampling.filter (*scene_keypoints);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "PCL implementation scene downsampling takes: " << duration << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    uniform_sampling.setInputCloud (model);
    uniform_sampling.setRadiusSearch (0.01f);
    uniform_sampling.filter (*model_keypoints);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "PCL implementation model downsampling takes: " << duration << std::endl;

    rgb = pcl::visualization::PointCloudColorHandlerRGBField<PointType>(model_keypoints);
    //viewer->addPointCloud(model_keypoints, rgb, "model_keypoints");
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Model total points CPU: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    Eigen::Vector4f min_p, max_p;

#if VERBOSE
    // Get the minimum and maximum dimensions
    pcl::getMinMax3D<PointType>(*model, min_p, max_p);
    std::cout << "The min for each dimension using pcl is " << min_p << std::endl;
#endif

    (*scene_keypoints).points.clear();
    for (int i = 0 ; i < 2; ++i){
//        detectionInit(model, model_keypoints, model_normals, model_descriptors);
//        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        detectionInit(scene, scene_keypoints, model_normals, model_descriptors);
        std::cout << "---------------------------------------------------------" << std::endl;
//        std::cout << "---------------------------------------------------------" << std::endl;
    }


    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Model total points GPU: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    rgb = pcl::visualization::PointCloudColorHandlerRGBField<PointType>(scene_keypoints);
    viewer->addPointCloud(scene_keypoints, rgb, "scene_keypoints");
    rgb = pcl::visualization::PointCloudColorHandlerRGBField<PointType>(model_keypoints);
    viewer->addPointCloud(model_keypoints, rgb, "model_keypoints");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model");

//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene");
#endif

//    auto pts = (*scene).points;
//    KDTree tree(pts);


    //viewer->addPointCloudNormals<PointType, pcl::Normal>(model, model_normals, 10, 0.05f, "model_normals");

    // scene
//
//    rgb = pcl::visualization::PointCloudColorHandlerRGBField<PointType> (scene);
//    viewer->addPointCloud(scene, rgb, "scene");



    viewer->addCoordinateSystem(1.0);
    viewer->registerKeyboardCallback(keyCallback, (void*)viewer.get());
    viewer->registerMouseCallback(mouseCallback, (void*) viewer.get());

    // TODO:: Initialization for GPU


    return true;
}

void mainLoop(){

#if !GPU
    detection_cpu();
#endif

    // TODO:: Change visualization spin time laps
    while(!viewer->wasStopped()){
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));

//        pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_est;
//        normal_est.setKSearch(10);
//        normal_est.setInputCloud(scene);
//        normal_est.compute(*scene_normals);

        // TODO :: run CUDA
#if GPU
        runCUDA();
#endif
        //display();
    }

}

void keyCallback(const pcl::visualization::KeyboardEvent & event, void *viewer_void){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
    if (event.getKeySym () == "k" && event.keyDown ())
    {
        std::cout << "k was pressed => toggle key points" << std::endl;

        showKeyPoints = !showKeyPoints;
    }
}
void mouseCallback(const pcl::visualization::MouseEvent &event, void *viewer_void){

}

void display(){
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(model);
    viewer->updatePointCloud(model,rgb, "model");

    rgb = pcl::visualization::PointCloudColorHandlerRGBField<PointType> (scene);
    viewer->updatePointCloud(scene, rgb, "scene");
}

void detection_cpu(){
//    pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
//    pcl::PointCloud<PointType>::Ptr scene_keypoints(new pcl::PointCloud<PointType> ());

    // compute normals
    pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_est;
    normal_est.setKSearch(10);
    normal_est.setInputCloud(model);
    normal_est.compute(*model_normals);
    normal_est.setInputCloud (scene);
    normal_est.compute (*scene_normals);

    // downsample pts
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (model);
    uniform_sampling.setRadiusSearch (0.01f);
    uniform_sampling.filter (*model_keypoints);
    std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

    uniform_sampling.setInputCloud (scene);
    uniform_sampling.setRadiusSearch (0.02f);
    uniform_sampling.filter (*scene_keypoints);
    std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;



    //descriptors
    pcl::SHOTEstimationOMP<PointType, pcl::Normal, pcl::SHOT352> descr_est;
    descr_est.setRadiusSearch (0.02f);

    descr_est.setInputCloud (model_keypoints);
    descr_est.setInputNormals (model_normals);
    descr_est.setSearchSurface (model);
    descr_est.compute (*model_descriptors);

    descr_est.setInputCloud (scene_keypoints);
    descr_est.setInputNormals (scene_normals);
    descr_est.setSearchSurface (scene);
    descr_est.compute (*scene_descriptors);

    //
    //  Find Model-Scene Correspondences with KdTree
    //
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    pcl::KdTreeFLANN<pcl::SHOT352> match_search;
    match_search.setInputCloud (model_descriptors);

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < scene_descriptors->size (); ++i)
    {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
        {
            continue;
        }
        int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
            pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back (corr);
        }
    }
    std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

    //
    //  Actual Clustering
    //
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize (0.01f);
    gc_clusterer.setGCThreshold (5.0f);

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize (rototranslations, clustered_corrs);

    std::cout << "Model instances found: " << rototranslations.size () << std::endl;
    // output
    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

        // Print the rotation matrix and translation vector
        Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
        Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

        printf ("\n");
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
        printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
        printf ("\n");
        printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
    }

    //display
    viewer->addPointCloud (scene, "scene_cloud");

    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
    viewer->addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");

    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
        pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;

        pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
        viewer->addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());


        for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
        {
            std::stringstream ss_line;
            ss_line << "correspondence_line" << i << "_" << j;
            PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
            PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

            //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
            viewer->addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
        }

    }

}