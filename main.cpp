#include "main.hpp"
#include <pcl/filters/uniform_sampling.h>
#include <pcl/common/common.h>

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
        detectFree();
        return 0;
    }else{
        return -1;
    }

}

void runCUDA(){

}

bool init(){
    cudaDeviceProp deviceProp;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 0){
        cout << "Error: GPU device not found " << endl;
        return false;
    }
    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    // model

    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(model);
    viewer->addPointCloud(model, rgb, "model");

    // compute normals
    pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_est;
    normal_est.setKSearch(10);
    normal_est.setInputCloud(model);
    normal_est.compute(*model_normals);

    pcl::UniformSampling<PointType> uniform_sampling;
    pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
    uniform_sampling.setInputCloud (model);
    uniform_sampling.setRadiusSearch (0.01f);
    uniform_sampling.filter (*model_keypoints);
    std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

    Eigen::Vector4f min_p, max_p;
    // Get the minimum and maximum dimensions
    pcl::getMinMax3D<PointType>(*model, min_p, max_p);
    std::cout << "The min for each dimension using pcl is " << min_p << std::endl;
    detectionInit(model);

//    auto pts = (*scene).points;
//    KDTree tree(pts);


    //viewer->addPointCloudNormals<PointType, pcl::Normal>(model, model_normals, 10, 0.05f, "model_normals");

    // scene
//
//    rgb = pcl::visualization::PointCloudColorHandlerRGBField<PointType> (scene);
//    viewer->addPointCloud(scene, rgb, "scene");


    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model");
    //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene");
    viewer->addCoordinateSystem(1.0);
    viewer->registerKeyboardCallback(keyCallback, (void*)viewer.get());
    viewer->registerMouseCallback(mouseCallback, (void*) viewer.get());

    // TODO:: Initialization for GPU


    return true;
}

void mainLoop(){


    // TODO:: Change visualization spin time laps
    while(!viewer->wasStopped()){
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));

//        pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_est;
//        normal_est.setKSearch(10);
//        normal_est.setInputCloud(scene);
//        normal_est.compute(*scene_normals);

        // TODO :: run CUDA
        runCUDA();

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