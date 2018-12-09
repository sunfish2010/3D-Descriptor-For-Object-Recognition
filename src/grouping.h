////
//// Created by sun on 12/9/18.
////
//
//#pragma once
//
//#include "common.h"
//#include "cudaCommon.h"
//
//class Grouping {
//public:
//    explicit Grouping():_thres(1.0),_min_size(3), _scene(NULL), _input(NULL), _corrs(NULL), _N_corrs(0){}
//    ~Grouping(){
//    	_scene.reset();
//    	_input.reset();
//    	_corrs.reset();
//    }
//    inline void setThreshold(double thres){_thres = thres;}
//    inline void setMinGroupSize(int size){_min_size = size;}
//    inline void setInputCloud(const pcl::PointCloud<PointType>::ConstPtr &input){
//        _input = input;
//        _N_input = static_cast<int>(input->points.size());
//    }
//    inline void setSceneCloud(const pcl::PointCloud<PointType>::ConstPtr &scene){
//        _scene = scene;
//        _N_scene = static_cast<int>(scene->points.size());
//    }
//    inline void setCorrespondences(const pcl::CorrespondencesConstPtr &model_scene_corrs){
//    	_corrs = model_scene_corrs;
//    	_N_corrs = static_cast<int>(model_scene_corrs->size());
// 	}
//    void groupCorrespondence();
//
//private:
//    int _min_size;
//    double _thres;
//    int _N_corrs;
//    int _N_input;
//    int _N_scene;
//    pcl::PointCloud<PointType>::ConstPtr _scene;
//    pcl::PointCloud<PointType>::ConstPtr _input;
//    pcl::CorrespondencesConstPtr _corrs;
//    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> _transformations;
//    std::vector<pcl::Correspondences> _grouped_corrs;
//};
//
//
