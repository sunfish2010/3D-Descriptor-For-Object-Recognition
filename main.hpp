//
// Created by Yu Sun on 11/14/18.
//

#pragma once

#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/recognition/cg/geometric_consistency.h>

#include "src/sample.h"
#include "src/shot.h"
#include "src/grid.h"
#include "src/search.h"
#include "src/shot_lrf.h"
#include "util/utilityCore.hpp"

using namespace std;


bool showKeyPoints = false;
bool toggled_keypoints = false;
bool showNormals = false;
bool toggled_normals = false;
bool showCorresp = false;
bool toggled_corresp = false;
int iter = 0;

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;


bool init(const pcl::PointCloud<PointType>::ConstPtr &model,
          const pcl::PointCloud<PointType >::Ptr &model_keypoints,
          const pcl::PointCloud<pcl::Normal>::Ptr &model_normals,
          const pcl::PointCloud<pcl::SHOT352>::Ptr &model_descriptors);


void keyCallback(const pcl::visualization::KeyboardEvent & event, void *viewer_void);
void mouseCallback(const pcl::visualization::MouseEvent &event, void *viewer_void);

void display(const pcl::PointCloud<PointType>::ConstPtr &model,
             const pcl::PointCloud<PointType >::ConstPtr &model_keypoints,
             const pcl::PointCloud<PointType>::ConstPtr &scene,
             const pcl::PointCloud<PointType >::ConstPtr &scene_keypoints,
             const pcl::PointCloud<pcl::Normal>::ConstPtr &scene_normals,
             const pcl::CorrespondencesConstPtr &model_scene_corrs,
             const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &rototranslations,
             const std::vector<pcl::Correspondences> &clustered_corrs);

void detectionInit(const pcl::PointCloud<PointType>::ConstPtr &model,
                   const pcl::PointCloud<PointType >::Ptr &model_keypoints,
                   const pcl::PointCloud<pcl::Normal>::Ptr &model_normals,
                   const pcl::PointCloud<pcl::SHOT352>::Ptr &model_descriptors);

void detect(const pcl::PointCloud<pcl::SHOT352>::ConstPtr &model_descriptors,
            const pcl::PointCloud<PointType>::ConstPtr &scene,
            const pcl::PointCloud<PointType >::Ptr &scene_keypoints,
            const pcl::PointCloud<pcl::Normal>::Ptr &scene_normals,
            const pcl::PointCloud<pcl::SHOT352>::Ptr &scene_descriptors,
            const pcl::CorrespondencesPtr &model_scene_corrs);


void computeDescriptor(const pcl::PointCloud<PointType>::ConstPtr &model,
                       const pcl::PointCloud<PointType >::Ptr &model_keypoints,
                       const pcl::PointCloud<pcl::Normal>::Ptr &model_normals,
                       const pcl::PointCloud<pcl::SHOT352>::Ptr &model_descriptors);
