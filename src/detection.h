#pragma once

#include "sample.h"
#include "shot.h"
#include "grid.h"
#include "search.h"
#include "shot_lrf.h"
#include <iostream>


void detectionInit(const pcl::PointCloud<PointType>::ConstPtr &model,
                   const pcl::PointCloud<PointType >::Ptr &model_keypoints,
                   const pcl::PointCloud<pcl::Normal>::Ptr &model_normals,
                   const pcl::PointCloud<pcl::SHOT352>::Ptr &model_descriptors);

void detectFree();


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
