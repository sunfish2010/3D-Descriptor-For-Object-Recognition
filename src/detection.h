#pragma once

#include "sample.h"
#include "shot.h"
#include "grid.h"
#include "shot_lrf.h"
#include <iostream>


void detectionInit(pcl::PointCloud<PointType>::ConstPtr model,
                   pcl::PointCloud<PointType >::Ptr model_keypoints,
                   pcl::PointCloud<pcl::Normal>::ConstPtr model_normals,
                   pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors);

void detectFree();
