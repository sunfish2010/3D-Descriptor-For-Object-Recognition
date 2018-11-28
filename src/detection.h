#pragma once

#include "sample.h"
#include "search.h"
#include "shot.h"
#include <iostream>

#include <chrono>
void detectionInit(pcl::PointCloud<PointType>::ConstPtr model,
                   pcl::PointCloud<PointType >::Ptr model_keypoints,
                   pcl::PointCloud<pcl::Normal>::ConstPtr model_normals,
                   pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors);

void detectFree();

