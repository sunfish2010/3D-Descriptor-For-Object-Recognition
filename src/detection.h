#pragma once

#include "sample.h"
#include "descriptor.h"
#include <iostream>


void detectionInit(pcl::PointCloud<PointType>::ConstPtr model,
                   pcl::PointCloud<PointType >::Ptr model_keypoints);

void detectFree();

