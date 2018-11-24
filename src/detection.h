#pragma once

#include "sample.h"

#include <iostream>


const int blockSize = 128;


void detectionInit(pcl::PointCloud<PointType>::ConstPtr model,
                   pcl::PointCloud<PointType >::Ptr model_keypoints);

void detectFree();

