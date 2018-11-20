#pragma once

#include "sample.h"

#include <iostream>


const int blockSize = 128;


void detectionInit(const pcl::PointCloud<PointType>::ConstPtr model);

void detectFree();

