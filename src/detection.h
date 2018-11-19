#pragma once

#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

#include <boost/version.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/sync/file_lock.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include "checkCUDAError.h"

#include <iostream>

typedef pcl::PointXYZRGB PointType;
const int blockSize = 128;
const float radius = 0.01f;


void detectionInit(const pcl::PointCloud<PointType>::ConstPtr model);

void detectFree();

void UniformDownSample(int& N, const pcl::PointCloud<PointType >::ConstPtr input, bool mode);