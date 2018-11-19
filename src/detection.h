#pragma once

#include <cmath>
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

typedef pcl::PointXYZRGB PointType;
const int blockSize = 128;


void detectionInit(const pcl::PointCloud<PointType>::ConstPtr &scene);

void detectFree();

