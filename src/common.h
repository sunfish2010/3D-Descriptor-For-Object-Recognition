#pragma once

#include <iostream>

#include <boost/version.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/sync/file_lock.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>

#include <chrono>
typedef pcl::PointXYZRGB PointType;
typedef boost::shared_ptr<std::vector<int>> IndicesPtr;

const int blockSize = 256;