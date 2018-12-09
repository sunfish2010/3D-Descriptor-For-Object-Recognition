#pragma once

#include <iostream>

#include <boost/version.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/make_shared.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/correspondence.h>
#include <pcl/point_types.h>

#include <chrono>

// for easier typing
typedef pcl::PointXYZRGB PointType;
typedef boost::shared_ptr<std::vector<int>> IndicesPtr;
typedef boost::shared_ptr<const std::vector<int>> IndicesConstPtr;
const int blockSize = 256;
