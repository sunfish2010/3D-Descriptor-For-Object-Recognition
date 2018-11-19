#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

class Node{
public :
    // can't use ptr for cuda
    //using NodePtr = std::shared_ptr<Node>;
// has to be public due to call in __device__ function
    //NodePtr left;
    //NodePtr right;
    //NodePtr parent;

    int left, right, parent;

    pcl::PointXYZRGB data;
    int axis;

    Node();
    Node(const pcl::PointXYZRGB &value, int axis);
    Node(const pcl::PointXYZRGB &value, int left, int right);
    ~Node()= default;
    //int getAxis();



};

using ptsIter = std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> >::iterator;

class KDTree{
public:
    KDTree()= default;
    ~KDTree()= default;
    KDTree(std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> >&);
    std::vector<Node> getTree()const { return tree; }

private:
    using ptsIter = std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> >::iterator;
    void make_tree(ptsIter begin, ptsIter end, int axis, int length, int index);
    std::vector<Node> tree;
};


bool sortDimx(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB& pt2);
bool sortDimy(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB& pt2);
bool sortDimz(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB& pt2);