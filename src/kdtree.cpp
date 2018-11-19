#include "kdtree.hpp"

Node::Node():left(-1), right(-1), parent(-1),
             data(pcl::PointXYZRGB()), axis(0){}

Node::Node(const pcl::PointXYZRGB &value, int axis):left(-1), right(-1), parent(-1), data(value), axis(axis) {}

Node::Node(const pcl::PointXYZRGB &value, int left,
           int right):left(left), right(right), parent(-1), data(value), axis(0) {}

KDTree::KDTree(std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> >& pts) {
    //std::sort(pts.begin(), pts.end(), sortX());
    this->tree = std::vector<Node>(pts.size(), Node());
    make_tree(pts.begin(), pts.end(), 0, pts.size(),  0);

}


void KDTree::make_tree(ptsIter begin, ptsIter end, int axis, int length, int index) {
    // just edge case checking, will it happen?
    if (begin == end) return;

    if (axis == 0)
        std::sort(begin, end, sortDimx);
    else if (axis == 1)
        std::sort(begin, end, sortDimy);
    else
        std::sort(begin, end, sortDimz);

    auto mid = begin + (length / 2);

    int llen = length / 2;
    int rlen = length - llen - 1;

    tree[index] = Node((*mid), axis);

    int leftNode, rightNode;

    if (llen > 0 && begin != mid){
        leftNode = index + 1;
        make_tree(begin, mid, (axis + 1) % 3, llen, index + 1);
    }else{
        leftNode = -1;
    }
    if (rlen > 0 && mid+1 != end){
        rightNode = index + (length / 2) + 1;
        make_tree(mid + 1, end, (axis + 1) % 3, rlen,  index + (length / 2) + 1);
    }else{
        rightNode = -1;
    }
    tree[index].left = leftNode;
    tree[index].right = rightNode;
    if (leftNode != -1) tree[tree[index].left].parent = index;
    if (rightNode != -1) tree[tree[index].right].parent = index;

}

bool sortDimx(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB &pt2) {
    return pt1.x < pt2.x;
}


bool sortDimy(const pcl::PointXYZRGB &pt1,const pcl::PointXYZRGB &pt2) {
    return pt1.y < pt2.y;
}


bool sortDimz(const pcl::PointXYZRGB &pt1,const pcl::PointXYZRGB& pt2) {
    return pt1.z < pt2.z;
}