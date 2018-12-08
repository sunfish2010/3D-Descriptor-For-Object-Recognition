#pragma once
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "cudaCommon.h"

#include <algorithm>
#include <vector>
#include <memory>

class KDTree;
static KDTree* kdtree;

class Node{
public :
    // can't use ptr for cuda
    //using NodePtr = std::shared_ptr<Node>;
// has to be public due to call in __device__ function
    //NodePtr left;
    //NodePtr right;
    //NodePtr parent;

    int left, right, parent;
    int id;
    pcl::SHOT352 data;
    int axis;
    std::vector<pcl::SHOT352, Eigen::aligned_allocator<pcl::SHOT352>>::iterator search_begin;
    std::vector<pcl::SHOT352, Eigen::aligned_allocator<pcl::SHOT352>>::iterator search_end;

    Node():left(-1), right(-1), parent(-1), id(-1), axis(-1), search_begin(nullptr), search_end(nullptr){}
    ~Node()= default;
    //int getAxis();
private:


};



class KDTree{
public:
    KDTree():_dim(352), _axis(0),_num_elements(0){kdtree = this;}
    virtual ~KDTree()= default;
    void make_tree (std::vector<pcl::SHOT352, Eigen::aligned_allocator<pcl::SHOT352>> input);
    std::vector<Node, Eigen::aligned_allocator<Node>> getTree()const { return tree; }
    static bool sortDim(const pcl::SHOT352& a, const pcl::SHOT352& b){
        return a.descriptor[kdtree->_axis] < b.descriptor[kdtree->_axis];}
protected:
    int _dim;
    int _axis;

private:
    std::vector<Node, Eigen::aligned_allocator<Node>> tree;
    int _num_elements;

};




class Search{
public:

    explicit Search(int k):_k(k){};
    Search():_k(1), dev_input(NULL),
             dev_neighbor_indices(NULL), dev_search(NULL), _N_input(0), _N_search(0){};
    ~Search();

    inline void setK(int k){
        _k = k;
    }

    void setInputCloud(const pcl::PointCloud<pcl::SHOT352>::ConstPtr &input);

    inline void setSearchCloud(const pcl::PointCloud<pcl::SHOT352>::Ptr &input){
        _search = input;
        _N_search = static_cast<int>(input->points.size());
    }


    void search(const pcl::CorrespondencesPtr &model_scene_corrs);

    inline IndicesConstPtr getNeighborIndices(){return IndicesConstPtr(&_neighbor_indices);}


private:
    int _k;
    pcl::PointCloud<pcl::SHOT352>::ConstPtr _input;
    pcl::PointCloud<pcl::SHOT352>::ConstPtr _search;

    int *dev_neighbor_indices;
    pcl::SHOT352 *dev_search;
    pcl::SHOT352 *dev_input;
    int _N_input;
    int _N_search;
    KDTree _kdtree;
    std::vector<int> _neighbor_indices;

};

