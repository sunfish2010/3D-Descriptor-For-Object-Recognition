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
    int idx;

    /** \brief indicate whether is a left node **/
    bool isleft;

    int axis;
    std::vector<int>::iterator search_begin;
    std::vector<int>::iterator search_end;
    Node():left(-1), right(-1), parent(-1), id(-1), axis(-1), isleft(false), search_begin(nullptr), search_end(nullptr){}
    ~Node()= default;
private:


};



class KDTree{
public:
    KDTree();
    ~KDTree()= default;
    void make_tree (const std::vector<pcl::SHOT352, Eigen::aligned_allocator<pcl::SHOT352>>& input);
    std::vector<Node, Eigen::aligned_allocator<Node>> getTree()const { return tree; }

protected:
    /** \brief max dim of data **/
    int _dim;
    /** \brief current axis for splitting **/
    int _axis;

private:
    /** \brief tree formed **/
    std::vector<Node, Eigen::aligned_allocator<Node>> tree;
    /** \brief for creating id **/
    int _num_elements;

};



/** \brief correspondence searching with kdtree **/
class Search{
public:

    explicit Search(int k):_k(k){};
    Search():_k(1), _N_input(0), _N_search(0){};
    ~Search() {
        _input.reset();
        _search.reset();
    }

    inline void setK(int k){
        _k = k;
    }

    /** \brief model **/
    void setInputCloud(const pcl::PointCloud<pcl::SHOT352>::ConstPtr &input);

    /** \brief scene **/
    inline void setSearchCloud(const pcl::PointCloud<pcl::SHOT352>::ConstPtr &input){
        _search = input;
        _N_search = static_cast<int>(input->points.size());
    }

    void search(const pcl::CorrespondencesPtr &model_scene_corrs);
    void bruteForce(const pcl::CorrespondencesPtr &model_scene_corrs);

private:
    int _k;
    pcl::PointCloud<pcl::SHOT352>::ConstPtr _input;
    pcl::PointCloud<pcl::SHOT352>::ConstPtr _search;

    int _N_input;
    int _N_search;
    KDTree _kdtree;


};

