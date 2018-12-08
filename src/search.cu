#include "search.h"


void KDTree::make_tree(std::vector<pcl::SHOT352, Eigen::aligned_allocator<pcl::SHOT352>> input) {

    Node root;
    root.id = _num_elements++;
    root.axis = 0;
    root.search_begin = input.begin();
    root.search_end = input.end();

    std::vector<Node, Eigen::aligned_allocator<Node>>Nodes;

    while(!Nodes.empty()){
        Node curr = Nodes.back();
        Nodes.pop_back();
        if(curr.search_end > curr.search_begin ){
            Node left, right;
            left.id = _num_elements++;
            right.id = _num_elements++;
            curr.left = left.id;
            curr.right = right.id;
            left.axis = curr.axis + 1;
            right.axis = curr.axis + 1;
            left.parent = curr.id;
            right.parent = curr.id;
            _axis = curr.axis % _dim;
            std::sort(curr.search_begin, curr.search_end, sortDim);
            auto mid = curr.search_begin + (curr.search_end - curr.search_begin)/2;
            curr.data = *mid;
            left.search_begin = curr.search_begin;
            left.search_end = mid - 1;
            right.search_begin = mid + 1;
            right.search_end = curr.search_end;
            if (left.search_end >= left.search_begin )
                Nodes.emplace_back(left);
            if (right.search_end >= right.search_begin)
                Nodes.emplace_back(right);

        }
        else if (curr.search_begin == curr.search_end){
            curr.data = *curr.search_end;
        }
        tree.emplace_back(curr);
    }


}

Search::~Search() {

//    cudaFree(dev_grid_indices);
    dev_neighbor_indices = NULL;

}


void Search::setInputCloud(const pcl::PointCloud<pcl::SHOT352>::ConstPtr &input) {
    _input = input;
    _N_input = static_cast<int>(input->points.size());
    _kdtree.make_tree(input->points);

}

void Search::search(const pcl::CorrespondencesPtr &model_scene_corrs) {
    if (!_search || !_input || _N_input > _N_search){
        std::cerr << "Search function not properly setup" << std::endl;
        exit(1);
    }

    const std::vector<Node, Eigen::aligned_allocator<Node>>& tree = _kdtree.getTree();
    assert(_N_input == tree.size());
    Node* dev_tree = NULL;
    cudaMalloc((void**)&dev_tree, _N_input * sizeof(Node));
    checkCUDAError("cudamalloc dev tree error");
    cudaMemcpy(dev_tree, &tree[0], _N_input * sizeof(Node), cudaMemcpyDeviceToHost);
    checkCUDAError("cudammcpy dev_tree error");

    cudaMalloc((void**)&dev_neighbor_indices, _N_search * sizeof(int));
    checkCUDAError("malloc dev_neighbor indices error");
    cudaMemset(dev_neighbor_indices, -1, _N_search * sizeof(int));
    checkCUDAError("memset ni error");

    cudaMalloc((void**)&dev_input, _N_input * sizeof(pcl::SHOT352));
    checkCUDAError("malloc dev_neighbor distances error");
    cudaMemcpy(dev_input, &(_input->points[0]), _N_input * sizeof(pcl::SHOT352), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_search, _N_search * sizeof(pcl::SHOT352));
    checkCUDAError("malloc dps error");
    cudaMemcpy(dev_search, &(_search->points[0]), _N_search * sizeof(pcl::SHOT352), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy ps error");

    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((_N_search + blockSize - 1)/blockSize));

//    kernSearchCorrespondence

    checkCUDAError("KernSearchCorres error");


    _neighbor_indices.resize(_N_search);
    cudaMemcpy(&_neighbor_indices[0], dev_neighbor_indices, sizeof(int) * _N_search, cudaMemcpyDeviceToHost);
    checkCUDAError("cudamemcpy  num neigbors issue");


    for(auto&n: _neighbor_indices)
        std::cout << n << std::endl;

    cudaFree(dev_search);
    cudaFree(dev_input);
    cudaFree(dev_neighbor_indices);


}