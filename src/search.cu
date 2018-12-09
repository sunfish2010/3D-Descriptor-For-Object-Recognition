#include "search.h"


void KDTree::make_tree(const std::vector<pcl::SHOT352, Eigen::aligned_allocator<pcl::SHOT352>>& input) {

    std::vector<int> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0);

    Node root;
    root.axis = 0;
    root.search_begin = indices.begin();
    root.search_end = indices.end();

    std::vector<Node, Eigen::aligned_allocator<Node>>Nodes;
    Nodes.emplace_back(root);

    while(!Nodes.empty()){
        Node curr = Nodes.back();
        Nodes.pop_back();
        if(curr.search_end > curr.search_begin + 1){
            curr.id = _num_elements++;
            if (curr.parent != -1){
                if (curr.isleft)
                    tree[curr.parent].left = curr.id;
                else
                    tree[curr.parent].right = curr.id;
            }

            Node left, right;
            _axis = curr.axis

            std::sort(curr.search_begin, curr.search_end,
                      [&input](size_t i1, size_t i2){
                          return input[i1].descriptor[kdtree->_axis] < input[i2].descriptor[kdtree->_axis];});

            auto mid = curr.search_begin + (curr.search_end - curr.search_begin)/2;
            curr.idx = *mid;

            if (mid - curr.search_begin > 0){
                left.id = mid - curr.search_begin+ curr.id + 1;
                left.parent = curr.id;
                left.axis = (curr.axis + 1) % _dim;
                left.isleft = true;
            }
            if (curr.search_end - mid > 1){
                right.axis = (curr.axis + 1) % _dim;
                right.id = curr.id + 1;
                right.parent = curr.id;
            }

            curr.left = left.id;
            curr.right = right.id;

            left.search_begin = curr.search_begin;
            left.search_end = mid;
            right.search_begin = mid + 1;
            right.search_end = curr.search_end;

            if (left.search_end > left.search_begin )
                Nodes.emplace_back(left);
            if (right.search_end > right.search_begin)
                Nodes.emplace_back(right);
            tree.emplace_back(curr);

        }
        else if (curr.search_begin +1 == curr.search_end){
            curr.id = _num_elements++;
            if (curr.parent != -1){
                if (curr.isleft)
                    tree[curr.parent].left = curr.id;
                else
                    tree[curr.parent].right = curr.id;
            }
            curr.idx = *curr.search_begin;
            tree.emplace_back(curr);
        }
    }


}

__device__ float descriptorDistance(const pcl::SHOT352& pt1, const pcl::SHOT352 &pt2){
    const int desclen_ = 352;
    float dist = 0;
    for (int i = 0; i < desclen_; ++i){
        float delta = pt1.descriptor[i] - pt2.descriptor[i];
        dist += delta * delta;
    }
    return sqrt(dist);
}


__global__ void kernFindCorrespondence(int N, const Node* nodes, const pcl::SHOT352* input, const pcl::SHOT352* queries,
        int* indices, float* dist){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N){
        pcl::SHOT352 query = queries[index];
        int n_idx = nodes[0].idx;
        int n_closest = 0;
        int split_axis = nodes[0].axis;
        float d_closest = descriptorDistance( input[n_idx], query);
        int curr_node = query.descriptor[split_axis] > input[n_idx].descriptor[split_axis] ?
                nodes[0].right:nodes[0].left;
        bool explored = false;
        while(true){
            while(curr_node != -1){
                n_idx = nodes[curr_node].idx;
                split_axis = nodes[curr_node].axis;
                float distance = descriptorDistance(input[n_idx], query);
                if (distance < d_closest){
                    d_closest = distance;
                    n_closest = curr_node;
                    explored = false;
                }
                curr_node = query.descriptor[split_axis] > input[n_idx].descriptor[split_axis]?
                        nodes[curr_node].right: nodes[curr_node].left;
            }
            if (explored ||node[n_closest].parent == -1){
                break;
            } else{
                // explore parents
                curr_node = node[n_closest].parent;
                n_idx = nodes[curr_node].idx;
                split_axis = nodes[curr_node].axis;
                float hyper_dist = query.descriptor[split_axis] - input[n_idx].descriptor[split_axis];
                if (abs(hyper_dist) < d_closest){
                    explored = true;
                    curr_node = hyper_dist > 0? nodes[curr_node].right:nodes[curr_node].left;
                }else{
                    break;
                }
            }
        }
        indices[index] = nodes[n_closest].idx;

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

    float *dev_dist = NULL;
    cudaMalloc((void**)&dev_dist, _N_search * sizeof(float) );
    checkCUDAError("dev_dist malloc");


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