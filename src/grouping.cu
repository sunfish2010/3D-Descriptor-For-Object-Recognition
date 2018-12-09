//
// Created by sun on 12/9/18.
//
//
//#include "grouping.h"
//
//__host__ __device__ bool sortCorres(const pcl::Correspondence &lhs, const pcl::Correspondence &rhs) {
//    return lhs.distance < rhs.distance;
//}
//
//__global__ void kernClusterCorresp(int N, const PointType* model, const PointType* scene, const pcl::Correspondence* corrs,
//        const double thres, const int min_size,  int* cluster){
////    extern __shared__ bool group_used[]; // N
//    __shared__ int num_clustered_curr;
//    //  keeps whether has been used
////    if (index < N)
////        group_used[index] = false;
////    __syncthreads();
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    if (index < N){
//        bool in_dist[N];
////        if (!group_used[index * 2]){
//            int scene_self_index = corrs[index].index_match;
//            int model_self_index = corrs[index].index_query;
//            Eigen::Vector3f scene_self = Eigen::Vector3f(scene[scene_self_index].x, scene[scene_self_index].y,
//                                                         scene[scene_self_index].z);
//            Eigen::Vector3f model_self = Eigen::Vector3f(model[model_self_index].x, model[model_self_index].y,
//                                                         model[model_self_index].z);
//            int num_consistent = 0;
//            for (int i = 0; i < N; i++){
//                if (i == index ) continue;
//                int scene_other_index = corrs[i].index_match;
//                int model_other_index = corrs[i].index_query;
//                Eigen::Vector3f scene_other = Eigen::Vector3f(scene[scene_other_index].x, scene[scene_other_index].y,
//                                                              scene[scene_other_index].z);
//                Eigen::Vector3f model_other = Eigen::Vector3f(model[model_other_index].x, model[model_other_index].y,
//                                                              model[model_other_index].z);
//                Eigen::Vector3f dist_scene = scene_other - scene_self;
//                Eigen::Vector3f dist_model = model_other - model_self;
//                double dist = fabs(dist_scene.norm() - dist_model.norm());
////                if (dist > thres)
////                    group_used[i] = false;
//            }
////            __syncthreads();
////            if (grouped_used[index])
////                atomicAdd(&num_clustered_curr,1);
////            __syncthreads();
////            if (num_clustered_curr > min_size){
////                // update array for used & dist
////                if (group_used[index]){
////                    group_used[index * 2] = true;
////                    group_used[index] = false;
////                    num_clustered[0] = prev_num_clustered + 1;
////                }
////
////                // run ransac
////            }
//
////        }
//    }
//}
//
//
//void Grouping::groupCorrespondence() {
//    if (!_input || !_scene || !_corrs){
//    	std::cerr << "grouping has not been correctly set up " << std::endl;
//    	exit(1);
//    }
//
//    pcl::Correspondence* dev_corrs = NULL;
//    cudaMalloc((void**)&dev_corrs, _N_corrs * sizeof(pcl::Correspondence));
//    checkCUDAError("cudamalloc dev_corr ");
//    cudaMemcpy(dev_corrs, &(*_corrs)[0], sizeof(pcl::Correspondence), cudaMemcpyHostToDevice);
//    checkCUDAError("cudamemcpy corr error");
//
//    PointType *dev_input = NULL;
//    cudaMalloc((void**)&dev_input, _N_input * sizeof(PointType));
//    checkCUDAError("cudamalloc dev_input");
//    cudaMemcpy(dev_input, &_input->points[0], _N_input * sizeof(PointType), cudaMemcpyHostToDevice);
//
//    PointType *dev_scene = NULL;
//    cudaMalloc((void**)&dev_scene, _N_scene * sizeof(PointType));
//    checkCUDAError("cudamalloc dev_input");
//    cudaMemcpy(dev_scene, &_scene->points[0], _N_scene * sizeof(PointType), cudaMemcpyHostToDevice);
//
//    int *dev_num_clustered = NULL;
//    cudaMalloc((void**)&dev_num_clustered, sizeof(int));
//    cudaMemset(dev_num_clustered, 0, sizeof(int));
//
//    int blocksize = blockSize;
//    dim3 fullBlockPerGrid_points;
//    if (_N_corrs < blockSize){
//         fullBlockPerGrid_points = dim3(static_cast<u_int32_t >((_N_corrs + blockSize - 1)/blockSize));
//    }else {
//        while (blocksize < _N_corrs) blocksize *= 2;
//        fullBlockPerGrid_points = dim3(static_cast<u_int32_t >((_N_corrs + blocksize - 1) / blocksize));
//    }
//
//    thrust::sort(thrust::device, dev_corrs, dev_corrs + _N_corrs, sortCorres);
//
//    kernClusterCorresp<<< fullBlockPerGrid_points, blocksize>>> (_N_corrs, dev_input, dev_scene, dev_corrs, _thres, dev_num_clustered);
//
//
//}