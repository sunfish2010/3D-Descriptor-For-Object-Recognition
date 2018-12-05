//
// Created by sun on 11/26/18.
//

#include "shot_lrf.h"
#include <Eigen/Eigenvalues>
__device__ Eigen::Vector4f getVector4f(PointType pt){
    return Eigen::Vector4f(pt.x, pt.y, pt.z, 0.f);
}

__device__ Eigen::Vector4d Vector4f2Vector4d(Eigen::Vector4f pt){
    return Eigen::Vector4d(static_cast<double>(pt[0]), static_cast<double>(pt[1]), static_cast<double>(pt[2]),
            static_cast<double>(pt[3]));
}


// somehow if i don't include the implementation, it complains not definition
// while if I implement it complains about existing implementation so changed name
#if __CUDA_ARCH__ < 600
__device__ double AtomicAdd(double *address, double val){
    unsigned long long int *address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    }while(assumed != old);
    return __longlong_as_double(old);
}

#endif


__global__ void kernRadiusSearch(int N, int n, const PointType *surface, const float radius, const int *feature_indices,
        const Eigen::Vector4f inv_radius, const Eigen::Vector4i min_pi, int* num_neighbors, bool* connected){
//    extern __shared__ int neighbor[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
//    if (index < n) {
//        neighbor[idx] = 0;
//    }
//    __syncthreads();
    if (index < N) {
        PointType pt = surface[index];
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)) {
            uint8_t i = static_cast<uint8_t >(floor(pt.x * inv_radius[0]) - min_pi[0]);
            uint8_t j = static_cast<uint8_t >(floor(pt.y * inv_radius[1]) - min_pi[1]);
            uint8_t k = static_cast<uint8_t >(floor(pt.z * inv_radius[2]) - min_pi[2]);
//            float curr_dist = (pt.x - i) * (pt.x - i) + (pt.y - j) * (pt.y - j) + (pt.z - k) * (pt.z - k);
            for (int idx = 0; idx < n; ++idx) {
//                printf("N is %d, n is %d, number is %d, feature idx is %d\n",N, n, idx, feature_indices[idx]);
                PointType central_point = surface[feature_indices[idx]];
                Eigen::Vector3i min_idx = Eigen::Vector3i(
                        static_cast<int>(floor((central_point.x - radius) * inv_radius[0]) - min_pi[0]),
                        static_cast<int>(floor((central_point.y - radius) * inv_radius[1]) - min_pi[1]),
                        static_cast<int>(floor((central_point.z - radius) * inv_radius[2]) - min_pi[2]));
                Eigen::Vector3i max_idx = Eigen::Vector3i(
                        static_cast<int>(floor((central_point.x + radius) * inv_radius[0]) - min_pi[0]),
                        static_cast<int>(floor((central_point.y + radius) * inv_radius[1]) - min_pi[1]),
                        static_cast<int>(floor((central_point.z + radius) * inv_radius[2]) - min_pi[2]));


                if (i >= min_idx[0] && i <= max_idx[0] && j >= min_idx[1] && j <= max_idx[1]
                && k >= min_idx[2] && k <= max_idx[2]
                && !(pt.x == central_point.x && pt.y == central_point.y && pt.z == central_point.z)) {

                    atomicAdd(&num_neighbors[idx],1);
                    connected[idx * N + index] = true;

                }
            }
        }
    }
}



__global__ void kernComputeLF(int N, int n, const PointType * surface,const int* num_neighbors,const int * feature_indices,
      const bool* connected, const int radius, Eigen::Matrix3f* rf ){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < n) {
        PointType central_point = surface[feature_indices[index]];
        if (num_neighbors[index] > 5) {
            Eigen::MatrixXd vij(num_neighbors[index], 4);
            Eigen::Matrix3d cov_m = Eigen::Matrix3d::Zero();
            int valid_nn_points = 0;
            double sum = 0;
            for (int i = 0; i < N; i++) {
                if (connected[index * N + i]) {
                    PointType pt = surface[index];
                    vij.row(valid_nn_points).matrix() = Eigen::Vector4d(static_cast<double>(pt.x - central_point.x),
                        static_cast<double>(pt.y - central_point.y), static_cast<double>(pt.z - central_point.z), 0.0);;

                   double distance = radius - sqrt (vij(valid_nn_points,0)*vij(valid_nn_points,0)
                           + vij(valid_nn_points,1) * vij(valid_nn_points,1)
                           + vij(valid_nn_points,2) * vij(valid_nn_points,2));
                   cov_m += distance * (vij.row (valid_nn_points).head<3> ().transpose () * vij.row (valid_nn_points).head<3> ());
                   sum += distance;
                   valid_nn_points++;
                }
            }
            cov_m /= sum;
//
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver (cov_m);
//
            const double& e1c = solver.eigenvalues ()[0];
            const double& e2c = solver.eigenvalues ()[1];
            const double& e3c = solver.eigenvalues ()[2];

            if (isfinite (e1c) || isfinite (e2c) || isfinite (e3c)){
                 //Disambiguation
                Eigen::Vector4d v1 = Eigen::Vector4d::Zero ();
                Eigen::Vector4d v3 = Eigen::Vector4d::Zero ();
                v1.head<3> ().matrix () = solver.eigenvectors ().col (2);
                v3.head<3> ().matrix () = solver.eigenvectors ().col (0);

                int plusNormal = 0, plusTangentDirection1=0;
                for (int ne = 0; ne < valid_nn_points; ne++)
                {
                    double dp = vij.row (ne).dot (v1);
                    if (dp >= 0)
                        plusTangentDirection1++;

                    dp = vij.row (ne).dot (v3);
                    if (dp >= 0)
                        plusNormal++;
                }
                //TANGENT
                plusTangentDirection1 = 2*plusTangentDirection1 - valid_nn_points;
                if (plusTangentDirection1 == 0)
                {
                    int points = 5;
                    int medianIndex = valid_nn_points/2;

                    for (int i = -points/2; i <= points/2; i++)
                        if ( vij.row (medianIndex - i).dot (v1) > 0)
                            plusTangentDirection1 ++;

                    if (plusTangentDirection1 < points/2+1)
                        v1 *= - 1;
                }
                else if (plusTangentDirection1 < 0)
                    v1 *= - 1;

                //Normal
                plusNormal = 2*plusNormal - valid_nn_points;
                if (plusNormal == 0)
                {
                    int points = 5;
                    int medianIndex = valid_nn_points/2;

                    for (int i = -points/2; i <= points/2; i++)
                        if ( vij.row (medianIndex - i).dot (v3) > 0)
                            plusNormal ++;

                    if (plusNormal < points/2+1)
                        v3 *= - 1;
                } else if (plusNormal < 0)
                    v3 *= - 1;
                Eigen::Matrix3f rf_tmp;
                rf_tmp(0, 0) = static_cast<float>(v1[0]);
                rf_tmp(0, 1) = static_cast<float>(v1[1]);
                rf_tmp(0, 2) = static_cast<float>(v1[2]);
                rf_tmp(2, 0) = static_cast<float>(v3[0]);
                rf_tmp(2, 1) = static_cast<float>(v3[1]);
                rf_tmp(2, 2) = static_cast<float>(v3[2]);
                rf_tmp.row (1).matrix () = rf_tmp.row (2).cross (rf_tmp.row (0));
                rf[index] = rf_tmp;
            }

//
//
//
//
//
//        for(int i = 0; i < 3; ++i){
//            output[idx].x_axis = rf.rows(0)[d];
//            output[idx].y_axis = rf.rows(1)[d];
//            output[idx].z_axis = rf.rows(2)[d];
//        }
        }
    }
//                                Eigen::Vector4d vij = Eigen::Vector4d(static_cast<double>(pt.x - central_point.x),
//                        static_cast<double>(pt.y - central_point.y), static_cast<double>(pt.z - central_point.z), 0.0);
//
//                    double distance = radius - sqrt (vij[0]*vij[0] + vij[1] * vij[1] + vij[2] * vij[2]);
//                        // Multiply vij * vij'
//                    AtomicAdd(&cov[idx](0,0), vij[0] * vij[0]);
//                    AtomicAdd(&cov[idx](0,1), vij[0] * vij[1]);
//                    AtomicAdd(&cov[idx](0,2), vij[0] * vij[2]);
//
//                    AtomicAdd(&cov[idx](1,0), vij[1] * vij[0]);
//                    AtomicAdd(&cov[idx](1,1), vij[1] * vij[1]);
//                    AtomicAdd(&cov[idx](1,2), vij[1] * vij[2]);
//
//                    AtomicAdd(&cov[idx](2,0), vij[2] * vij[0]);
//                    AtomicAdd(&cov[idx](2,1), vij[2] * vij[1]);
//                    AtomicAdd(&cov[idx](2,2), vij[2] * vij[2]);
//                    printf(t)
//                    AtomicAdd(&sum[idx], distance);
//
}


void SHOT_LRF::computeDescriptor(pcl::PointCloud<pcl::ReferenceFrame> &output, const Eigen::Vector4f &inv_radius,
                                 const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi) {
    _N_features = static_cast<int>(_input->points.size());
    _N_surface = static_cast<int>(_surface->points.size());
    if (!_N_surface || !_N_features){
        std::cerr << "input not properly set up for shor lrf" << std::endl;
        exit(1);
    }

//    boost::function<int (const PointCloud<PointType> &cloud, size_t index, double,
//            std::vector<int> &, std::vector<float> &)> radius_search;
//    radius_search(*_surface, _kept_indices, _radius * 0.5, _neighbor_indices, _neighbor_indices);

    for (auto &n:_num_neighbors)
        std::cout << n << std::endl;
    std::cout << "printing finished" << std::endl;

    cudaMalloc((void**)&dev_features_indices, _N_features * sizeof(int));
    checkCUDAError("mallod dev_features_indices error");
    cudaMemcpy(dev_features_indices, &(*_kept_indices)[0], _N_features * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy dev_features_indices error");

    cudaMalloc((void**)&dev_pos_surface, _N_surface * sizeof(PointType));
    checkCUDAError("malloc dps error");
    cudaMemcpy(dev_pos_surface, &(_surface->points[0]), _N_surface * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy ps error");

    cudaMalloc((void**)&dev_num_neighbors, _N_features * sizeof(int));
    checkCUDAError("malloc num neighbors error");
    cudaMemset(dev_num_neighbors, 0, sizeof(int) * _N_features);
    checkCUDAError("memset num neighbors error");

//    cudaMalloc((void**)&dev_cov, _N_features * sizeof(Eigen::Matrix3d));
//    checkCUDAError("cudamalloc cov error");
//    //  TODO: change this to thrust implementation
//    thrust::device_ptr<Eigen::Matrix3d> dev_ptr(dev_cov);
//    thrust::fill(dev_ptr, dev_ptr + _N_features , Eigen::Matrix3d::Zero());
//    checkCUDAError("thrust memset cov error");
//
//    cudaMalloc((void**)&dev_sum, _N_features * sizeof(double));
//    checkCUDAError("malloc dev_sum error");
//    thrust::device_ptr<double >dev_ptr_sum(dev_sum);
//    thrust::fill(dev_ptr_sum, dev_ptr_sum + _N_features, 0.0);
//
    bool* dev_connected = NULL;
    cudaMalloc((void**)&dev_connected, _N_features * _N_surface * sizeof(bool));
    checkCUDAError("malloc dev connected error");
    thrust::device_ptr<bool> dev_ptr_connected(dev_connected);
    thrust::fill(dev_ptr_connected, dev_ptr_connected + _N_surface * _N_features, false);

    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((_N_surface + blockSize - 1)/blockSize));

    kernRadiusSearch<<<fullBlockPerGrid_points, blockSize, sizeof(int) * _N_features>>> (_N_surface, _N_features,
            dev_pos_surface, _radius, dev_features_indices, inv_radius, min_pi,  dev_num_neighbors, dev_connected);
    checkCUDAError("KernComputeCov error");
//
//
//
//    _num_neighbors.resize(_N_features);
//    cudaMemcpy(&_num_neighbors[0], dev_num_neighbors, sizeof(int) * _N_features, cudaMemcpyDeviceToHost);
//    checkCUDAError("cudamemcpy  num neigbors issue");
//
////
//    cudaFree(dev_num_neighbors);
//    checkCUDAError("dev_num free error");
//
//    _sum.resize(_N_features);
//    cudaMemcpy(&_sum[0], dev_sum, sizeof(double) * _N_features, cudaMemcpyDeviceToHost);
//    checkCUDAError("cudamemcpy error sum ");
//    cudaFree(dev_sum);
//    checkCUDAError("cudafree sum error");
//
//    _covs.resize(_N_features);
//    cudaMemcpy(&_covs[0], dev_cov, sizeof(Eigen::Matrix3d) * _N_features, cudaMemcpyDeviceToHost);
//    checkCUDAError("cudamemcpy dev_cov error");
//    cudaFree(dev_cov);
//    checkCUDAError("cudafree cov ");



}