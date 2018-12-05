//
// Created by sun on 11/26/18.
//

#include "shot_lrf.h"

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


__global__ void kernComputeCov(int N, int n, const PointType *surface, const float radius, const int *feature_indices,
        const Eigen::Vector4f inv_radius, const Eigen::Vector4i min_pi, int* num_neighbors, Eigen::Matrix3d* cov,
        double *sum){
    extern __shared__ uint8_t search_range[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < n) {
        PointType pt = surface[feature_indices[index]];
        search_range[index] = static_cast<u_int8_t >(floor((pt.x - radius) * inv_radius[0]) - min_pi[0]);
        search_range[index + 1] = static_cast<u_int8_t >(floor((pt.y - radius) * inv_radius[1]) - min_pi[1]);
        search_range[index + 2] = static_cast<u_int8_t >(floor((pt.z - radius) * inv_radius[2]) - min_pi[2]);
        search_range[index + 3] = static_cast<u_int8_t >(floor((pt.x + radius) * inv_radius[0]) - min_pi[0]);
        search_range[index + 4] = static_cast<u_int8_t >(floor((pt.y + radius) * inv_radius[1]) - min_pi[1]);
        search_range[index + 5] = static_cast<u_int8_t >(floor((pt.z + radius) * inv_radius[2]) - min_pi[2]);
    }
    __syncthreads();
    if (index < N) {
        PointType pt = surface[index];
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)) {
            uint8_t i = static_cast<uint8_t >(floor(pt.x * inv_radius[0]) - min_pi[0]);
            uint8_t j = static_cast<uint8_t >(floor(pt.y * inv_radius[1]) - min_pi[1]);
            uint8_t k = static_cast<uint8_t >(floor(pt.z * inv_radius[2]) - min_pi[2]);
//            float curr_dist = (pt.x - i) * (pt.x - i) + (pt.y - j) * (pt.y - j) + (pt.z - k) * (pt.z - k);
            for (int idx = 0; idx < n; ++idx) {
                PointType central_point = surface[feature_indices[idx]];
                double distance;
                if (i >= search_range[idx] && i <= search_range[idx + 3]
                    && j >= search_range[idx + 1] && j <= search_range[idx + 4]
                    && k >= search_range[idx + 2] && k <= search_range[idx + 5]
                    && !(pt.x == central_point.x && pt.y == central_point.y && pt.z == central_point.z)) {

                    Eigen::Vector4d vij = Eigen::Vector4d(static_cast<double>(pt.x - central_point.x),
                        static_cast<double>(pt.y - central_point.y), static_cast<double>(pt.z - central_point.z), 0.0);

                    distance = radius - sqrt (vij[0]*vij[0] + vij[1] * vij[1] + vij[2] * vij[2]);
                        // Multiply vij * vij'
                    AtomicAdd(&cov[idx](0,0), vij[0] * vij[0]);
                    AtomicAdd(&cov[idx](0,1), vij[0] * vij[1]);
                    AtomicAdd(&cov[idx](0,2), vij[0] * vij[2]);

                    AtomicAdd(&cov[idx](1,0), vij[1] * vij[0]);
                    AtomicAdd(&cov[idx](1,1), vij[1] * vij[1]);
                    AtomicAdd(&cov[idx](1,2), vij[1] * vij[2]);

                    AtomicAdd(&cov[idx](2,0), vij[2] * vij[0]);
                    AtomicAdd(&cov[idx](2,1), vij[2] * vij[1]);
                    AtomicAdd(&cov[idx](2,2), vij[2] * vij[2]);

                    atomicAdd(&num_neighbors[idx],1);
                    AtomicAdd(&sum[idx], distance);


                }
            }
        }
    }
}


void SHOT_LRF::computeDescriptor(pcl::PointCloud<pcl::ReferenceFrame> &output, const Eigen::Vector4f &inv_radius,
                                 const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi) {
    _N_features = static_cast<int>(_input->points.size());
    _N_surface = static_cast<int>(_surface->points.size());
    if (!_N_surface || !_N_features){
        std::cerr << "input not properly set up for shor lrf" << std::endl;
        exit(1);
    }

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

    cudaMalloc((void**)&dev_cov, _N_features * sizeof(Eigen::Matrix3d));
    checkCUDAError("cudamalloc cov error");
    //  TODO: change this to thrust implementation
    thrust::device_ptr<Eigen::Matrix3d> dev_ptr(dev_cov);
    thrust::fill(dev_ptr, dev_ptr + _N_features , Eigen::Matrix3d::Zero());
    checkCUDAError("thrust memset cov error");

    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((_N_surface + blockSize - 1)/blockSize));

    kernComputeCov<<<fullBlockPerGrid_points, blockSize, _N_features * sizeof(u_int8_t) * 6>>> (_N_surface, _N_features,
            dev_pos_surface, _radius, dev_features_indices, inv_radius, min_pi,  dev_num_neighbors, dev_cov, dev_sum);
    checkCUDAError("KernSearchRadius error");

//    _num_neighbors.resize(_N_features);
//    cudaMemcpy(&_num_neighbors[0], dev_num_neighbors, sizeof(int) * _N_features, cudaMemcpyDeviceToHost);
//    checkCUDAError("cudamemcpy  num neigbors issue");
//
//    _neighbor_indices.resize(_n * _N_features);
//    cudaMemcpy(&_neighbor_indices[0], dev_neighbor_indices, sizeof(int) * _N_features * _n, cudaMemcpyDeviceToHost);
//    checkCUDAError("cudamemcpy  num neigbors issue");
//
//    _neighbor_distances.resize(_n * _N_features);
//    cudaMemcpy(&_neighbor_distances[0], dev_distances, sizeof(float) * _N_features * _n, cudaMemcpyDeviceToHost);
//    checkCUDAError("cudamemcpy  distances issue");

//    Search nn_search;
//    nn_search.setRadius(_radius * 0.5);
//    nn_search.setSurface(_surface);
//    nn_search.setFeatures(_input);
//    nn_search.setFeaturesIndices(_kept_indices);
//    nn_search.search(inv_radius, pc_dimension, min_pi);
//    IndicesConstPtr numNeighbor = nn_search.getNumNeighbors();
//    IndicesConstPtr neighborIndices = nn_search.getNeighborIndices();
//       boost::shared_ptr<const std::vector<float>> neighborDist = nn_search.getNeighborDistance();


    // now assume that for each pt, we already knew the neighbors within radius
//    int N = static_cast<int>(_input->points.size());
//    // it seems that the calculation for local reference is best calculated using cpu, mainly copying implementation
//    //from pcl
//    output.points.resize(static_cast<u_int32_t >(N));
//    for (int idx = 0; idx < N; ++idx){
//        Eigen::Matrix3f rf;
//        Eigen::MatrixXd vij (_k, 4);
//
//        Eigen::Matrix3d cov_m = Eigen::Matrix3d::Zero ();
//        const Eigen::Vector4f& central_point = _input->points[idx].getVector4fMap();
//        double distance;
//        double sum = 0.0;
//        int valid_nn_points = 0;
//        for (int j = 0; j < _k; j++){
//            Eigen::Vector4f pt = _surface->points[(*_neighbor_indices)[j + idx * _k]].getVector4fMap();
//            if (pt.head<3> () == central_point.head<3>() || !isfinite(pt.x) || !isfinite(pt.y) || !isfinite(pt.z))
//                continue;
//            vij.row(valid_nn_points).matrix() = (pt - central_point).cast<double>();
//            vij(valid_nn_points, 3) = 0;
//
//            distance = _radius - sqrt (n_sqr_distances[idx]);
//
//            // Multiply vij * vij'
//            cov_m += distance * (vij.row (valid_nn_points).head<3> ().transpose () * vij.row (valid_nn_points).head<3> ());
//
//            sum += distance;
//            valid_nn_points++;
//        }
//
//        if (valid_nn_points < 5)
//        {
//            rf.setConstant (std::numeric_limits<float>::quiet_NaN ());
//            output->is_dense = false;
//            for(int i = 0; i < 3; ++i){
//                output[idx].x_axis = rf.rows(0)[i];
//                output[idx].y_axis = rf.rows(1)[i];
//                output[idx].z_axis = rf.rows(2)[i];
//            }
//            continue;
//
//        }
//
//        cov_m /= sum;
//
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver (cov_m);
//
//        const double& e1c = solver.eigenvalues ()[0];
//        const double& e2c = solver.eigenvalues ()[1];
//        const double& e3c = solver.eigenvalues ()[2];
//
//        if (!pcl_isfinite (e1c) || !pcl_isfinite (e2c) || !pcl_isfinite (e3c))
//        {
//            //PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Eigenvectors are NaN. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
//            rf.setConstant (std::numeric_limits<float>::quiet_NaN ());
//            output.is_dense = false;
//            for(int i = 0; i < 3; ++i){
//                output[idx].x_axis = rf.rows(0)[i];
//                output[idx].y_axis = rf.rows(1)[i];
//                output[idx].z_axis = rf.rows(2)[i];
//            }
//            continue;
//        }
//
//        // Disambiguation
//        Eigen::Vector4d v1 = Eigen::Vector4d::Zero ();
//        Eigen::Vector4d v3 = Eigen::Vector4d::Zero ();
//        v1.head<3> ().matrix () = solver.eigenvectors ().col (2);
//        v3.head<3> ().matrix () = solver.eigenvectors ().col (0);
//
//        int plusNormal = 0, plusTangentDirection1=0;
//        for (int ne = 0; ne < valid_nn_points; ne++)
//        {
//            double dp = vij.row (ne).dot (v1);
//            if (dp >= 0)
//                plusTangentDirection1++;
//
//            dp = vij.row (ne).dot (v3);
//            if (dp >= 0)
//                plusNormal++;
//        }
//        //TANGENT
//        plusTangentDirection1 = 2*plusTangentDirection1 - valid_nn_points;
//        if (plusTangentDirection1 == 0)
//        {
//            int points = 5; //std::min(valid_nn_points*2/2+1, 11);
//            int medianIndex = valid_nn_points/2;
//
//            for (int i = -points/2; i <= points/2; i++)
//                if ( vij.row (medianIndex - i).dot (v1) > 0)
//                    plusTangentDirection1 ++;
//
//            if (plusTangentDirection1 < points/2+1)
//                v1 *= - 1;
//        }
//        else if (plusTangentDirection1 < 0)
//            v1 *= - 1;
//
//        //Normal
//        plusNormal = 2*plusNormal - valid_nn_points;
//        if (plusNormal == 0)
//        {
//            int points = 5; //std::min(valid_nn_points*2/2+1, 11);
//            int medianIndex = valid_nn_points/2;
//
//            for (int i = -points/2; i <= points/2; i++)
//                if ( vij.row (medianIndex - i).dot (v3) > 0)
//                    plusNormal ++;
//
//            if (plusNormal < points/2+1)
//                v3 *= - 1;
//        } else if (plusNormal < 0)
//            v3 *= - 1;
//
//        rf.row (0).matrix () = v1.head<3> ().cast<float> ();
//        rf.row (2).matrix () = v3.head<3> ().cast<float> ();
//        rf.row (1).matrix () = rf.row (2).cross (rf.row (0));
//
//        for(int i = 0; i < 3; ++i){
//            output[idx].x_axis = rf.rows(0)[d];
//            output[idx].y_axis = rf.rows(1)[d];
//            output[idx].z_axis = rf.rows(2)[d];
//        }
//    }



}