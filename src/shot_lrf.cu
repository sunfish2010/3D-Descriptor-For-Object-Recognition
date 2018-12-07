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

/** \brief radius search
 * N : number of surface pts,
 * n : number of feature pts
 * feature indices: index of feature in original surface pt cloud
 * connected: bool array to indicate whether pt- pt is within radius
 * */

// TODO:: Change this to calculate directly cov, use N_features as threadIdx.x

__global__ void kernRadiusSearch(int N, int n, const PointType *surface, const float radius, const int *feature_indices,
        int* num_neighbors, bool* connected){
//    extern __shared__ int neighbor[];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
//    if (index < n) {
//        neighbor[idx] = 0;
//    }
//    __syncthreads();
    if (index < N) {
        PointType pt = surface[index];
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)) {

            for (int idx = 0; idx < n; ++idx) {
//                printf("N is %d, n is %d, number is %d, feature idx is %d\n",N, n, idx, feature_indices[idx]);
                PointType central_point = surface[feature_indices[idx]];


                if (fabs(central_point.x - pt.x) < radius  && fabs(central_point.y - pt.y) < radius
                && fabs(central_point.z - pt.z) < radius
                && !(pt.x == central_point.x && pt.y == central_point.y && pt.z == central_point.z)) {

                    atomicAdd(&num_neighbors[idx],1);
                    connected[idx * N + index] = true;

                }
            }
        }
    }
}


/** \brief vij^T * vij
 * */
__device__ Eigen::Matrix3d computevij(Eigen::Vector3d vij){
    Eigen::Matrix3d output;
    output(0,0) =  vij[0] * vij[0];
    output(0,1) =  vij[0] * vij[1];
    output(0,2) =  vij[0] * vij[2];

    output(1,0) =  vij[1] * vij[0];
    output(1,1) =  vij[1] * vij[1];
    output(1,2) =  vij[1] * vij[2];

    output(2,0) =  vij[2] * vij[0];
    output(2,1) =  vij[2] * vij[1];
    output(2,2) =  vij[2] * vij[2];
    return output;
}


/** \brief originally code for computing local reference directly
 * but calling selfAdjointEigen solver inside kernel always lead to invalid memory access
 * Therefore, this now only calculates covariances
 * N : number of surface pts,
 * n : number of feature pts
 * feature indices: index of feature in original surface pt cloud
 * connected: bool array to indicate whether pt- pt is within radius
 * */
__global__ void kernComputeLF(int N, int n, const PointType * surface,const int* num_neighbors,const int * feature_indices,
      const bool* connected, const int radius, Eigen::Vector3d* vij, Eigen::Matrix3d *covs ){
    const int max_n = 128;
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < n) {
        PointType central_point = surface[feature_indices[index]];
        if (num_neighbors[index] >= max_n) {
//            printf("num neighbor: %d, index is %d \n", num_neighbors[index], index);
            Eigen::Matrix3d cov_m = Eigen::Matrix3d::Zero();
            int valid_nn_points = 0;
            double sum = 0;
            for (int i = 0; i < N; i++) {
                if (connected[index * N + i]) {
                    PointType pt = surface[feature_indices[index]];
                    vij[max_n*index + valid_nn_points] = Eigen::Vector3d(static_cast<double>(pt.x - central_point.x),
                        static_cast<double>(pt.y - central_point.y), static_cast<double>(pt.z - central_point.z));

                   double distance = radius - sqrt (vij[max_n*index + valid_nn_points][0]*vij[max_n*index + valid_nn_points][0]
                           + vij[max_n*index + valid_nn_points][1] *vij[max_n*index + valid_nn_points][1]
                           + vij[max_n*index + valid_nn_points][2] * vij[max_n*index + valid_nn_points][2]);

                   cov_m += distance * computevij(vij[max_n*index + valid_nn_points]);
                   sum += distance;
                   valid_nn_points++;
                   if (valid_nn_points >=max_n)
                       break;
                }
            }
            cov_m /= sum;
            covs[index] = cov_m;
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


//    boost::function<int (const PointCloud<PointType> &cloud, size_t index, double,
//            std::vector<int> &, std::vector<float> &)> radius_search;
//    radius_search(*_surface, _kept_indices, _radius * 0.5, _neighbor_indices, _neighbor_indices);

    // Declare the search locator definition
//    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> _search_input(new(pcl::PointCloud<pcl::PointXYZ>));
//    pcl::copyPointCloud(*_surface, *_search_input);
//    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
//    kdtree.setInputCloud (_search_input);
//    kdtree.radiusSearch(_search_input->points[(*_kept_indices)[0]], _radius, _neighbor_indices, _neighbor_distances);
//    std::cout << "Num neighbors: " << _neighbor_distances.size() << std::endl;
    std::cout << "feature index " << (*_kept_indices)[0] << std::endl;

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

    Eigen::Matrix3f* dev_rf = NULL;
    cudaMalloc((void**)&dev_rf, _N_features * sizeof(Eigen::Matrix3f));
    checkCUDAError("malloc dev_rf error");

    Eigen::Vector3d *dev_vij = NULL;
    cudaMalloc((void**)&dev_vij, _N_features * _n * sizeof(Eigen::Vector3d));
    checkCUDAError("malloc dev_vij error");

    cudaMalloc((void**)&dev_cov, _N_features * sizeof(Eigen::Matrix3d));
    checkCUDAError("cudamalloc cov error");
    //  TODO: change this to thrust implementation
    thrust::device_ptr<Eigen::Matrix3d> dev_ptr(dev_cov);
    thrust::fill(dev_ptr, dev_ptr + _N_features , Eigen::Matrix3d::Zero());
    checkCUDAError("thrust memset cov error");
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
    checkCUDAError("thrust fill error");

    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((_N_surface + blockSize - 1)/blockSize));

    kernRadiusSearch<<<fullBlockPerGrid_points, blockSize>>> (_N_surface, _N_features,
            dev_pos_surface, _radius, dev_features_indices, dev_num_neighbors, dev_connected);
    checkCUDAError("KernComputeCov error");

    fullBlockPerGrid_points = dim3 (static_cast<u_int32_t >((_N_features + blockSize - 1)/blockSize));

    kernComputeLF<<<fullBlockPerGrid_points, blockSize>>> (_N_surface,_N_features, dev_pos_surface, dev_num_neighbors,
            dev_features_indices, dev_connected, _radius, dev_vij, dev_cov);
    checkCUDAError("kernCompute Local Reference error");

    _num_neighbors.resize(_N_features);
    cudaMemcpy(&_num_neighbors[0], dev_num_neighbors, sizeof(int) * _N_features, cudaMemcpyDeviceToHost);
    checkCUDAError("cuda memcpy dev_numneigh error");

//
    cudaFree(dev_pos_surface);
    cudaFree(dev_num_neighbors);
    cudaFree(dev_features_indices);
    cudaFree(dev_connected);
    checkCUDAError("cudafree dev_connected");


    _covs.resize(_N_features);
    cudaMemcpy(&_covs[0], dev_cov, sizeof(Eigen::Matrix3d) * _N_features, cudaMemcpyDeviceToHost);
    checkCUDAError("cudamemcpy dev_cov error");
    cudaFree(dev_cov);
    checkCUDAError("cudafree cov ");

    _vij.resize(_N_features * _n);
    cudaMemcpy(&_vij[0], dev_vij, sizeof(Eigen::Vector3d) * _N_features * _n, cudaMemcpyDeviceToHost);
    checkCUDAError("cudamemcpy dev_vji error");
    cudaFree(dev_vij);
    checkCUDAError("cudafree vij ");

    for (int idx = 0; idx <_N_features;idx++){
        Eigen::Matrix3f rf;
        if (_num_neighbors[idx] < _n)
        {
            //PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Eigenvectors are NaN. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
            rf.setConstant (std::numeric_limits<float>::quiet_NaN ());

            output.is_dense = false;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver (_covs[idx]);

        const double& e1c = solver.eigenvalues ()[0];
        const double& e2c = solver.eigenvalues ()[1];
        const double& e3c = solver.eigenvalues ()[2];

        if (!isfinite (e1c) || !isfinite (e2c) || !isfinite (e3c))
        {
            //PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Eigenvectors are NaN. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
            rf.setConstant (std::numeric_limits<float>::quiet_NaN ());

            output.is_dense = false;
        }

        // Disambiguation
        Eigen::Vector3d v1 = Eigen::Vector3d::Zero ();
        Eigen::Vector3d v3 = Eigen::Vector3d::Zero ();
        v1.matrix () = solver.eigenvectors ().col (2);
        v3.matrix () = solver.eigenvectors ().col (0);

        int plusNormal = 0, plusTangentDirection1=0;
        for (int ne = 0; ne < _n; ne++)
        {
            double dp = _vij[idx *_n + ne].dot (v1);
            if (dp >= 0)
                plusTangentDirection1++;

            dp = _vij[idx *_n + ne].dot (v3);
            if (dp >= 0)
                plusNormal++;
        }

        //TANGENT
        plusTangentDirection1 = 2*plusTangentDirection1 - _n;
        if (plusTangentDirection1 == 0)
        {
            int points = 5; //std::min(valid_nn_points*2/2+1, 11);
            int medianIndex = _n/2;

            for (int i = -points/2; i <= points/2; i++)
                if ( _vij[idx *_n + medianIndex - i].dot (v1) > 0)
                    plusTangentDirection1 ++;

            if (plusTangentDirection1 < points/2+1)
                v1 *= - 1;
        }
        else if (plusTangentDirection1 < 0)
            v1 *= - 1;

        //Normal
        plusNormal = 2*plusNormal - _n;
        if (plusNormal == 0)
        {
            int points = 5; //std::min(valid_nn_points*2/2+1, 11);
            int medianIndex = _n/2;

            for (int i = -points/2; i <= points/2; i++)
                if ( _vij[idx *_n + medianIndex - i].dot (v3) > 0)
                    plusNormal ++;

            if (plusNormal < points/2+1)
                v3 *= - 1;
        } else if (plusNormal < 0)
            v3 *= - 1;

        rf.row (0).matrix () = v1.head<3> ().cast<float> ();
        rf.row (2).matrix () = v3.head<3> ().cast<float> ();
        rf.row (1).matrix () = rf.row (2).cross (rf.row (0));

        for (int d = 0; d < 3; ++d)
        {
            output[idx].x_axis[d] = rf.row (0)[d];
            output[idx].y_axis[d] = rf.row (1)[d];
            output[idx].z_axis[d] = rf.row (2)[d];
        }
    }


}