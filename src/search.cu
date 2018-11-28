#include "search.h"


/*
 * Atomic functions for float
 * https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
 */
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val){
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


/** \brief common functions for search  */
__global__ void getMinMax(int N ,const PointType *pts_in, Eigen::Vector4f *min_pt, Eigen::Vector4f  *max_pt){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N){

        PointType pt = pts_in[index] ;
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
            //float pt_x = pt.x, pt_y = pt.y, pt_z = pt.z;
            atomicMin(&(*min_pt)[0], pt.x);
            atomicMin(&(*min_pt)[1], pt.y);
            atomicMin(&(*min_pt)[2], pt.z);
            atomicMax(&(*max_pt)[0], pt.x);
            atomicMax(&(*max_pt)[1], pt.y);
            atomicMax(&(*max_pt)[2], pt.z);
        }
    }
}


/** \brief 3d indices to 1d  */
__device__ int kernComputeIndices(Eigen::Vector4i pos, Eigen::Vector4i grid_res){
    return pos[0] + pos[1] * grid_res[0] + pos[2] * grid_res[1] * grid_res[2];
}


/** \brief common functions for search  */
__global__ void kernComputeIndices(int N, Eigen::Vector4i grid_res, Eigen::Vector4i grid_min,
                                   Eigen::Vector4f inv_radius, PointType *pos, int *indices, int *grid_indices){
    int index = threadIdx.x + (blockIdx.x *blockDim.x);
    if (index < N){
        PointType pt = pos[index] ;
        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)){
            Eigen::Vector4i ijk(static_cast<int>(floor(pt.x * inv_radius[0])),
                                static_cast<int>(floor(pt.y * inv_radius[1])), static_cast<int>(floor(pt.z * inv_radius[2])), 0);


            Eigen::Vector4i offset = ijk - grid_min;
//            printf("offset is %d, %d, %d \n",offset[0], offset[1], offset[2]);
//            printf("grid res is %d, %d, %d \n", grid_res[0], grid_res[1], grid_res[2]);
            grid_indices[index] = kernComputeIndices(offset, grid_res);
//            printf("indice is %d \n", grid_indices[index] );
            indices[index] = index;
        }

    }
}



__global__ void kernSearchRadius(int N, int *feature_indices, int * output, PointType *surface) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < N){

    }
}


Search::~Search() {
    cudaFree(dev_pos_surface);
    cudaFree(dev_neighbor_indices);
    cudaFree(dev_features_indices);
    cudaFree(dev_grid_indices);
    cudaFree(dev_array_indices);
    cudaFree(dev_min);
    cudaFree(dev_max);
    dev_neighbor_indices = NULL;
    dev_features_indices = NULL;
    dev_pos_surface = NULL;
    dev_array_indices = NULL;
    dev_grid_indices = NULL;
    dev_max = NULL;
    dev_min = NULL;
}


void Search::initSearch(float radius) {
    if (!_surface){
        std::cerr << "Must set up surface input "  << std::endl;
        exit(1);
    }



    dim3 fullBlockPerGrid_points (static_cast<u_int32_t >((_N_surface + blockSize - 1)/blockSize));
    cudaMalloc((void**) &dev_pos_surface, _N_surface * sizeof(PointType));
    cudaMemcpy(dev_pos_surface, &(*_surface).points[0], _N_surface * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy pc");

    // calculate min max for the pc
    Eigen::Vector4f min_p, max_p;
    min_p.setConstant(FLT_MAX);
    max_p.setConstant(-FLT_MAX);
    cudaMemcpy(dev_min, &min_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_max, &max_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy min,max");

    getMinMax <<< fullBlockPerGrid_points, blockSize>>>(_N_surface, dev_pos_surface, dev_min, dev_max);
    checkCUDAError("getMinMax error");
    cudaMemcpy(&min_p, dev_min, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy min  error");
    cudaMemcpy(&max_p, dev_max, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy max error");

    // devide the pc into cells
    inv_radius = Eigen::Array4f::Ones()/ (Eigen::Vector4f(radius, radius, radius, 1.0f).array());
    max_pi = Eigen::Vector4i(static_cast<int>(floor(max_p[0] * inv_radius[0])),
                           static_cast<int>(floor(max_p[1] * inv_radius[1])), static_cast<int>(floor(max_p[2] * inv_radius[2])), 0);
    min_pi = Eigen::Vector4i(static_cast<int>(floor(min_p[0] * inv_radius[0])),
                           static_cast<int>(floor(min_p[1] * inv_radius[1])), static_cast<int>(floor(inv_radius[2] * min_p[2])), 0);


    pc_dimension = max_pi - min_pi + Eigen::Vector4i::Ones();
    pc_dimension[3] = 0;

    _grid_count = pc_dimension[0] * pc_dimension[1] * pc_dimension[2];

    //Eigen::Vector4i grid_res = Eigen::Vector4i(1, pc_dimension[0], pc_dimension[0] * pc_dimension[1], 0);

    cudaMalloc((void**)&dev_grid_indices, _N_surface * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");
    cudaMalloc((void**)&dev_array_indices, _N_surface * sizeof(int));
    checkCUDAError("cudaMalloc dev_indices error");

    kernComputeIndices <<< fullBlockPerGrid_points, blockSize >>> (_N_surface, pc_dimension, min_pi,
            inv_radius, dev_pos_surface, dev_array_indices, dev_grid_indices);
    checkCUDAError("kernComputeIndices Failed");
}


//void get


void Search::search(const pcl::PointCloud<PointType>::Ptr &output) {
    if (_method == SearchMethod::KDTree) {
        std::cout << "Function not implemented yet" << std::endl;
        exit(1);
    }
    if (_method == SearchMethod::Radius){
//        cudaMalloc((void**)&dev_features_indices, _N_features * sizeof(int));
//        checkCUDAError("mallod dev_features_indices error");
//        cudaMemcpy(dev_features_indices, &(*_feature_indices)[0], _N_features * sizeof(int), cudaMemcpyHostToDevice);
//        checkCUDAError("memcpy dev_features_indices error");
//        cudaMalloc((void**)&dev_neighbor_indices, _N_features * _n * sizeof(int));
//        checkCUDAError("malloc dev_neighbor indices error");
//        cudaMemset(dev_neighbor_indices, -1, _N_features * _n * sizeof(int));
//        checkCUDAError("memset ni error");
//        cudaMalloc((void**)&dev_pos_surface, _N_surface * sizeof(PointType));
//        checkCUDAError("malloc dps error");
//        cudaMemcpy(dev_pos_surface, &(_surface->points[0]), _N_surface * sizeof(PointType), cudaMemcpyHostToDevice);
//        checkCUDAError("memcpy ps error");
//        cudaMalloc((void**)&dev_grid_indices,_N_surface * sizeof(int));
//        checkCUDAError("malloc gi failed");
//        cudaMemcpy(dev_grid_indices, &(*_grid_indices)[0], _N_surface * sizeof(int), cudaMemcpyHostToDevice);
//        checkCUDAError("memcpy gi failed");

    }

}