#include "detection.h"
#define VERBOSE 1

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

// uniform downsampling the points
__global__ void kernUniformDownSample(int N, float radius, PointType *pts_in, PointType *pts_out){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N){

    }
}

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

UniformDownSample::UniformDownSample(float radius): radius(radius), N_new(0), N(0){
    cudaMalloc((void**)&dev_min, sizeof(Eigen::Vector4f));
    cudaMalloc((void**)&dev_max, sizeof(Eigen::Vector4f));
    checkCUDAError("cudaMalloc min,max");
}

UniformDownSample::~UniformDownSample() {
    cudaFree(dev_min);
    cudaFree(dev_max);
    dev_min = NULL;
    dev_max = NULL;
}

void UniformDownSample::setRadius(float radius) {this->radius = radius;}

void UniformDownSample::downSample(const pcl::PointCloud<PointType >::ConstPtr input) {
        N = (int)(*input).size();
        dim3 fullBlockPerGrid_points ((N + blockSize - 1)/blockSize);
        static PointType *dev_pc = NULL;
        cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
        cudaMemcpy(dev_pc, &(*input).points[0], N * sizeof(PointType), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy pc");

        // calculate min max for the pc

        Eigen::Vector4f min_p, max_p;

        min_p.setConstant(FLT_MAX);
        max_p.setConstant(-FLT_MAX);
        cudaMemcpy(dev_min, &min_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_max, &max_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy min,max");

        getMinMax <<< fullBlockPerGrid_points, blockSize>>>(N, dev_pc, dev_min, dev_max);
        checkCUDAError("getMinMax error");
        cudaMemcpy(&min_p, dev_min, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
        cudaMemcpy(&max_p, dev_max, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
        Eigen::Vector4i max_pi(static_cast<int>(max_p[0]), static_cast<int>(max_p[1]), static_cast<int>(max_p[2]), 1);
        Eigen::Vector4i min_pi(static_cast<int>(min_p[0]), static_cast<int>(min_p[1]), static_cast<int>(min_p[2]), 1);

        // device the pc into cells

        Eigen::Vector4f inv_radius = Eigen::Array4f::Ones()/ (Eigen::Vector4f(radius, radius, radius, 1.0f).array());

        Eigen::Vector4f pc_dimension = max_p - min_p + Eigen::Vector4f::Ones();
        Eigen::Vector4i grid_count(static_cast<int>(floor(pc_dimension[0] * inv_radius[0]) + 1),
                                   static_cast<int>(floor(pc_dimension[1] * inv_radius[1]) + 1),
                                   static_cast<int>(floor(pc_dimension[2] * inv_radius[2]) + 1), 0);

        int total_grid_count = grid_count[0]* grid_count[1] * grid_count[2];

    #if VERBOSE
        std::cout << "The inverse radius is " << inv_radius << std::endl;
        std::cout << "The grid count is " << grid_count << std::endl;
        std::cout << "The min for each dimension is " << min_p << std::endl;
    #endif

        //kernUniformDownSample <<<fullBlockPerGrid_points, blockSize>>> (n_model, radius, dev_ss_pc_model, dev_ss_pc_model);
}
