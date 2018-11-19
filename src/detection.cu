#include "detection.h"

static int n_scene;
static int n_model;

static PointType *dev_ss_pc_scene = NULL;
static PointType *dev_ss_pc_model = NULL;
static PointType *dev_pc = NULL;
static PointType *dev_kp_scene = NULL;
static PointType *dev_kp_model = NULL;


// uniform downsampling the points
__global__ void kernUniformDownSample(int N, float radius, PointType *pts_in, PointType *pts_out){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N){

    }
}

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

void UniformDownSample(int& N, const pcl::PointCloud<PointType >::ConstPtr input, bool mode){
    dim3 fullBlockPerGrid_points ((N + blockSize - 1)/blockSize);
    cudaMalloc((void**) &dev_pc, N * sizeof(PointType));
    cudaMemcpy(dev_pc, &(*input).points[0], n_model * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy pc");

    Eigen::Vector4f *dev_min = NULL;
    Eigen::Vector4f *dev_max = NULL;
    cudaMalloc((void**)&dev_min, sizeof(Eigen::Vector4f));
    cudaMalloc((void**)&dev_max, sizeof(Eigen::Vector4f));
    checkCUDAError("cudaMalloc min,max");
    Eigen::Vector4f min_p, max_p;

    min_p.setConstant(FLT_MAX);
    max_p.setConstant(-FLT_MAX);
    cudaMemcpy(dev_min, &min_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_max, &max_p, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy min,max");

    getMinMax <<< fullBlockPerGrid_points, blockSize>>>(N, dev_pc, dev_min, dev_max);
    checkCUDAError("getMinMax error");
    cudaMemcpy(&min_p, dev_min, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    std::cout << "The min for each dimension is " << min_p << std::endl;
    //kernUniformDownSample <<<fullBlockPerGrid_points, blockSize>>> (n_model, radius, dev_ss_pc_model, dev_ss_pc_model);
}

void detectionInit(const pcl::PointCloud<PointType >::ConstPtr model){
    n_model = (int)(*model).size();
    UniformDownSample(n_model, model, 0);

}

void detectFree(){
    cudaFree(dev_ss_pc_model);
    cudaFree(dev_ss_pc_scene);
    cudaFree(dev_kp_model);
    cudaFree(dev_kp_scene);

    dev_ss_pc_scene = NULL;
    dev_ss_pc_model = NULL;
    dev_kp_scene = NULL;
    dev_kp_model = NULL;

    checkCUDAError("cuda Free error");

}