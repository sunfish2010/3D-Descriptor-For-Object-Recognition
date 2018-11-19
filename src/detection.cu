#include "detection.h"

static int n_scene;
static int n_model;

static PointType *dev_ss_pc_scene = NULL;
static PointType *dev_ss_pc_model = NULL;
static PointType *dev_kp_scene = NULL;
static PointType *dev_kp_model = NULL;


__global__ void UniformDownSample(int N, float radius, PointType *pts_in, PointType *pts_out, int* new_size){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N){
        
    }

}

void detectionInit(const pcl::PointCloud<PointType >::ConstPtr &model, float radius){
    n_model = (int)(*model).size();
    dim3 fullBlocksPerGrid((n_model + blockSize - 1)/blockSize);
    cudaMalloc((void**) &dev_ss_pc_model, n_model * sizeof(PointType));
    cudaMemcpy(dev_ss_pc_model, &(*model).points[0], n_model * sizeof(PointType), cudaMemcpyHostToDevice);
    int n_model_ss = n_model;
    UniformDownSample <<<fullBlocksPerGrid, blockSize>>> (n_model, radius, dev_ss_pc_model, dev_ss_pc_model, &n_model_ss);
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