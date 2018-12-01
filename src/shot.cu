#include "shot.h"


__global__ void computeBinDistShape(int N,const pcl::Normal* norms, const pcl::ReferenceFrame *lrf,
        double *bin_dist, int* neighbor_indices, const int n_bin, const int k){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        for (int i = 0; i < k; ++i){
            if (neighbor_indices[index * k + i] != -1){
                const Eigen::Vector4f& norm = norms[neighbor_indices[index * _k + i]];
                if (! isfinite(norm[0]) || !isfinite(norm[1] || !isfinite(norm[2]))){
                    bin_dist[index * k + i] = std::numeric_limits<double>::quiet_NaN();
                }else{
                    double cosDesc = norm[0] * lrf[index].z_axis[0] +
                                     norm[1] * lrf[index].z_axis[1] + norm[2] * lrf[index].z_axis[2];
                    if (cosDesc > 1) cosDesc = 1;
                    else if (cosDesc < -1) cosDesc = -1;
                    bin_dist[index * k + i] = ((1.0 + cosDesc) * n_bin) / 2;
                }
            }
            else
                bin_dist[index * k + i] = std::numeric_limits<double>::quiet_NaN();

        }
    }
}

__device__ rgb2lab(const unsigned char r, const unsigned char g, const unsigned char b, float &a, float &b, float &l){

}

__global__ computeBinColorShape(int N, const PointType* surface,double *bin_dist, const int* neighbor_indices,
        const int k){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        float L,A,B;
        rgb2lab(surface[index].r, surface[index].g, surface[index].b, A, B, L );
        for (int i = 0; i < k; ++i){
            if (neighbor_indices[index * k + i] != -1){

            }
        }
    }
}


void SHOT::computeDescriptor(const pcl::PointCloud<pcl::SHOT352> &output) {

    if (!output || !lrf){
        std::cerr << "Compute Descriptor shot not properly set up" << std::endl;
        exit(1);
    }
    descLength_ = nr_grid_sector_ * (nr_shape_bins_ + 1);

    sqradius_ = _radius * _radius;
    radius3_4_ = (_radius * 3) / 4;
    radius1_4_ = _radius / 4;
    radius1_2_ = _radius / 2;

    assert(descLength_ == 352);

    int N = static_cast<int> _input->points.size();
    dim3 numThreadsPerBlock = (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));

    int *dev_kept_indices = NULL;
    cudaMalloc((void**)&dev_kept_indices, N * sizeof(int));
    checkCUDAError("cuda malloc kept indices error");
    cudaMemcpy(dev_kept_indices, &(*_kept_indices)[0], N * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cuda memcpy kept_indices error");

    pcl::ReferenceFrame *dev_lrf = NULL;
    cudaMalloc((void**)&dev_lrf, N * sizeof (pcl::ReferenceFrame));
    checkCUDAError("cuda malloc dev_lrf error");
    cudaMemcpy(dev_lrf, &_lrf.points[0], N * sizeof(pcl::ReferenceFrame), cudaMemcpyHostToDevice);
    checkCUDAError("cuda Memcpy lrf error");

    int *dev_normals = NULL;
    int N_surface = static_cast<int>( _surface->points.size());
    cudaMalloc((void**)&dev_normals, sizeof(pcl::Normal) * N_surface);
    checkCUDAError("cuda malloc dev_normals error");
    cudaMemcpy(dev_normals, &_normals->points[0]. N_surface * sizeof(pcl::Normal), cudaMemcpyHostToDevice);
    checkCUDAError("cuda memcpy dev_normals error")';

    double *dev_bin_distance = NULL;
    cudaMalloc((void**)&dev_bin_distance, N * _k * sizeof(double));





}
