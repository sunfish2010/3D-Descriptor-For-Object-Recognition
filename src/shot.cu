#include "shot.h"
#include "shot_lrf.h"

__global__ void computeBinDistShape(int N,const pcl::Normal* norms, const pcl::ReferenceFrame *lrf,
        double *bin_dist, int* neighbor_indices, const int n_bin, const int k){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        for (int i = 0; i < k; ++i){
            if (neighbor_indices[index * k + i] != -1){
                const pcl::Normal& norm = norms[neighbor_indices[index * k + i]];
                if (! isfinite(norm.normal_x) || !isfinite(norm.normal_y) || !isfinite(norm.normal_z)){
                    bin_dist[index * k + i] = NAN;
                }else{
                    double cosDesc = norm.normal_x * lrf[index].z_axis[0] +
                                     norm.normal_y * lrf[index].z_axis[1] + norm.normal_z * lrf[index].z_axis[2];
                    if (cosDesc > 1) cosDesc = 1;
                    else if (cosDesc < -1) cosDesc = -1;
                    bin_dist[index * k + i] = ((1.0 + cosDesc) * n_bin) / 2;
                }
            }
            else
                bin_dist[index * k + i] = NAN;

        }
    }
}

__device__ void rgb2lab(const float* LUT, const unsigned char r, const unsigned char g, const unsigned char b, float &a, float &b2, float &l){
    float x = (LUT[r] * 0.412453f + LUT[g] * 0.357580f + LUT[b] * 0.180423f) / 0.95047f;
    float y = LUT[r] * 0.212671f + LUT[g] * 0.715160f + LUT[b] * 0.072169f;
    float z = (LUT[r] * 0.019334f + LUT[g] * 0.119193f + LUT[b] * 0.950227f) / 1.08883f;

    x = LUT[int(x*4000) + 256];
    y = LUT[int(y*4000) + 256];
    z = LUT[int(z*4000) + 256];

    l = 116.0f * y - 16.0f;
    if (l > 100)
        l = 100.0f;

    a = 500.0f * (x - y);
    if (a > 120)
        a = 120.0f;
    else if (a <- 120)
        a = -120.0f;

    b2 = 200.0f * (y - z);
    if (b2 > 120)
        b2 = 120.0f;
    else if (b2< -120)
        b2 = -120.0f;

}

__global__ void computeBinColorShape(int N, const PointType* surface, double *bin_dist, const int* neighbor_indices,
        const int k, const int n_color_bin){
    // even if the same computation is performed many times, it should still be faster that global memory
     __shared__ float LUT[256 + 4000];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int num = (4000 + 256)/blockSize + 1;
    for (int i = 0; i < num; i++){
        int idx = num * threadIdx.x + i;
        if (idx  < 4000 + 256){
            if (idx < 256){
                float f = static_cast<float>(idx)/ 255.f;
                if (f > 0.04045)
                    LUT[idx] = powf((f + 0.055f)/1.055f, 2.4f);
                else
                    LUT[idx] = f / 12.92f;
            }else{
                float f = static_cast<float>(idx) / 4000.f;
                if (f > 0.008856)
                    LUT[idx] = powf(f, 0.3333f);
                else
                    LUT[idx] = (7.787 * f) + (16.0 / 116.0);
            }
        }
    }
    __syncthreads();

    if (index < N){
        float L,A,B;
        rgb2lab(LUT, surface[index].r, surface[index].g, surface[index].b, A, B, L );
        for (int i = 0; i < k; ++i){
            if (neighbor_indices[index * k + i] != -1){
                float l, a, b;
                int neighbor = neighbor_indices[index * k + i];
                rgb2lab(LUT, surface[neighbor].r, surface[neighbor].g, surface[neighbor].b, a, b, l);
                double color_dist = (fabs(L - l) + (fabs(A - a) + fabs(B - b))/2) / 3;
                color_dist = color_dist > 1.0? 1.0:color_dist;
                color_dist = color_dist < 0.0? 0.0:color_dist;
                bin_dist[index * k + i] = color_dist * n_color_bin;
            }else{
                bin_dist[index * k + i] = NAN;
            }
        }
    }
}


void SHOT::computeDescriptor(pcl::PointCloud<pcl::SHOT352> &output, const Eigen::Vector4f &inv_radius,
                             const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi) {

    descLength_ = nr_grid_sector_ * (nr_shape_bins_ + 1);

    sqradius_ = _radius * _radius;
    radius3_4_ = (_radius * 3) / 4;
    radius1_4_ = _radius / 4;
    radius1_2_ = _radius / 2;

    assert(descLength_ == 352);

    // compute local reference
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr reference (new pcl::PointCloud<pcl::ReferenceFrame>);
    SHOT_LRF lrf;
    lrf.setRadius(_radius);
    lrf.setInputCloud(_input);
    lrf.setSurface(_surface);
    lrf.setNormals(_normals);
    lrf.setKeptIndices(_kept_indices);
    lrf.compute(*reference, inv_radius, pc_dimension, min_pi);



    int N = static_cast<int> (_input->points.size());
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
    cudaMemcpy(dev_normals, &_normals->points[0], N_surface * sizeof(pcl::Normal), cudaMemcpyHostToDevice);
    checkCUDAError("cuda memcpy dev_normals error");

    double *dev_bin_distance = NULL;
    cudaMalloc((void**)&dev_bin_distance, N * _k * sizeof(double));





}
