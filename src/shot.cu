#include "shot.h"
#include "shot_lrf.h"

/**  Reference
 *   - F. Tombari, S. Salti, L. Di Stefano
 *     Unique Signatures of Histograms for Local Surface Description.
 *     In Proceedings of the 11th European Conference on Computer Vision (ECCV),
         *     Heraklion, Greece, September 5-11 2010.
 *   - F. Tombari, S. Salti, L. Di Stefano
 *     A Combined Texture-Shape Descriptor For Enhanced 3D Feature Matching.
 *     In Proceedings of the 18th International Conference on Image Processing (ICIP),
        *     Brussels, Belgium, September 11-14 2011.
 *
 * \author Samuele Salti, Federico Tombari
 *
 *
 * Implemented based on PCL & Original Paper's Author's implementation, modified to run on CUDA
 *
 *
 */

//
//__global__ void computeBinDistShape(int N,const pcl::Normal* norms, const pcl::ReferenceFrame *lrf,
//        double *bin_dist, int* neighbor_indices, const int n_bin, const int k){
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    if (index < N){
//        for (int i = 0; i < k; ++i){
//            if (neighbor_indices[index * k + i] != -1){
//                const pcl::Normal& norm = norms[neighbor_indices[index * k + i]];
//                if (! isfinite(norm.normal_x) || !isfinite(norm.normal_y) || !isfinite(norm.normal_z)){
//                    bin_dist[index * k + i] = NAN;
//                }else{
//                    double cosDesc = norm.normal_x * lrf[index].z_axis[0] +
//                                     norm.normal_y * lrf[index].z_axis[1] + norm.normal_z * lrf[index].z_axis[2];
//                    if (cosDesc > 1) cosDesc = 1;
//                    else if (cosDesc < -1) cosDesc = -1;
//                    bin_dist[index * k + i] = ((1.0 + cosDesc) * n_bin) / 2;
//                }
//            }
//            else
//                bin_dist[index * k + i] = NAN;
//
//        }
//    }
//}

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

//__global__ void computeBinColorShape(int N, const PointType* surface, double *bin_dist, const int* neighbor_indices,
//        const int k, const int n_color_bin){
//    // even if the same computation is performed many times, it should still be faster that global memory
//     __shared__ float LUT[256 + 4000];
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    int num = (4000 + 256)/blockSize + 1;
//    for (int i = 0; i < num; i++){
//        int idx = num * threadIdx.x + i;
//        if (idx  < 4000 + 256){
//            if (idx < 256){
//                float f = static_cast<float>(idx)/ 255.f;
//                if (f > 0.04045)
//                    LUT[idx] = powf((f + 0.055f)/1.055f, 2.4f);
//                else
//                    LUT[idx] = f / 12.92f;
//            }else{
//                float f = static_cast<float>(idx) / 4000.f;
//                if (f > 0.008856)
//                    LUT[idx] = powf(f, 0.3333f);
//                else
//                    LUT[idx] = (7.787f * f) + (16.f / 116.f);
//            }
//        }
//    }
//    __syncthreads();
//
//    if (index < N){
//        float L,A,B;
//        rgb2lab(LUT, surface[index].r, surface[index].g, surface[index].b, A, B, L );
//        for (int i = 0; i < k; ++i){
//            if (neighbor_indices[index * k + i] != -1){
//                float l, a, b;
//                int neighbor = neighbor_indices[index * k + i];
//                rgb2lab(LUT, surface[neighbor].r, surface[neighbor].g, surface[neighbor].b, a, b, l);
//                double color_dist = (fabs(L - l) + (fabs(A - a) + fabs(B - b))/2) / 3;
//                color_dist = color_dist > 1.0? 1.0:color_dist;
//                color_dist = color_dist < 0.0? 0.0:color_dist;
//                bin_dist[index * k + i] = color_dist * n_color_bin;
//            }else{
//                bin_dist[index * k + i] = NAN;
//            }
//        }
//    }
//}


inline __device__ bool areEquals (double val1, double val2)
{
    return (fabs (val1 - val2)<1E-15);
}


inline __device__ bool areEquals (float val1, float val2)
{
    return (fabs (val1 - val2)< 1E-8f);
}



__global__ void computeCOLORSHOT(int N, int n, const PointType *surface, const float radius, const int *feature_indices,
        const pcl::Normal* norms, const pcl::ReferenceFrame *lrf, const int n_color_bin, const int n_dist_bin,
        const int nr_grid_sector_, float* shot ){
    // even if the same computation is performed many times, it should still be faster that global memory
    __shared__ float LUT[256 + 4000];
    const double PST_PI = 3.1415926535897932384626433832795;
    const double PST_RAD_45 = 0.78539816339744830961566084581988;
    const double PST_RAD_90 = 1.5707963267948966192313216916398;
    const double PST_RAD_135 = 2.3561944901923449288469825374596;
    const double PST_RAD_180 = PST_PI;
//    const double PST_RAD_360 = 6.283185307179586476925286766558;
    const double PST_RAD_PI_7_8 = 2.7488935718910690836548129603691;
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
                    LUT[idx] = (7.787f * f) + (16.f / 116.f);
            }
        }
    }
    __syncthreads();
    int shapeToColorStride = nr_grid_sector_*(n_dist_bin+1);
//    double sqradius_ = radius * radius;
    double radius3_4_ = (radius * 3) / 4;
    double radius1_4_ = radius / 4;
    double radius1_2_ = radius / 2;
    const int descLength_ = 1344;
    const int maxAngularSectors_ = 32;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n){
        int num_neighbors = 0;
        PointType central_point = surface[feature_indices[index]];
        pcl::ReferenceFrame current_frame = lrf[index];


        Eigen::Vector3f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2]);
        Eigen::Vector3f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2]);
        Eigen::Vector3f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2]);
        float L,A,B;
        rgb2lab(LUT, central_point.r, central_point.g, central_point.b, A, B, L );
        double bin_color_dist[1000];
        double bin_dist[1000];
        for(int idx = 0; idx < N;  idx++){
            PointType pt = surface[idx];
            if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)) {
                if (fabs(central_point.x - pt.x) < radius  && fabs(central_point.y - pt.y) < radius
                    && fabs(central_point.z - pt.z) < radius
                    && !(pt.x == central_point.x && pt.y == central_point.y && pt.z == central_point.z)){

                    float l, a, b;
                    rgb2lab(LUT, pt.r, pt.g, pt.b, a, b, l);
                    double color_dist = (fabs(L - l) + (fabs(A - a) + fabs(B - b))/2) / 3;
                    color_dist = color_dist > 1.0? 1.0:color_dist;
                    color_dist = color_dist < 0.0? 0.0:color_dist;
                    bin_color_dist[num_neighbors] = color_dist * n_color_bin;

                    const pcl::Normal& norm = norms[idx];
                    if (! isfinite(norm.normal_x) || !isfinite(norm.normal_y) || !isfinite(norm.normal_z)){
                        bin_dist[num_neighbors] = NAN;
                    }else{
                        double cosDesc = norm.normal_x * lrf[index].z_axis[0] +
                                         norm.normal_y * lrf[index].z_axis[1] + norm.normal_z * lrf[index].z_axis[2];
                        if (cosDesc > 1) cosDesc = 1;
                        else if (cosDesc < -1) cosDesc = -1;
                        bin_dist[num_neighbors] = ((1.0 + cosDesc) * n_dist_bin) / 2;
                    }

                    int offset = descLength_ * index;
                    Eigen::Vector3f delta(pt.x- central_point.x, pt.y - central_point.y, pt.z - central_point.z);

                    double distance = sqrt((central_point.x - pt.x) * (central_point.x - pt.x) +
                        (central_point.y - pt.y) * (central_point.y - pt.y) +
                        (central_point.z - pt.z)*(central_point.z - pt.z));
                    if (areEquals (distance, 0.0))
                        continue;

                    double xInFeatRef = delta.dot (current_frame_x);
                    double yInFeatRef = delta.dot (current_frame_y);
                    double zInFeatRef = delta.dot (current_frame_z);

                    // To avoid numerical problems afterwards
                    if (fabs (yInFeatRef) < 1E-30)
                        yInFeatRef  = 0;
                    if (fabs (xInFeatRef) < 1E-30)
                        xInFeatRef  = 0;
                    if (fabs (zInFeatRef) < 1E-30)
                        zInFeatRef = 0;

                    unsigned char bit4 = ((yInFeatRef > 0) || ((yInFeatRef == 0.0) && (xInFeatRef < 0))) ? 1 : 0;
                    unsigned char bit3 = static_cast<unsigned char> (((xInFeatRef > 0) || ((xInFeatRef == 0.0)
                            && (yInFeatRef > 0))) == !bit4 );

                    assert (bit3 == 0 || bit3 == 1);

                    int desc_index = (bit4<<3) + (bit3<<2);

                    desc_index = desc_index << 1;

                    if ((xInFeatRef * yInFeatRef > 0) || (xInFeatRef == 0.0))
                        desc_index += (fabs (xInFeatRef) >= fabs (yInFeatRef)) ? 0 : 4;
                    else
                        desc_index += (fabs (xInFeatRef) > fabs (yInFeatRef)) ? 4 : 0;

                    desc_index += zInFeatRef > 0 ? 1 : 0;

                    // 2 RADII
                    desc_index += (distance > radius1_2_) ? 2 : 0;

                    int step_index_shape = static_cast<int>(floor (bin_dist[num_neighbors] +0.5));
                    int step_index_color = static_cast<int>(floor (bin_color_dist[num_neighbors] +0.5));

                    int volume_index_shape = desc_index * (n_dist_bin+1);
                    int volume_index_color = shapeToColorStride + desc_index * (n_color_bin+1);

                    //Interpolation on the cosine (adjacent bins in the histrogram)
                    bin_dist[num_neighbors] -= step_index_shape;
                    bin_color_dist[num_neighbors] -= step_index_color;

                    double intWeightShape = (1- fabs (bin_dist[num_neighbors]));
                    double intWeightColor = (1- fabs (bin_color_dist[num_neighbors]));

                    if (bin_dist[num_neighbors] > 0)
                        shot[offset + volume_index_shape + ((step_index_shape + 1) % n_dist_bin)] +=
                                static_cast<float> (bin_dist[num_neighbors]);
                    else
                        shot[offset + volume_index_shape + ((step_index_shape - 1 + n_dist_bin) % n_dist_bin)] -=
                                static_cast<float> (bin_dist[num_neighbors]);

                    if (bin_color_dist[num_neighbors] > 0)
                        shot[offset + volume_index_color + ((step_index_color+1) % n_color_bin)] +=
                                static_cast<float> (bin_color_dist[num_neighbors]);
                    else
                        shot[offset + volume_index_color + ((step_index_color - 1 + n_color_bin) % n_color_bin)] -=
                                static_cast<float> (bin_color_dist[num_neighbors]);

                    //Interpolation on the distance (adjacent husks)

                    if (distance > radius1_2_)   //external sphere
                    {
                        double radiusDistance = (distance - radius3_4_) / radius1_2_;

                        if (distance > radius3_4_) //most external sector, votes only for itself
                        {
                            intWeightShape += 1 - radiusDistance; //weight=1-d
                            intWeightColor += 1 - radiusDistance; //weight=1-d
                        }
                        else  //3/4 of radius, votes also for the internal sphere
                        {
                            intWeightShape += 1 + radiusDistance;
                            intWeightColor += 1 + radiusDistance;
                            shot[offset + (desc_index - 2) * (n_dist_bin+1) + step_index_shape] -=
                                    static_cast<float> (radiusDistance);
                            shot[offset + shapeToColorStride + (desc_index - 2) * (n_color_bin+1) + step_index_color] -=
                                    static_cast<float> (radiusDistance);
                        }
                    }
                    else    //internal sphere
                    {
                        double radiusDistance = (distance - radius1_4_) / radius1_2_;

                        if (distance < radius1_4_) //most internal sector, votes only for itself
                        {
                            intWeightShape += 1 + radiusDistance;
                            intWeightColor += 1 + radiusDistance;
                        }
                        else  //3/4 of radius, votes also for the external sphere
                        {
                            intWeightShape += 1 - radiusDistance;
                            intWeightColor += 1 - radiusDistance;
                            shot[offset + (desc_index + 2) * (n_dist_bin+1) + step_index_shape] +=
                                    static_cast<float> (radiusDistance);
                            shot[offset + shapeToColorStride + (desc_index + 2) * (n_color_bin+1) + step_index_color] +=
                                    static_cast<float> (radiusDistance);
                        }
                    }

                    //Interpolation on the inclination (adjacent vertical volumes)
                    double inclinationCos = zInFeatRef / distance;
                    if (inclinationCos < - 1.0)
                        inclinationCos = - 1.0;
                    if (inclinationCos > 1.0)
                        inclinationCos = 1.0;

                    double inclination = acos (inclinationCos);

                    assert (inclination >= 0.0 && inclination <= PST_RAD_180);

                    if (inclination > PST_RAD_90 || (fabs (inclination - PST_RAD_90) < 1e-30 && zInFeatRef <= 0))
                    {
                        double inclinationDistance = (inclination - PST_RAD_135) / PST_RAD_90;
                        if (inclination > PST_RAD_135)
                        {
                            intWeightShape += 1 - inclinationDistance;
                            intWeightColor += 1 - inclinationDistance;
                        }
                        else
                        {
                            intWeightShape += 1 + inclinationDistance;
                            intWeightColor += 1 + inclinationDistance;
//                            assert ((desc_index + 1) * (n_dist_bin+1) + step_index_shape >= 0 && (desc_index + 1) * (n_dist_bin+1) + step_index_shape < descLength_);
//                            assert (shapeToColorStride + (desc_index + 1) * (n_color_bin+ 1) + step_index_color >= 0 && shapeToColorStride + (desc_index + 1) * (n_color_bin+1) + step_index_color < descLength_);
                            shot[offset + (desc_index + 1) * (n_dist_bin+1) + step_index_shape] -=
                                    static_cast<float> (inclinationDistance);
                            shot[offset + shapeToColorStride + (desc_index + 1) * (n_color_bin+1) + step_index_color] -=
                                    static_cast<float> (inclinationDistance);
                        }
                    }
                    else
                    {
                        double inclinationDistance = (inclination - PST_RAD_45) / PST_RAD_90;
                        if (inclination < PST_RAD_45)
                        {
                            intWeightShape += 1 + inclinationDistance;
                            intWeightColor += 1 + inclinationDistance;
                        }
                        else
                        {
                            intWeightShape += 1 - inclinationDistance;
                            intWeightColor += 1 - inclinationDistance;
//                            if (!((desc_index - 1) * (n_dist_bin+1) + step_index_shape >= 0 && (desc_index - 1) * (n_dist_bin+1) + step_index_shape < descLength_)){
//                                printf("desc_index is %d, step_index_shape is %d, n_dist_bin: %d, num_neighbors %d, bin_dist: %f  \n",
//                                        desc_index, step_index_shape, n_dist_bin, num_neighbors, bin_dist[num_neighbors]);
//                            }

//                            assert ((desc_index - 1) * (n_dist_bin+1) + step_index_shape >= 0 && (desc_index - 1) * (n_dist_bin+1) + step_index_shape < descLength_);
//                            assert (shapeToColorStride + (desc_index - 1) * (n_color_bin+ 1) + step_index_color >= 0 && shapeToColorStride + (desc_index - 1) * (n_color_bin+1) + step_index_color < descLength_);
                            shot[offset + (desc_index - 1) * (n_dist_bin+1) + step_index_shape] +=
                                    static_cast<float> (inclinationDistance);
                            shot[offset + shapeToColorStride + (desc_index - 1) * (n_color_bin+1) + step_index_color] +=
                                    static_cast<float> (inclinationDistance);
                        }
                    }

                    if (yInFeatRef != 0.0 || xInFeatRef != 0.0)
                    {
                        //Interpolation on the azimuth (adjacent horizontal volumes)
                        double azimuth = atan2 (yInFeatRef, xInFeatRef);

                        int sel = desc_index >> 2;
                        double angularSectorSpan = PST_RAD_45;
                        double angularSectorStart = - PST_RAD_PI_7_8;

                        double azimuthDistance = (azimuth - (angularSectorStart + angularSectorSpan*sel)) / angularSectorSpan;
//                        assert ((azimuthDistance < 0.5 || areEquals (azimuthDistance, 0.5)) && (azimuthDistance > - 0.5 || areEquals (azimuthDistance, - 0.5)));
                        azimuthDistance = max(- 0.5, min (azimuthDistance, 0.5));

                        if (azimuthDistance > 0)
                        {
                            intWeightShape += 1 - azimuthDistance;
                            intWeightColor += 1 - azimuthDistance;
                            int interp_index = (desc_index + 4) % maxAngularSectors_;
//                            assert (interp_index * (n_dist_bin+1) + step_index_shape >= 0 && interp_index * (n_dist_bin+1) + step_index_shape < descLength_);
//                            assert (shapeToColorStride + interp_index * (n_color_bin+1) + step_index_color >= 0 && shapeToColorStride + interp_index * (n_color_bin+1) + step_index_color < descLength_);
                            shot[offset + interp_index * (n_dist_bin+1) + step_index_shape] +=
                                    static_cast<float> (azimuthDistance);
                            shot[offset + shapeToColorStride + interp_index * (n_color_bin+1) + step_index_color] +=
                                    static_cast<float> (azimuthDistance);
                        }
                        else
                        {
                            int interp_index = (desc_index - 4 + maxAngularSectors_) % maxAngularSectors_;
                            intWeightShape += 1 + azimuthDistance;
                            intWeightColor += 1 + azimuthDistance;
//                            assert (interp_index * (n_dist_bin+1) + step_index_shape >= 0 && interp_index * (n_dist_bin+1) + step_index_shape < descLength_);
//                            assert (shapeToColorStride + interp_index * (n_color_bin+1) + step_index_color >= 0 && shapeToColorStride + interp_index * (n_color_bin+1) + step_index_color < descLength_);
                            shot[offset + interp_index * (n_dist_bin+1) + step_index_shape] -=
                                    static_cast<float> (azimuthDistance);
                            shot[offset + shapeToColorStride + interp_index * (n_color_bin+1) + step_index_color] -=
                                    static_cast<float> (azimuthDistance);
                        }
                    }

                    assert (volume_index_shape + step_index_shape >= 0 &&  volume_index_shape + step_index_shape < descLength_);
                    assert (volume_index_color + step_index_color >= 0 &&  volume_index_color + step_index_color < descLength_);
                    shot[offset + volume_index_shape + step_index_shape] += static_cast<float> (intWeightShape);
                    shot[offset + volume_index_color + step_index_color] += static_cast<float> (intWeightColor);

                    num_neighbors++;
                    if (num_neighbors >= 1000) break;
                }
            }

        }
    }

}



__global__ void computeSHOT(int N, const PointType *surface, const float radius, const int *feature_indices, const int* array_indices,
                     const int* grid_start_indices, const int* grid_end_indices, const Eigen::Vector4i grid_res,
                     const Eigen::Vector4i grid_min, const Eigen::Vector4i grid_max, const Eigen::Vector4f inv_radius,
                     const pcl::Normal* norms, const pcl::ReferenceFrame *lrf,  const int n_dist_bin, float* shot ){
    // even if the same computation is performed many times, it should still be faster that global memory
    const double PST_PI = 3.1415926535897932384626433832795;
    const double PST_RAD_45 = 0.78539816339744830961566084581988;
    const double PST_RAD_90 = 1.5707963267948966192313216916398;
    const double PST_RAD_135 = 2.3561944901923449288469825374596;
    const double PST_RAD_180 = PST_PI;
//    const double PST_RAD_360 = 6.283185307179586476925286766558;
    const double PST_RAD_PI_7_8 = 2.7488935718910690836548129603691;

//    double sqradius_ = radius * radius;
    double radius3_4_ = (radius * 3) / 4;
    double radius1_4_ = radius / 4;
    double radius1_2_ = radius / 2;
    const int descLength_ = 352;
    const int maxAngularSectors_ = 32;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        int num_neighbors = 0;
        int offset = descLength_ * index;
        PointType central_point = surface[feature_indices[index]];
        pcl::ReferenceFrame current_frame = lrf[index];

        Eigen::Vector3f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2]);
        Eigen::Vector3f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2]);
        Eigen::Vector3f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2]);
        if(!isfinite(current_frame_x[0]) || !isfinite(current_frame_x[1]) || !isfinite(current_frame_x[2])
           || !isfinite(current_frame_y[0]) || !isfinite(current_frame_y[1]) || !isfinite(current_frame_y[2])
           || !isfinite(current_frame_z[0]) || !isfinite(current_frame_z[1]) || !isfinite(current_frame_z[2])  ){
            shot[offset] = NAN;
        }
        Eigen::Vector4i ijk(static_cast<int>(floor(central_point.x * inv_radius[0])),
                            static_cast<int>(floor(central_point.y * inv_radius[1])),
                            static_cast<int>(floor(central_point.z * inv_radius[2])), 0);
        ijk = ijk - grid_min;
        double bin_dist[1000];
        for (int i = ijk[2] - 1; i <= ijk[2] + 1; i++) {
            for (int j = ijk[1] - 1; j <= ijk[1] + 1; j++) {
                for (int k = ijk[0] - 1; k <= ijk[0] + 1; k++) {
                    int _x = imax(k, 0);
                    int _y = imax(j, 0);
                    int _z = imax(i, 0);
                    _x = imin(_x, grid_max[0]);
                    _y = imin(_y, grid_max[1]);
                    _z = imin(_z, grid_max[2]);
                    int grid_idx = _x + _y * grid_res[0] + _z * grid_res[0] * grid_res[1];
                    if (grid_start_indices[grid_idx] == -1) continue;
                    for (int s = grid_start_indices[grid_idx]; s < grid_end_indices[grid_idx]; s++){
                        int idx = array_indices[s];
                        PointType pt = surface[idx];
                        if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z)) {
                            Eigen::Vector3f delta(pt.x- central_point.x, pt.y - central_point.y, pt.z - central_point.z);
                            double distance = sqrt(delta.dot(delta));
                            if ( distance <= radius && !(pt.x == central_point.x && pt.y == central_point.y && pt.z == central_point.z)){

                                pcl::Normal norm = norms[idx];
                                const Eigen::Vector3f norm_vec(norm.normal_x, norm.normal_y, norm.normal_z);
                                if (! isfinite(norm_vec[0]) || !isfinite(norm_vec[1]) || !isfinite(norm_vec[2])){
                                    continue;
                                }else{
                                    double cosDesc = norm_vec.dot(current_frame_z);
                                    if (cosDesc > 1) cosDesc = 1;
                                    else if (cosDesc < -1) cosDesc = -1;
                                    bin_dist[num_neighbors] = ((1.0 + cosDesc) * n_dist_bin) / 2;
                                }


                                if (areEquals (distance, 0.0))
                                    continue;

                                double xInFeatRef = delta.dot (current_frame_x);
                                double yInFeatRef = delta.dot (current_frame_y);
                                double zInFeatRef = delta.dot (current_frame_z);

                                // To avoid numerical problems afterwards
                                if (fabs (yInFeatRef) < 1E-30)
                                    yInFeatRef  = 0;
                                if (fabs (xInFeatRef) < 1E-30)
                                    xInFeatRef  = 0;
                                if (fabs (zInFeatRef) < 1E-30)
                                    zInFeatRef = 0;

                                unsigned char bit4 = static_cast<unsigned char>(((yInFeatRef > 0) || ((yInFeatRef == 0.0) && (xInFeatRef < 0))) ? 1 : 0);
                                unsigned char bit3 = static_cast<unsigned char> (((xInFeatRef > 0) ||
                                                                                  ((xInFeatRef == 0.0) && (yInFeatRef > 0))) == (!bit4));

                                assert (bit3 == 0 || bit3 == 1);

                                int desc_index = (bit4<<3) + (bit3<<2);

                                desc_index = desc_index << 1;

                                if ((xInFeatRef * yInFeatRef > 0) || (xInFeatRef == 0.0))
                                    desc_index += (fabs (xInFeatRef) >= fabs (yInFeatRef)) ? 0 : 4;
                                else
                                    desc_index += (fabs (xInFeatRef) > fabs (yInFeatRef)) ? 4 : 0;

                                desc_index += zInFeatRef > 0 ? 1 : 0;

                                // 2 RADII
                                desc_index += (distance > radius1_2_) ? 2 : 0;

                                int step_index_shape = static_cast<int>(floor (bin_dist[num_neighbors] +0.5));

                                int volume_index_shape = desc_index * (n_dist_bin+1);


                                //Interpolation on the cosine (adjacent bins in the histrogram)
                                bin_dist[num_neighbors] -= step_index_shape;


                                double intWeightShape = (1- fabs (bin_dist[num_neighbors]));

                                if (bin_dist[num_neighbors] > 0)
                                    shot[offset + volume_index_shape + ((step_index_shape + 1) % n_dist_bin)] +=
                                            static_cast<float> (bin_dist[num_neighbors]);
                                else
                                    shot[offset + volume_index_shape + ((step_index_shape - 1 + n_dist_bin) % n_dist_bin)] -=
                                            static_cast<float> (bin_dist[num_neighbors]);

                                //Interpolation on the distance (adjacent husks)

                                if (distance > radius1_2_)   //external sphere
                                {
                                    double radiusDistance = (distance - radius3_4_) / radius1_2_;

                                    if (distance > radius3_4_) //most external sector, votes only for itself
                                    {
                                        intWeightShape += 1 - radiusDistance; //weight=1-d
                                    }
                                    else  //3/4 of radius, votes also for the internal sphere
                                    {
                                        intWeightShape += 1 + radiusDistance;
                                        shot[offset + (desc_index - 2) * (n_dist_bin+1) + step_index_shape] -=
                                                static_cast<float> (radiusDistance);
                                    }
                                }
                                else    //internal sphere
                                {
                                    double radiusDistance = (distance - radius1_4_) / radius1_2_;

                                    if (distance < radius1_4_) //most internal sector, votes only for itself
                                    {
                                        intWeightShape += 1 + radiusDistance;
                                    }
                                    else  //3/4 of radius, votes also for the external sphere
                                    {
                                        intWeightShape += 1 - radiusDistance; //weight=1-d
                                        shot[offset + (desc_index + 2) * (n_dist_bin+1) + step_index_shape] +=
                                                static_cast<float> (radiusDistance);
                                    }
                                }

                                //Interpolation on the inclination (adjacent vertical volumes)
                                double inclinationCos = zInFeatRef / distance;
                                if (inclinationCos < - 1.0)
                                    inclinationCos = - 1.0;
                                if (inclinationCos > 1.0)
                                    inclinationCos = 1.0;

                                double inclination = acos (inclinationCos);

                                assert (inclination >= 0.0 && inclination <= PST_RAD_180);

                                if (inclination > PST_RAD_90 || (fabs (inclination - PST_RAD_90) < 1e-30 && zInFeatRef <= 0))
                                {
                                    double inclinationDistance = (inclination - PST_RAD_135) / PST_RAD_90;
                                    if (inclination > PST_RAD_135)
                                    {
                                        intWeightShape += 1 - inclinationDistance;
                                    }
                                    else
                                    {
                                        intWeightShape += 1 + inclinationDistance;
//                            assert ((desc_index + 1) * (n_dist_bin+1) + step_index_shape >= 0 && (desc_index + 1) * (n_dist_bin+1) + step_index_shape < descLength_);
                                        shot[offset + (desc_index + 1) * (n_dist_bin+1) + step_index_shape] -=
                                                static_cast<float> (inclinationDistance);
                                    }
                                }
                                else
                                {
                                    double inclinationDistance = (inclination - PST_RAD_45) / PST_RAD_90;
                                    if (inclination < PST_RAD_45)
                                    {
                                        intWeightShape += 1 + inclinationDistance;
                                    }
                                    else
                                    {
                                        intWeightShape += 1 - inclinationDistance;
//                            assert ((desc_index - 1) * (n_dist_bin+1) + step_index_shape >= 0 && (desc_index - 1) * (n_dist_bin+1) + step_index_shape < descLength_);
                                        shot[offset + (desc_index - 1) * (n_dist_bin+1) + step_index_shape] +=
                                                static_cast<float> (inclinationDistance);
                                    }
                                }

                                if (yInFeatRef != 0.0 || xInFeatRef != 0.0)
                                {
                                    //Interpolation on the azimuth (adjacent horizontal volumes)
                                    double azimuth = atan2 (yInFeatRef, xInFeatRef);

                                    int sel = desc_index >> 2;
                                    double angularSectorSpan = PST_RAD_45;
                                    double angularSectorStart = - PST_RAD_PI_7_8;

                                    double azimuthDistance = (azimuth - (angularSectorStart + angularSectorSpan*sel)) / angularSectorSpan;
//                        assert ((azimuthDistance < 0.5 || areEquals (azimuthDistance, 0.5)) && (azimuthDistance > - 0.5 || areEquals (azimuthDistance, - 0.5)));
                                    azimuthDistance = max(- 0.5, min (azimuthDistance, 0.5));

                                    if (azimuthDistance > 0)
                                    {
                                        intWeightShape += 1 - azimuthDistance;
                                        int interp_index = (desc_index + 4) % maxAngularSectors_;
//                            assert (interp_index * (n_dist_bin+1) + step_index_shape >= 0 && interp_index * (n_dist_bin+1) + step_index_shape < descLength_);
                                        shot[offset + interp_index * (n_dist_bin+1) + step_index_shape] +=
                                                static_cast<float> (azimuthDistance);
                                    }
                                    else
                                    {
                                        int interp_index = (desc_index - 4 + maxAngularSectors_) % maxAngularSectors_;
                                        intWeightShape += 1 + azimuthDistance;
//                            assert (interp_index * (n_dist_bin+1) + step_index_shape >= 0 && interp_index * (n_dist_bin+1) + step_index_shape < descLength_);
                                        shot[offset + interp_index * (n_dist_bin+1) + step_index_shape] -=
                                                static_cast<float> (azimuthDistance);
                                    }
                                }

//                    assert (volume_index_shape + step_index_shape >= 0 &&  volume_index_shape + step_index_shape < descLength_);
                                shot[offset + volume_index_shape + step_index_shape] += static_cast<float> (intWeightShape);

                                num_neighbors++;
                                if (num_neighbors >= 1000) break;
                            }
                        }
                    }



                }
            }
        }


        assert(num_neighbors > 5);

        // normalize histogram
        if (isfinite(shot[offset])){
            double hist_sum = 0;
            for (int i = 0; i < descLength_; ++i)
                hist_sum += shot[offset + i] * shot[offset + i];
            float norm_fact = static_cast<float>(sqrt(hist_sum));

            for (int i = 0; i < descLength_; ++i)
                shot[offset + i] = shot[offset + i]/ norm_fact;
        }


    }

}



void SHOT352::computeDescriptor(pcl::PointCloud<pcl::SHOT352> &output, const Eigen::Vector4f &inv_radius,
        const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi, const Eigen::Vector4i &max_pi,
        const IndicesConstPtr &grid_start_indices, const IndicesConstPtr &grid_end_indices) {

    descLength_ = nr_grid_sector_ * (nr_shape_bins_ + 1);

    assert(descLength_ == 352);

    int _grid_count_max = pc_dimension[0] + pc_dimension[0] * pc_dimension[1] + pc_dimension[0] * pc_dimension[1] * pc_dimension[2];

    // compute local reference
//    pcl::PointCloud<pcl::ReferenceFrame> local_ref;

    // gpu implementation of lrf, local frame is not determinastic

//    SHOT_LRF lrf;
//    lrf.setRadius(_radius);
//    lrf.setInputCloud(_input);
//    lrf.setSurface(_surface);
//    lrf.setNormals(_normals);
//    lrf.setKeptIndices(_kept_indices);
//    lrf.compute(local_ref, inv_radius, pc_dimension, min_pi);

    std::cout << "-------------------calculating descriptor shot------------------------" << std::endl;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::vector<int> indices;
    for (int i =0; i < _input->points.size();i++){
        indices.emplace_back(i);
    }
    IndicesPtr indices_ = boost::make_shared<std::vector<int>>(indices);

    pcl::SHOTLocalReferenceFrameEstimationOMP<PointType> lrf_estimator;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr default_frames(new pcl::PointCloud<pcl::ReferenceFrame>);
    lrf_estimator.setRadiusSearch (_radius);
    lrf_estimator.setInputCloud (_input);
    lrf_estimator.setSearchSurface(_surface);
    lrf_estimator.setIndices (indices_);
    lrf_estimator.compute (*default_frames);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "local reference calculation takes: " << duration << std::endl;

    int N = static_cast<int> (_input->points.size());
    dim3 numThreadsPerBlock = (static_cast<u_int32_t >((N + blockSize - 1)/blockSize));

    int N_surface = static_cast<int>(_surface->points.size());

    int *dev_gridCellStartIndices = NULL;
    cudaMalloc((void**)&dev_gridCellStartIndices, _grid_count_max * sizeof(int));
    checkCUDAError("cudaMalloc dev_gridCellStartIndices failed");
    cudaMemcpy(dev_gridCellStartIndices, &(*grid_start_indices)[0], sizeof(int) * _grid_count_max, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMmcpy dev_gridCellStartIndices failed");

    int *dev_gridCellEndIndices = NULL;
    cudaMalloc((void**)&dev_gridCellEndIndices, _grid_count_max * sizeof(int));
    checkCUDAError("cudaMalloc dev_gridCellEndIndices failed");
    cudaMemcpy(dev_gridCellEndIndices, &(*grid_end_indices)[0], sizeof(int) * _grid_count_max, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy dev_gridCellEndIndices failed");

    int *dev_array_indices = NULL;
    cudaMalloc((void**)&dev_array_indices, N_surface * sizeof(int));
    checkCUDAError("dev_grid malloc");
    cudaMemcpy(dev_array_indices, &(*_array_indices)[0], N_surface * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("dev_grid memcpy");


    int *dev_kept_indices = NULL;
    cudaMalloc((void**)&dev_kept_indices, N * sizeof(int));
    checkCUDAError("cuda malloc kept indices error");
    cudaMemcpy(dev_kept_indices, &(*_kept_indices)[0], N * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cuda memcpy kept_indices error");

    pcl::ReferenceFrame *dev_lrf = NULL;
    cudaMalloc((void**)&dev_lrf, N * sizeof (pcl::ReferenceFrame));
    checkCUDAError("cuda malloc dev_lrf error");

    cudaMemcpy(dev_lrf, &default_frames->points[0], N * sizeof(pcl::ReferenceFrame), cudaMemcpyHostToDevice);
    checkCUDAError("cuda Memcpy lrf error");

    pcl::Normal *dev_normals = NULL;
    cudaMalloc((void**)&dev_normals, sizeof(pcl::Normal) * N_surface);
    checkCUDAError("cuda malloc dev_normals error");
    cudaMemcpy(dev_normals, &_normals->points[0], N_surface * sizeof(pcl::Normal), cudaMemcpyHostToDevice);
    checkCUDAError("cuda memcpy dev_normals error");

    PointType* dev_pos_surface;
    cudaMalloc((void**)&dev_pos_surface, N_surface * sizeof(PointType));
    checkCUDAError("malloc dps error");
    cudaMemcpy(dev_pos_surface, &(_surface->points[0]), N_surface * sizeof(PointType), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy ps error");

    float *dev_shot = NULL;
    cudaMalloc((void**)&dev_shot, N * descLength_ * sizeof(float));
    checkCUDAError("dev_bin_dist error");


    computeSHOT<<<numThreadsPerBlock, blockSize>>> (N, dev_pos_surface, _radius, dev_kept_indices, dev_array_indices,
             dev_gridCellStartIndices, dev_gridCellEndIndices, pc_dimension, min_pi, max_pi, inv_radius,
            dev_normals, dev_lrf, nr_shape_bins_, dev_shot);
    checkCUDAError("compute shot error");

//    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;

    std::vector<float> shot(static_cast<uint32_t >(N * descLength_));
    cudaMemcpy(&shot[0], dev_shot, sizeof(float) * N * descLength_, cudaMemcpyDeviceToHost);
    checkCUDAError("copy shot error");


    // write to output
    for (int i = 0; i < N; ++i){
        int offset = i * descLength_;
        if (!isfinite(shot[offset])){
            output.is_dense = false;
            for (int j = 0; j < descLength_; ++j)
                output.points[i].descriptor[j] = std::numeric_limits<float >::quiet_NaN();
            for (int j = 0; j < 9; ++j)
                output.points[i].rf[j] = std::numeric_limits<float >::quiet_NaN();
        }
        else{
            for (int j = 0; j < descLength_; ++j)
                output.points[i].descriptor[j] = shot[offset + j];
            for (int j = 0; j < 3; ++j){
                output.points[i].rf[j] = default_frames->points[i].x_axis[j];
                output.points[i].rf[j + 3] = default_frames->points[i].y_axis[j];
                output.points[i].rf[j + 6] = default_frames->points[i].z_axis[j];
            }
        }
    }

    std::cout << "descriptor calculation done" << std::endl;

    cudaFree(dev_kept_indices);
    cudaFree(dev_lrf);
    cudaFree(dev_normals);
    cudaFree(dev_pos_surface);
    cudaFree(dev_shot);
    checkCUDAError("cudafree err");

}
