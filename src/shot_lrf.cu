//
// Created by sun on 11/26/18.
//

#include "shot_lrf.h"
#include "search.h"


void SHOT_LRF::computeDescriptor(pcl::PointCloud<pcl::ReferenceFrame> &output, const Eigen::Vector4f &inv_radius,
                                 const Eigen::Vector4i &pc_dimension, const Eigen::Vector4i &min_pi) {
    Search nn_search;
    nn_search.setRadius(_radius);
    nn_search.setSurface(_surface);
    nn_search.setFeatures(_input);
    nn_search.setFeaturesIndices(_kept_indices);
    nn_search.search(inv_radius, pc_dimension, min_pi);
    IndicesConstPtr numNeighbor = nn_search.getNumNeighbors();
    IndicesConstPtr neighborIndices = nn_search.getNeighborIndices();
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