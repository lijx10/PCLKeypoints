#ifndef KEYPOINTS_HPP
#define KEYPOINTS_HPP

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <random>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/harris_6d.h>
#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_omp.h>

#include <Eigen/Dense>


template <typename T>
typename pcl::PointCloud<T>::Ptr eigen2p_pcl(const Eigen::MatrixXf points){
    typename pcl::PointCloud<T>::Ptr p_pc(new typename pcl::PointCloud<T>());
    // Fill in the model cloud
    p_pc->width    = points.rows();
    p_pc->height   = 1;
    p_pc->is_dense = true;
    p_pc->points.resize (p_pc->width * p_pc->height);

    for (size_t i = 0; i < p_pc->points.size(); ++i){
        Eigen::Vector3f pt = points.row(i);
        p_pc->points[i].x = pt[0];
        p_pc->points[i].y = pt[1];
        p_pc->points[i].z = pt[2];
    }

    return p_pc;
}

template <typename T>
typename pcl::PointCloud<T>::Ptr eigen2p_pclxyzrgb(const Eigen::MatrixXf points){
    typename pcl::PointCloud<T>::Ptr p_pc(new typename pcl::PointCloud<T>());
    // Fill in the model cloud
    p_pc->width    = points.rows();
    p_pc->height   = 1;
    p_pc->is_dense = true;
    p_pc->points.resize (p_pc->width * p_pc->height);

    for (size_t i = 0; i < p_pc->points.size(); ++i){
        Eigen::VectorXf pt = points.row(i);
        p_pc->points[i].x = pt[0];
        p_pc->points[i].y = pt[1];
        p_pc->points[i].z = pt[2];
        p_pc->points[i].r = pt[3];
        p_pc->points[i].g = pt[4];
        p_pc->points[i].b = pt[5];
    }

    return p_pc;
}


template <typename T>
typename pcl::PointCloud<T>::Ptr eigen2p_pclNormal(const Eigen::MatrixXf points){
    typename pcl::PointCloud<T>::Ptr p_pc(new typename pcl::PointCloud<T>());
    // Fill in the model cloud
    p_pc->width    = points.rows();
    p_pc->height   = 1;
    p_pc->is_dense = true;
    p_pc->points.resize (p_pc->width * p_pc->height);

    for (size_t i = 0; i < p_pc->points.size(); ++i){
        Eigen::Vector3f pt = points.row(i);
        p_pc->points[i].normal_x = pt[0];
        p_pc->points[i].normal_y = pt[1];
        p_pc->points[i].normal_z = pt[2];
    }

    return p_pc;
}


template <typename T>
Eigen::MatrixXf p_pcl2eigen(const typename pcl::PointCloud<T>::Ptr p_pcl){
    Eigen::MatrixXf point_eigen(p_pcl->size(), 3);
    for(size_t i=0;i<p_pcl->size();++i){
        point_eigen(i, 0) = p_pcl->points[i].x;
        point_eigen(i, 1) = p_pcl->points[i].y;
        point_eigen(i, 2) = p_pcl->points[i].z;
    }
    return point_eigen;
}


template <typename T>
Eigen::MatrixXf p_pclNormal2eigen(const typename pcl::PointCloud<T>::Ptr p_pcl){
    Eigen::MatrixXf point_eigen(p_pcl->size(), 3);
    for(size_t i=0;i<p_pcl->size();++i){
        point_eigen(i, 0) = p_pcl->points[i].normal_x;
        point_eigen(i, 1) = p_pcl->points[i].normal_y;
        point_eigen(i, 2) = p_pcl->points[i].normal_z;
    }
    return point_eigen;
}

template <typename T>
Eigen::MatrixXf p_histogramFeature2eigen(const typename pcl::PointCloud<T>::Ptr p_feature){
    Eigen::MatrixXf feature_eigen(p_feature->size(), p_feature->points[0].descriptorSize());
    for(size_t i=0;i<feature_eigen.rows();++i){
        for(size_t j=0;j<feature_eigen.cols();++j){
            feature_eigen(i, j) = p_feature->points[i].histogram[j];
        }
    }

    return feature_eigen;
}


template <typename T>
Eigen::MatrixXf p_descriptorFeature2eigen(const typename pcl::PointCloud<T>::Ptr p_feature){
    Eigen::MatrixXf feature_eigen(p_feature->size(), p_feature->points[0].descriptorSize());
    for(size_t i=0;i<feature_eigen.rows();++i){
        for(size_t j=0;j<feature_eigen.cols();++j){
            feature_eigen(i, j) = p_feature->points[i].descriptor[j];
        }
    }

    return feature_eigen;
}



Eigen::MatrixXf keypointIss(const Eigen::MatrixXf points,
                            const float iss_salient_radius=3.0,
                            const float iss_non_max_radius=2.0,
                            const float iss_gamma_21=0.975,
                            const float iss_gamma_32=0.975,
                            const int iss_min_neighbors=5,
                            const int threads=0);

Eigen::MatrixXf keypointHarris3D(const Eigen::MatrixXf points,
                               const float radius=0.5,
                               const float nms_threshold=1e-3,
                               const int threads=0,
                               const bool is_nms=false,
                               const bool is_refine=false);

Eigen::MatrixXf keypointHarris6D(const Eigen::MatrixXf points,
                               const float radius=0.5,
                               const float nms_threshold=1e-3,
                               const int threads=0,
                               const bool is_nms=false,
                               const bool is_refine=false);

namespace pcl
{
template<>
struct SIFTKeypointFieldSelector<PointXYZ>
{
    inline float
    operator () (const PointXYZ &p) const
    {
        return p.y;
    }
};
}

Eigen::MatrixXf keypointSift(const Eigen::MatrixXf points,
                             const float min_scale=0.1,
                             const int n_octaves=6,
                             const int n_scales_per_octave=10,
                             const float min_contrast=0.05);


Eigen::MatrixXf featureFPFH33(const Eigen::MatrixXf pointcloud,
                              const Eigen::MatrixXf keypoints,
                              const int compute_normal_k,
                              const float feature_radius);

Eigen::MatrixXf featureFPFH33WithNormal(const Eigen::MatrixXf pointcloud,
                              const Eigen::MatrixXf normals,
                              const Eigen::MatrixXf keypoints,
                              const float feature_radius);

Eigen::MatrixXf featureSHOT352(const Eigen::MatrixXf pointcloud,
                               const Eigen::MatrixXf keypoints,
                               const int compute_normal_k,
                               const float feature_radius);

Eigen::MatrixXf featureSHOT352WithNormal(const Eigen::MatrixXf pointcloud,
                               const Eigen::MatrixXf normals,
                               const Eigen::MatrixXf keypoints,
                               const float feature_radius);


#endif // KEYPOINTS_HPP
