#include "keypoints.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"


Eigen::MatrixXf keypointIss(const Eigen::MatrixXf points,
                            const float iss_salient_radius,
                            const float iss_non_max_radius,
                            const float iss_gamma_21,
                            const float iss_gamma_32,
                            const int iss_min_neighbors,
                            const int threads){
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_pc = eigen2p_pcl<pcl::PointXYZ>(points);

    //
    // Compute keypoints
    //
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_keypoints (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

    iss_detector.setSearchMethod (kdtree);
    iss_detector.setSalientRadius (iss_salient_radius);
    iss_detector.setNonMaxRadius (iss_non_max_radius);
    iss_detector.setThreshold21 (iss_gamma_21);
    iss_detector.setThreshold32 (iss_gamma_32);
    iss_detector.setMinNeighbors (iss_min_neighbors);
    iss_detector.setNumberOfThreads (threads);
    iss_detector.setInputCloud (p_pc);
    iss_detector.compute(*p_keypoints);

    //    std::cout<<"ISS point number: "<<p_keypoints->size()<<std::endl;
    return p_pcl2eigen<pcl::PointXYZ>(p_keypoints);
}


Eigen::MatrixXf keypointHarris3D(const Eigen::MatrixXf points,
                               const float radius,
                               const float nms_threshold,
                               const int threads,
                               const bool is_nms,
                               const bool is_refine){
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_pc = eigen2p_pcl<pcl::PointXYZ>(points);

    pcl::PointCloud<pcl::PointXYZI>::Ptr p_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris_detector;

    harris_detector.setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>::HARRIS);
    harris_detector.setNonMaxSupression (is_nms);
    harris_detector.setRadius(radius);
    harris_detector.setRefine(is_refine);
    harris_detector.setThreshold(nms_threshold);
    harris_detector.setNumberOfThreads(threads);
    harris_detector.setInputCloud(p_pc);
    harris_detector.compute(*p_keypoints);

    return p_pcl2eigen<pcl::PointXYZI>(p_keypoints);
}


Eigen::MatrixXf keypointHarris6D(const Eigen::MatrixXf points,
                               const float radius,
                               const float nms_threshold,
                               const int threads,
                               const bool is_nms,
                               const bool is_refine){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr p_pc = eigen2p_pclxyzrgb<pcl::PointXYZRGB>(points);

    pcl::PointCloud<pcl::PointXYZI>::Ptr p_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
    pcl::HarrisKeypoint6D<pcl::PointXYZRGB, pcl::PointXYZI> harris_detector;

    harris_detector.setNonMaxSupression (is_nms);
    harris_detector.setRadius(radius);
    harris_detector.setRefine(is_refine);
    harris_detector.setThreshold(nms_threshold);
    harris_detector.setNumberOfThreads(threads);
    harris_detector.setInputCloud(p_pc);
    harris_detector.compute(*p_keypoints);

    return p_pcl2eigen<pcl::PointXYZI>(p_keypoints);
}


Eigen::MatrixXf keypointSift(const Eigen::MatrixXf points,
                             const float min_scale,
                             const int n_octaves,
                             const int n_scales_per_octave,
                             const float min_contrast){
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_pc = eigen2p_pcl<pcl::PointXYZ>(points);

    pcl::PointCloud<pcl::PointXYZ>::Ptr p_keypoints (new pcl::PointCloud<pcl::PointXYZ> ());


    pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ> ());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(p_pc);
    sift.compute(result);

    pcl::copyPointCloud(result, *p_keypoints);

    // std::cout<<p_keypoints->size()<<std::endl;

    return p_pcl2eigen<pcl::PointXYZ>(p_keypoints);
}


Eigen::MatrixXf featureFPFH33(const Eigen::MatrixXf pointcloud,
                              const Eigen::MatrixXf keypoints,
                              const int compute_normal_k,
                              const float feature_radius){
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_pc = eigen2p_pcl<pcl::PointXYZ>(pointcloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_keypoints = eigen2p_pcl<pcl::PointXYZ>(keypoints);
    pcl::PointCloud<pcl::Normal>::Ptr p_normals(new pcl::PointCloud<pcl::Normal> ());

    // compute normal for keypoints
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
    norm_est.setKSearch(compute_normal_k);
    norm_est.setSearchSurface(p_pc);
    norm_est.setInputCloud(p_pc);
    norm_est.compute(*p_normals);

    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(p_keypoints);
    fpfh.setInputNormals(p_normals);
    fpfh.setSearchSurface(p_pc);
    // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

    // Create an empty kdtree representation, and pass it to the FPFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr p_tree (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod(p_tree);

    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_descriptors (new pcl::PointCloud<pcl::FPFHSignature33> ());

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch(feature_radius);

    // Compute the features
    fpfh.compute(*fpfh_descriptors);

    return p_histogramFeature2eigen<pcl::FPFHSignature33>(fpfh_descriptors);
}


Eigen::MatrixXf featureFPFH33WithNormal(const Eigen::MatrixXf pointcloud,
                              const Eigen::MatrixXf normals,
                              const Eigen::MatrixXf keypoints,
                              const float feature_radius){
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_pc = eigen2p_pcl<pcl::PointXYZ>(pointcloud);
    pcl::PointCloud<pcl::Normal>::Ptr p_normals = eigen2p_pclNormal<pcl::Normal>(normals);
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_keypoints = eigen2p_pcl<pcl::PointXYZ>(keypoints);

    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(p_keypoints);
    fpfh.setInputNormals(p_normals);
    fpfh.setSearchSurface(p_pc);
    // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

    // Create an empty kdtree representation, and pass it to the FPFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr p_tree (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod(p_tree);

    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_descriptors (new pcl::PointCloud<pcl::FPFHSignature33> ());

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch(feature_radius);

    // Compute the features
    fpfh.compute(*fpfh_descriptors);

    return p_histogramFeature2eigen<pcl::FPFHSignature33>(fpfh_descriptors);
}


Eigen::MatrixXf featureSHOT352(const Eigen::MatrixXf pointcloud,
                               const Eigen::MatrixXf keypoints,
                               const int compute_normal_k,
                               const float feature_radius){
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_pc = eigen2p_pcl<pcl::PointXYZ>(pointcloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_keypoints = eigen2p_pcl<pcl::PointXYZ>(keypoints);
    pcl::PointCloud<pcl::Normal>::Ptr p_normals(new pcl::PointCloud<pcl::Normal> ());

    // compute normal for keypoints
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
    norm_est.setKSearch(compute_normal_k);
    norm_est.setSearchSurface(p_pc);
    norm_est.setInputCloud(p_pc);
    norm_est.compute(*p_normals);

    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descr_est;
    descr_est.setRadiusSearch (feature_radius);

    descr_est.setInputCloud (p_keypoints);
    descr_est.setInputNormals (p_normals);
    descr_est.setSearchSurface (p_pc);

    pcl::PointCloud<pcl::SHOT352>::Ptr shot_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    descr_est.compute (*shot_descriptors);

    return p_descriptorFeature2eigen<pcl::SHOT352>(shot_descriptors);
}


Eigen::MatrixXf featureSHOT352WithNormal(const Eigen::MatrixXf pointcloud,
                               const Eigen::MatrixXf normals,
                               const Eigen::MatrixXf keypoints,
                               const float feature_radius){
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_pc = eigen2p_pcl<pcl::PointXYZ>(pointcloud);
    pcl::PointCloud<pcl::Normal>::Ptr p_normals = eigen2p_pclNormal<pcl::Normal>(normals);
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_keypoints = eigen2p_pcl<pcl::PointXYZ>(keypoints);

    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descr_est;
    descr_est.setRadiusSearch (feature_radius);

    descr_est.setInputCloud (p_keypoints);
    descr_est.setInputNormals (p_normals);
    descr_est.setSearchSurface (p_pc);

    pcl::PointCloud<pcl::SHOT352>::Ptr shot_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    descr_est.compute (*shot_descriptors);

    return p_descriptorFeature2eigen<pcl::SHOT352>(shot_descriptors);
}


namespace py = pybind11;

PYBIND11_MODULE(PCLKeypoint, m) {
    m.doc() = "PCL Keypoint";

    m.def("keypointIss", &keypointIss,
          py::arg("points"),
          py::arg("iss_salient_radius")=3.0f,
          py::arg("iss_non_max_radius")=2.0f,
          py::arg("iss_gamma_21")=0.975f,
          py::arg("iss_gamma_32")=0.975f,
          py::arg("iss_min_neighbors")=5,
          py::arg("threads")=0);
    m.def("keypointHarris3D", &keypointHarris3D,
          py::arg("points"),
          py::arg("radius")=0.5f,
          py::arg("nms_threshold")=0.001f,
          py::arg("threads")=0,
          py::arg("is_nms")=false,
          py::arg("is_refine")=false);
    m.def("keypointHarris6D", &keypointHarris6D,
          py::arg("points"),
          py::arg("radius")=0.5f,
          py::arg("nms_threshold")=0.001f,
          py::arg("threads")=0,
          py::arg("is_nms")=false,
          py::arg("is_refine")=false);
    m.def("keypointSift", &keypointSift,
          py::arg("points"),
          py::arg("min_scale")=0.1f,
          py::arg("n_octaves")=6,
          py::arg("n_scales_per_octave")=10,
          py::arg("min_contrast")=0.05f);

    m.def("featureFPFH33", &featureFPFH33,
          py::arg("pointcloud"),
          py::arg("keypoints"),
          py::arg("compute_normal_k")=10,
          py::arg("feature_radius")=1.0f);
    m.def("featureFPFH33WithNormal", &featureFPFH33WithNormal,
          py::arg("pointcloud"),
          py::arg("normals"),
          py::arg("keypoints"),
          py::arg("feature_radius")=1.0f);
    m.def("featureSHOT352", &featureSHOT352,
          py::arg("pointcloud"),
          py::arg("keypoints"),
          py::arg("compute_normal_k")=10,
          py::arg("feature_radius")=1.0f);
    m.def("featureSHOT352WithNormal", &featureSHOT352WithNormal,
          py::arg("pointcloud"),
          py::arg("normals"),
          py::arg("keypoints"),
          py::arg("feature_radius")=1.0f);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
