#pragma once

#include <Eigen/Core>
#include <scalable_ccd/scalar.hpp>

namespace scalable_ccd::cuda {

/// @brief Run CCD using the IPC [Li et al. 2020] strategy.
/// @param V0 Vertices at t=0
/// @param V1 Vertices at t=1
/// @param E Edges
/// @param F Faces
/// @param min_distance Minimum distance between objects
/// @param max_iter Maximum number of iterations
/// @param tolerance Tolerance for the CCD algorithm
/// @return Time of impact
Scalar ipc_ccd_strategy(
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const Scalar min_distance,
    const int max_iter,
    const Scalar tolerance);

} // namespace scalable_ccd::cuda