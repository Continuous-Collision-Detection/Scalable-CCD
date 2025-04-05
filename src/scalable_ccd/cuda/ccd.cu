#include "ccd.cuh"

#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/narrow_phase/narrow_phase.cuh>
#include <scalable_ccd/cuda/utils/assert.cuh>
#include <scalable_ccd/cuda/utils/device_matrix.cuh>
#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>

namespace scalable_ccd::cuda {

namespace {
    template <bool run_vf>
    void partial_ccd(
        const DeviceMatrix<Scalar>& d_vertices_t0,
        const DeviceMatrix<Scalar>& d_vertices_t1,
        const DeviceMatrix<int>& d_edges,
        const DeviceMatrix<int>& d_faces,
        const std::shared_ptr<DeviceAABBs> d_vertex_boxes,
        const std::shared_ptr<DeviceAABBs> d_edge_boxes,
        const std::shared_ptr<DeviceAABBs> d_face_boxes,
        const Scalar min_distance,
        const int max_iterations,
        const Scalar tolerance,
        const bool allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
        std::vector<std::tuple<int, int, Scalar>>& collisions,
#endif
        Scalar& toi,
        const int memory_limit_GB)
    {
        constexpr int bp_threads = 32;
        constexpr int np_threads = 1024;

        // --- Copy boxes to device --------------------------------------------
        logger().trace("Building broad phase");

        std::shared_ptr<MemoryHandler> memory_handler =
            std::make_shared<MemoryHandler>();
        if (memory_limit_GB) {
            logger().trace("Setting memory limit to {:d} GB", memory_limit_GB);
            memory_handler->memory_limit_GB = memory_limit_GB;
        }

        BroadPhase broad_phase(memory_handler);
        broad_phase.threads_per_block = bp_threads;
        if constexpr (run_vf) {
            broad_phase.build(d_vertex_boxes, d_face_boxes);
        } else {
            broad_phase.build(d_edge_boxes);
        }

        // --- Run broad + narrow phase ---------------------------------------

        while (!broad_phase.is_complete()) {
            logger().trace("Running broad phase");
            {
                SCALABLE_CCD_GPU_PROFILE_POINT("Broad Phase");
                broad_phase.detect_overlaps_partial();
                gpuErrchk(cudaDeviceSynchronize());
            }

            logger().debug("Running narrow phase");
            {
                SCALABLE_CCD_GPU_PROFILE_POINT("Narrow Phase");
                narrow_phase<run_vf>(
                    d_vertices_t0, d_vertices_t1, d_edges, d_faces,
                    broad_phase.overlaps(), np_threads, max_iterations,
                    tolerance, min_distance, allow_zero_toi, memory_handler,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
                    collisions,
#endif
                    toi);
                gpuErrchk(cudaDeviceSynchronize());
            }
        }
    }
} // namespace

Scalar
ccd(const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    const Scalar min_distance,
    const int max_iterations,
    const Scalar tolerance,
    const bool allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<std::tuple<int, int, Scalar>>& collisions,
#endif
    const int memory_limit_GB)
{
    assert(vertices_t0.rows() == vertices_t1.rows());
    assert(vertices_t0.cols() == vertices_t1.cols());
    assert(vertices_t0.cols() == 3);
    assert(edges.cols() == 2);
    assert(faces.cols() == 3);

    // --- Copy mesh to device --------------------------------------------
    logger().trace("Copying mesh to device");

    const DeviceMatrix<Scalar> d_vertices_t0(vertices_t0);
    const DeviceMatrix<Scalar> d_vertices_t1(vertices_t1);
    const DeviceMatrix<int> d_edges(edges);
    const DeviceMatrix<int> d_faces(faces);

    // --- Construct boxes ------------------------------------------------
    logger().trace("Constructing boxes");

    std::vector<AABB> vertex_boxes, edge_boxes, face_boxes;
    build_vertex_boxes(vertices_t0, vertices_t1, vertex_boxes, min_distance);
    build_edge_boxes(vertex_boxes, edges, edge_boxes);
    build_face_boxes(vertex_boxes, faces, face_boxes);

    const std::shared_ptr<DeviceAABBs> d_vertex_boxes =
        std::make_shared<DeviceAABBs>(vertex_boxes);
    const std::shared_ptr<DeviceAABBs> d_edge_boxes =
        std::make_shared<DeviceAABBs>(edge_boxes);
    const std::shared_ptr<DeviceAABBs> d_face_boxes =
        std::make_shared<DeviceAABBs>(face_boxes);

    // --- Run broad + narrow phase -------------------------------------------

    Scalar toi = 1;

    partial_ccd</*run_vf=*/true>(
        d_vertices_t0, d_vertices_t1, d_edges, d_faces, d_vertex_boxes,
        d_edge_boxes, d_face_boxes, min_distance, max_iterations, tolerance,
        allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
        collisions,
#endif
        toi, memory_limit_GB);

    partial_ccd</*run_vf=*/false>(
        d_vertices_t0, d_vertices_t1, d_edges, d_faces, d_vertex_boxes,
        d_edge_boxes, d_face_boxes, min_distance, max_iterations, tolerance,
        allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
        collisions,
#endif
        toi, memory_limit_GB);

    return toi;
}

} // namespace scalable_ccd::cuda