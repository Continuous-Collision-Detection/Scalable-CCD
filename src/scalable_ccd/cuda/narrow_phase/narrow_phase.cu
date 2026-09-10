#include "narrow_phase.cuh"

#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/narrow_phase/root_finder.cuh>
#include <scalable_ccd/cuda/utils/device_buffer.cuh>
#include <scalable_ccd/utils/profiler.hpp>

#include <thrust/host_vector.h>
#include <thrust/copy.h>

namespace scalable_ccd::cuda {

namespace {
    /// @brief Populate the CCDData array with the necessary data for the narrow phase.
    /// @param vertices_t0 Vertex positions at time t=0
    /// @param vertices_t1 Vertex positions at time t=1
    /// @param edge Edge matrix
    /// @param faces Face matrix
    /// @param overlaps The array of pairs of indices of the boxes that overlap
    /// @param n_overlaps The number of overlaps
    /// @param ms Minimum separation distance
    /// @param data The output array of CCDData
    template <bool is_vf>
    __global__ void add_data(
        const RawDeviceMatrix<Scalar> vertices_t0,
        const RawDeviceMatrix<Scalar> vertices_t1,
        const RawDeviceMatrix<int> edges,
        const RawDeviceMatrix<int> faces,
        const int2* const overlaps,
        const int n_overlaps,
        const Scalar ms,
        CCDData* data)
    {
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= n_overlaps)
            return;

        data[tid].ms = ms;

        if constexpr (is_vf) {
            const auto& [vi, fi] = overlaps[tid];

            for (int i = 0; i < 3; i++) {
                data[tid].v0s[i] = vertices_t0(vi, i);
                data[tid].v1s[i] = vertices_t0(faces(fi, 0), i);
                data[tid].v2s[i] = vertices_t0(faces(fi, 1), i);
                data[tid].v3s[i] = vertices_t0(faces(fi, 2), i);
                data[tid].v0e[i] = vertices_t1(vi, i);
                data[tid].v1e[i] = vertices_t1(faces(fi, 0), i);
                data[tid].v2e[i] = vertices_t1(faces(fi, 1), i);
                data[tid].v3e[i] = vertices_t1(faces(fi, 2), i);
            }
        } else {
            const auto& [eai, ebi] = overlaps[tid];

            for (int i = 0; i < 3; i++) {
                data[tid].v0s[i] = vertices_t0(edges(eai, 0), i);
                data[tid].v1s[i] = vertices_t0(edges(eai, 1), i);
                data[tid].v2s[i] = vertices_t0(edges(ebi, 0), i);
                data[tid].v3s[i] = vertices_t0(edges(ebi, 1), i);
                data[tid].v0e[i] = vertices_t1(edges(eai, 0), i);
                data[tid].v1e[i] = vertices_t1(edges(eai, 1), i);
                data[tid].v2e[i] = vertices_t1(edges(ebi, 0), i);
                data[tid].v3e[i] = vertices_t1(edges(ebi, 1), i);
            }
        }

#ifdef SCALABLE_CCD_TOI_PER_QUERY
        data[tid].toi = INFINITY;
        data[tid].aid = overlaps[tid].x;
        data[tid].bid = overlaps[tid].y;
#endif
    }

#ifdef SCALABLE_CCD_TOI_PER_QUERY
    struct is_collision {
        __host__ __device__ bool operator()(const CCDData& data)
        {
            return data.toi < 1;
        }
    };

    void copy_out_collisions(
        const thrust::device_vector<CCDData>& d_ccd_data,
        std::vector<std::tuple<int, int, Scalar>>& collisions)
    {
        // Filter only the collisions on the device
        thrust::device_vector<CCDData> d_filtered_ccd_data(d_ccd_data.size());
        auto itr = thrust::copy_if(
            d_ccd_data.begin(), d_ccd_data.end(), d_filtered_ccd_data.begin(),
            is_collision());
        d_filtered_ccd_data.resize(
            thrust::distance(d_filtered_ccd_data.begin(), itr));

        // Copy the filtered collisions to the host
        thrust::host_vector<CCDData> filtered_ccd_data = d_filtered_ccd_data;

        // Transform the filtered collisions to the output format
        for (const auto& data : filtered_ccd_data) {
            collisions.emplace_back(data.aid, data.bid, data.toi);
        }
    }
#endif

} // namespace

template <bool is_vf>
void narrow_phase(
    const DeviceMatrix<Scalar>& d_vertices_t0,
    const DeviceMatrix<Scalar>& d_vertices_t1,
    const DeviceMatrix<int>& d_edges,
    const DeviceMatrix<int>& d_faces,
    const thrust::device_vector<int2>& d_overlaps,
    const int threads,
    const int max_iter,
    const Scalar tol,
    const Scalar ms,
    const bool allow_zero_toi,
    std::shared_ptr<MemoryHandler> memory_handler,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<std::tuple<int, int, Scalar>>& collisions,
#endif
    Scalar& toi)
{
    assert(toi >= 0);

    const bool use_ms = ms > 0;

    size_t start_id = 0;
    size_t size = d_overlaps.size();
    memory_handler->MAX_QUERIES = size;

    size_t remaining_queries;
#ifndef SCALABLE_CCD_TOI_PER_QUERY
    while ((remaining_queries = size - start_id) > 0 && toi > 0) {
#else
    while ((remaining_queries = size - start_id) > 0) {
#endif
        logger().trace("Remaining queries to process: {:d}", remaining_queries);

        bool overflowed = false;
        size_t n_queries_to_process;

        thrust::device_vector<CCDData> d_ccd_data;
        do {
            n_queries_to_process =
                std::min(remaining_queries, memory_handler->MAX_QUERIES);

            if (!overflowed) { // only true in first iteration
                memory_handler->handleNarrowPhase(n_queries_to_process);
            } else {
                memory_handler->handleOverflow(n_queries_to_process);
            }
            assert(n_queries_to_process > 0);
            assert(n_queries_to_process <= d_overlaps.size());

            {
                SCALABLE_CCD_GPU_PROFILE_POINT("create_ccd_data");

                d_ccd_data.resize(n_queries_to_process);
                add_data<is_vf><<<d_ccd_data.size() / threads + 1, threads>>>(
                    d_vertices_t0, d_vertices_t1, d_edges, d_faces,
                    thrust::raw_pointer_cast(d_overlaps.data()) + start_id,
                    n_queries_to_process, ms,
                    thrust::raw_pointer_cast(d_ccd_data.data()));
                gpuErrchk(cudaDeviceSynchronize());
            }

            logger().trace(
                "Narrow phase CCD data size: {:g} GB",
                d_ccd_data.size() * sizeof(CCDData) / 1e9);

            constexpr int parallel = 64;
            logger().trace(
                "Running memory-pooled CCD using {:d} threads", parallel);
            {
                SCALABLE_CCD_GPU_PROFILE_POINT(is_vf ? "FV CCD" : "EE CCD");

                overflowed = ccd<is_vf>(
                    d_ccd_data, memory_handler, parallel, max_iter, tol, use_ms,
                    allow_zero_toi, toi);

                gpuErrchk(cudaDeviceSynchronize());
            }

            if (overflowed) // rerun
            {
                logger().debug(
                    "Narrow-phase: overflowed; reducing parallel count");
                continue;
            }

            logger().debug("ToI after {}: {:e}", is_vf ? "FV" : "EE", toi);
        } while (overflowed);

#ifdef SCALABLE_CCD_TOI_PER_QUERY
        {
            SCALABLE_CCD_GPU_PROFILE_POINT("copy_out_collisions");
            copy_out_collisions(d_ccd_data, collisions);
        }
#endif

        start_id += n_queries_to_process;
    }
}

// === Template instantiation =================================================

template void narrow_phase<false>(
    const DeviceMatrix<Scalar>&,
    const DeviceMatrix<Scalar>&,
    const DeviceMatrix<int>& d_edges,
    const DeviceMatrix<int>& d_faces,
    const thrust::device_vector<int2>&,
    const int,
    const int,
    const Scalar,
    const Scalar,
    const bool,
    std::shared_ptr<MemoryHandler>,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<std::tuple<int, int, Scalar>>&,
#endif
    Scalar&);
template void narrow_phase<true>(
    const DeviceMatrix<Scalar>&,
    const DeviceMatrix<Scalar>&,
    const DeviceMatrix<int>& d_edges,
    const DeviceMatrix<int>& d_faces,
    const thrust::device_vector<int2>&,
    const int,
    const int,
    const Scalar,
    const Scalar,
    const bool,
    std::shared_ptr<MemoryHandler>,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<std::tuple<int, int, Scalar>>&,
#endif
    Scalar&);

} // namespace scalable_ccd::cuda