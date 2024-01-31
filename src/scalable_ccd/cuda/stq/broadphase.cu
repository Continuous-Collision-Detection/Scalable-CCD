#include "broadphase.cuh"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/utils/profiler.hpp>
#include <scalable_ccd/cuda/stq/sweep.cuh>
#include <scalable_ccd/cuda/stq/util.cuh>

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <tbb/parallel_for.h>

namespace scalable_ccd::cuda::stq {

void BroadPhase::clear()
{
    *memory_handler = MemoryHandler();

    d_boxes.clear();
    d_boxes.shrink_to_fit();

    d_sm.clear();
    d_sm.shrink_to_fit();

    d_mini.clear();
    d_mini.shrink_to_fit();

    d_overlaps.clear();
    d_overlaps.shrink_to_fit();

    num_boxes_per_thread = 0;
    threads_per_block = 32;
    start_thread_id = 0;
    num_devices = 1;
}

const thrust::device_vector<cuda::stq::Aabb>&
BroadPhase::build(const std::vector<cuda::stq::Aabb>& boxes)
{
    logger().trace("Number of boxes: {:d}", boxes.size());

    if (memory_handler->MAX_OVERLAP_CUTOFF == 0) {
        memory_handler->MAX_OVERLAP_CUTOFF = boxes.size();
        logger().trace(
            "Setting MAX_OVERLAP_CUTOFF to {:d}",
            memory_handler->MAX_OVERLAP_CUTOFF);
    }

    if (memory_limit_GB) {
        logger().trace("Setting memory limit to {:d} GB", memory_limit_GB);
        memory_handler->limitGB = memory_limit_GB;
    }

    setup(device_init_id, smemSize, threads_per_block, num_boxes_per_thread);
    cudaSetDevice(device_init_id);

    d_boxes = boxes; // copy to device
    d_sm.resize(boxes.size());
    d_mini.resize(boxes.size());

    // const Dimension axis = calc_sort_dimension();
    const Dimension axis = x;

    // Initialize d_sm and d_mini
    {
        SCALABLE_CCD_GPU_PROFILE_POINT("splitBoxes");
        splitBoxes<<<grid_dim_1d(), threads_per_block>>>(
            thrust::raw_pointer_cast(d_boxes.data()),
            thrust::raw_pointer_cast(d_sm.data()),
            thrust::raw_pointer_cast(d_mini.data()), d_boxes.size(), axis);
        gpuErrchk(cudaDeviceSynchronize());
    }

    {
        SCALABLE_CCD_GPU_PROFILE_POINT("sortingBoxes");
        thrust::sort_by_key(
            thrust::device, d_sm.begin(), d_sm.end(), d_mini.begin(),
            sort_aabb_x());
        thrust::sort(
            thrust::device, d_boxes.begin(), d_boxes.end(), sort_aabb_x());
    }

    gpuErrchk(cudaGetLastError());

    return d_boxes;
}

const thrust::device_vector<int2>& BroadPhase::detect_overlaps_partial()
{
    memory_handler->setOverlapSize();
    logger().trace(
        "Max overlap size: {:d} ({:g} GB)", memory_handler->MAX_OVERLAP_SIZE,
        memory_handler->MAX_OVERLAP_SIZE * sizeof(int2) / 1e9);
    logger().trace(
        "Max overlap cutoff: {:d}", memory_handler->MAX_OVERLAP_CUTOFF);

    // Device memory_handler to keep track of vars
    thrust::device_vector<MemoryHandler> d_memory_handler = {
        { *memory_handler }
    };

    int real_count;
    do {
        // Allocate a large chunk of memory for overlaps
        // d_overlaps.resize(memory_handler->MAX_OVERLAP_CUTOFF);
        d_overlaps.resize(memory_handler->MAX_OVERLAP_SIZE);

        {
            SCALABLE_CCD_GPU_PROFILE_POINT("runSTQ");
            // This will be the actual number of overlaps
            thrust::device_vector<int> d_num_overlaps = { { 0 } };
            thrust::device_vector<int> d_start = { { start_thread_id } };
            // runSTQ<<<grid_dim_1d(), threads_per_block>>>(
            //     thrust::raw_pointer_cast(d_sm.data()),
            //     thrust::raw_pointer_cast(d_mini.data()),
            //     /*num_boxes=*/d_boxes.size(),
            //     thrust::raw_pointer_cast(d_overlaps.data()),
            //     thrust::raw_pointer_cast(d_num_overlaps.data()),
            //     thrust::raw_pointer_cast(d_start.data()),
            //     thrust::raw_pointer_cast(d_memory_handler.data()));

            runSAP<<<grid_dim_1d(), threads_per_block>>>(
                thrust::raw_pointer_cast(d_sm.data()),
                thrust::raw_pointer_cast(d_mini.data()),
                /*num_boxes=*/d_boxes.size(),
                thrust::raw_pointer_cast(d_overlaps.data()),
                thrust::raw_pointer_cast(d_num_overlaps.data()),
                thrust::raw_pointer_cast(d_start.data()),
                thrust::raw_pointer_cast(d_memory_handler.data()));

            gpuErrchk(cudaDeviceSynchronize());

            // Resize overlaps to actual size (keeps the capacity the same)
            d_overlaps.resize(d_num_overlaps[0]);
        }

        gpuErrchk(cudaMemcpy(
            &real_count, &(d_memory_handler.data()->realcount), sizeof(int),
            cudaMemcpyDeviceToHost));

        if (real_count > memory_handler->MAX_OVERLAP_SIZE) {
            logger().debug(
                "Real count {:d} exceeds MAX_OVERLAP_SIZE {:d}; ending partial pass.",
                real_count, memory_handler->MAX_OVERLAP_SIZE);
        } else if (d_overlaps.size() < real_count) {
            logger().debug(
                "Found {:d} overlaps, but {:d} exist; re-running with increased capacity.",
                d_overlaps.size(), real_count);

            // Increase MAX_OVERLAP_SIZE (or decrease MAX_OVERLAP_CUTOFF)
            memory_handler->handleBroadPhaseOverflow(real_count);

            // Update memory handler on device
            d_memory_handler[0] = *memory_handler;

        } else {
            assert(real_count == d_overlaps.size());
            logger().trace("Found {:d} overlaps.", d_overlaps.size());
        }

    } while (d_overlaps.size() < real_count
             && real_count < memory_handler->MAX_OVERLAP_SIZE);

    // Increase start_thread_id for next run
    start_thread_id += memory_handler->MAX_OVERLAP_CUTOFF;

    // Free up excess memory
    d_overlaps.shrink_to_fit();

    logger().debug(
        "Final count for device {:d}: {:d} ({:g} GB)", 0, d_overlaps.size(),
        d_overlaps.size() * sizeof(int2) / 1e9);
    logger().trace("Next threadstart {:d}", start_thread_id);

    return d_overlaps;
}

std::vector<std::pair<int, int>> BroadPhase::detect_overlaps()
{
    thrust::host_vector<int2> h_overlaps;

    while (!is_complete()) {
        detect_overlaps_partial();

        h_overlaps.reserve(h_overlaps.size() + d_overlaps.size());

        h_overlaps.insert(
            h_overlaps.end(), d_overlaps.begin(), d_overlaps.end());
    }

    logger().debug("Complete overlaps size {:d}", h_overlaps.size());

    std::vector<std::pair<int, int>> overlaps;
    overlaps.reserve(h_overlaps.size());
    for (const int2& overlap : h_overlaps) {
        overlaps.emplace_back(overlap.x, overlap.y);
    }
    return overlaps;
}

// ----------------------------------------------------------------------------

Dimension BroadPhase::calc_sort_dimension() const
{
    // mean of all box points (used to find best axis)
    thrust::device_vector<Scalar3> d_mean(1, make_Scalar3(0, 0, 0));
    calc_mean<<<grid_dim_1d(), threads_per_block, smemSize>>>(
        thrust::raw_pointer_cast(d_boxes.data()), d_boxes.size(),
        thrust::raw_pointer_cast(d_mean.data()));

    // temporary
    const Scalar3 mean = d_mean[0];
    logger().trace("mean: x {:.6f} y {:.6f} z {:.6f}", mean.x, mean.y, mean.z);

    // calculate variance and determine which axis to sort on
    thrust::device_vector<Scalar3> d_variance(1, make_Scalar3(0, 0, 0));

    calc_variance<<<grid_dim_1d(), threads_per_block, smemSize>>>(
        thrust::raw_pointer_cast(d_boxes.data()), d_boxes.size(),
        thrust::raw_pointer_cast(d_mean.data()),
        thrust::raw_pointer_cast(d_variance.data()));
    cudaDeviceSynchronize();

    const Scalar3 variance = d_variance[0];
    logger().trace(
        "var: x {:.6f} y {:.6f} z {:.6f}", variance.x, variance.y, variance.z);
    const Scalar max_variance =
        std::max({ variance.x, variance.y, variance.z });

    Dimension axis;
    if (max_variance == variance.x) {
        axis = x;
    } else if (max_variance == variance.y) {
        axis = y;
    } else {
        axis = z;
    }
    logger().trace("Axis: {:s}", axis == x ? "x" : (axis == y ? "y" : "z"));
    return axis;
}

} // namespace scalable_ccd::cuda::stq