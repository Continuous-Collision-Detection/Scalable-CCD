#include "utils.cuh"

#include <scalable_ccd/cuda/broad_phase/queue.cuh>
#include <scalable_ccd/utils/logger.hpp>

namespace scalable_ccd::cuda {

void setup(
    int device_id, int& shared_memory_size, int& threads, int& boxes_per_thread)
{
    int max_shared_memory;
    cudaDeviceGetAttribute(
        &max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
    logger().trace(
        "Max shared memory per block: {:g} KiB",
        max_shared_memory / double(1 << 10));

    int max_threads;
    cudaDeviceGetAttribute(
        &max_threads, cudaDevAttrMaxThreadsPerBlock, device_id);
    logger().trace("Max threads per block: {:d} threads", max_threads);

    if (!boxes_per_thread) {
        boxes_per_thread =
            std::max(max_shared_memory / sizeof(AABB) / max_threads, size_t(1));
    }
    logger().trace("Boxes per thread: {:d}", boxes_per_thread);

    // divide threads by an arbitrary number as long as its reasonable >64
    if (!threads) {
        cudaDeviceGetAttribute(
            &threads, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);

        logger().trace("Max threads per multiprocessor: {:d} threads", threads);
    }
    // boxes_per_thread * threads * sizeof(AABB);
    shared_memory_size = QUEUE_SIZE * sizeof(int2);

    if (shared_memory_size > max_shared_memory) {
        logger().error(
            "Shared memory size exceeds max shared memory per block!");
        logger().error(
            "Max shared memory per block: {:d} B", max_shared_memory);
        logger().error("Shared memory size: {:d} B", shared_memory_size);
        throw std::runtime_error(
            "Shared memory size exceeds max shared memory per block");
    }
    // while (shared_memory_size > max_shared_memory || threads > max_threads) {
    //   // threads--;
    //   // shared_memory_size = boxes_per_thread * threads * sizeof(AABB);
    //   shared_memory_size /= 2;
    // }
    logger().trace("Actual threads per block: {:d} threads", threads);
    logger().trace("Shared memory allocated: {:d} B", shared_memory_size);
}

} // namespace scalable_ccd::cuda