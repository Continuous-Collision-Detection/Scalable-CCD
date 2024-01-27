#pragma once

#include <spdlog/spdlog.h>
#include <cuda/semaphore>

namespace scalable_ccd::cuda::stq {

static const int HEAP_SIZE = 64;
// Use a sentinel value to indicate an error
__device__ int2 QUEUE_ERROR();

// Prototype of a utility function to swap two integers
// __device__ void swap(Cell&x, Cell &y);

__device__ __host__ class Queue {
public:
    int2 harr[HEAP_SIZE]; // pointer to array of elements in heap
    // int current = 0;
    // ::cuda::binary_semaphore<::cuda::thread_scope_block> lock[HEAP_SIZE];
    // int capacity;  // maximum possible size of min heap
    int heap_size; // Current number of elements in min heap
                   // Cell root;// temporary variable used for extractMin()
    // unsigned old_start;
    unsigned start;
    unsigned end;
    int nbr_per_loop;
    // int old_nbr_per_loop;
    // unsigned pop_cnt;
    // unsigned push_cnt;

    __device__ __host__ Queue();

    __device__ int2 pop();

    // Inserts a new key 'k'
    __device__ bool push(const int2 pair);

    __device__ int size();

    __device__ bool is_full();

    __device__ bool is_empty();
};

} // namespace scalable_ccd::cuda::stq