#pragma once

#include <scalable_ccd/common/timer.hpp>
#include <scalable_ccd/stq/cuda/memory.cuh>
#include <scalable_ccd/tight_inclusion/cuda/type.hpp>
#include <scalable_ccd/tight_inclusion/cuda/record.hpp>

#include <array>
#include <vector>

// #include <gputi/book.h>
namespace scalable_ccd {
// can be removed once device-only run_memory_pool_ccd copied over
__global__ void initialize_memory_pool(MP_unit* units, int query_size);
__global__ void compute_vf_tolerance_memory_pool(
    CCDData* data, CCDConfig* config, const int query_size);
__global__ void compute_ee_tolerance_memory_pool(
    CCDData* data, CCDConfig* config, const int query_size);
__global__ void shift_queue_pointers(CCDConfig* config);
// __global__ void vf_ccd_memory_pool(MP_unit *units, int query_size, CCDData
// *data, CCDConfig *config, int *results);
__global__ void vf_ccd_memory_pool(
    MP_unit* units, int query_size, CCDData* data, CCDConfig* config);
__global__ void ee_ccd_memory_pool(
    MP_unit* units, int query_size, CCDData* data, CCDConfig* config);
__global__ void compute_ee_tolerance_memory_pool(
    CCDData* data, CCDConfig* config, const int query_size);

void run_memory_pool_ccd(
    CCDData* d_data_list,
    stq::gpu::MemHandler* memhandle,
    int tmp_nbr,
    bool is_edge,
    std::vector<int>& result_list,
    int parallel_nbr,
    int max_iter,
    Scalar tol,
    bool use_ms,
    bool allow_zero_toi,
    Scalar& toi,
    int& overflow,
    Record& r);

// get the filter of ccd. the inputs are the vertices of the bounding box of
// the simulation scene this function is directly copied from
// https://github.com/Continuous-Collision-Detection/Tight-Inclusion/
std::array<Scalar, 3> get_numerical_error(
    const std::vector<std::array<Scalar, 3>>& vertices,
    const bool& check_vf,
    const bool using_minimum_separation);
} // namespace scalable_ccd