#pragma once

#include <scalable_ccd/stq/cpu/aabb.hpp>
#include <vector>

namespace scalable_ccd::stq::cpu {

bool is_face(const std::array<int, 3>& vids);

bool is_edge(const std::array<int, 3>& vids);

bool is_vertex(const std::array<int, 3>& vids);

bool is_valid_pair(const std::array<int, 3>& a, const std::array<int, 3>& b);

void run_sweep_cpu(
    std::vector<Aabb>& boxes,
    int& n,
    std::vector<std::pair<int, int>>& finOverlaps);

void sweep_cpu_single_batch(
    std::vector<Aabb>& boxes_batching,
    int& n,
    int N,
    std::vector<std::pair<int, int>>& overlaps);

void sweep(
    const std::vector<Aabb>& boxes,
    std::vector<std::pair<int, int>>& overlaps,
    int n);

void sort_along_xaxis(std::vector<Aabb>& boxes);

} // namespace scalable_ccd::stq::cpu