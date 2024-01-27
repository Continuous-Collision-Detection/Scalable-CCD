#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <vector>

// for convenience
using json = nlohmann::json;

#include "io.hpp"

#include <scalable_ccd/stq/cpu/aabb.hpp>
#include <scalable_ccd/stq/cpu/sweep.hpp>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>

#include <spdlog/spdlog.h>

using namespace std;
using namespace scalable_ccd;

void compare_mathematica(vector<pair<int, int>> overlaps, const char* jsonPath)
{
    // Get from file
    ifstream in(jsonPath);
    if (in.fail()) {
        printf("%s does not exist", jsonPath);
        return;
    }
    json j_vec = json::parse(in);

    set<pair<int, int>> truePositives;
    vector<array<int, 2>> tmp = j_vec.get<vector<array<int, 2>>>();
    for (auto& arr : tmp)
        truePositives.emplace(arr[0], arr[1]);

    // Transform data to cantor
    set<pair<int, int>> algoBroadPhase;
    for (size_t i = 0; i < overlaps.size(); i++) {
        algoBroadPhase.emplace(overlaps[i].first, overlaps[i].second);
    }

    // Get intersection of true positive
    vector<pair<int, int>> algotruePositives(truePositives.size());
    vector<pair<int, int>>::iterator it = std::set_intersection(
        truePositives.begin(), truePositives.end(), algoBroadPhase.begin(),
        algoBroadPhase.end(), algotruePositives.begin());
    algotruePositives.resize(it - algotruePositives.begin());

    printf(
        "Contains %lu/%lu TP\n", algotruePositives.size(),
        truePositives.size());
    return;
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::trace);

    vector<char*> compare;

    const char* filet0 = argv[1];
    const char* filet1 = argv[2];

    vector<stq::cpu::Aabb> boxes;
    parse_mesh(filet0, filet1, boxes);

    int N = boxes.size();
    int n = N;
    vector<stq::cpu::Aabb> boxes_batching;
    // boxes_batching.resize(N);
    std::copy(boxes.begin(), boxes.end(), std::back_inserter(boxes_batching));
    int nbox = 0;

    int o;
    int parallel = 1;
    while ((o = getopt(argc, argv, "c:n:b:p:")) != -1) {
        switch (o) {
        case 'c':
            optind--;
            for (; optind < argc && *argv[optind] != '-'; optind++) {
                compare.push_back(argv[optind]);
                // compare_mathematica(overlaps, argv[optind]);
            }
            break;
        case 'n':
            N = atoi(optarg);
            break;
        case 'b':
            nbox = atoi(optarg);
            break;
        case 'p':
            parallel = stoi(optarg);
            break;
        }
    }

    auto start = std::chrono::system_clock::now();
    static const int CPU_THREADS =
        std::min(tbb::info::default_concurrency(), 64);
    tbb::global_control thread_limiter(
        tbb::global_control::max_allowed_parallelism, CPU_THREADS);
    spdlog::trace("Running with {:d} threads", CPU_THREADS);

    vector<pair<int, int>> overlaps;
    std::size_t count = 0;

    sweep_cpu_single_batch(boxes_batching, n, N, overlaps);
    while (overlaps.size()) {
        count += overlaps.size();
        sweep_cpu_single_batch(boxes_batching, n, N, overlaps);
    }

    auto stop = std::chrono::system_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    spdlog::trace("Elapsed time: {:.6f} ms", elapsed);
    printf("Overlaps: %zu\n", overlaps.size());
    printf("Final count: %zu\n", count);
    for (auto i : compare) {
        printf("%s\n", i);
        compare_mathematica(overlaps, i);
    }
    exit(0);
}