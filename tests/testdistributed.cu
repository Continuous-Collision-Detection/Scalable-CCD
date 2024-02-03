#include <stq/gpu/broad_phase.cuh>
#include <stq/gpu/groundtruth.h>
#include <stq/gpu/utils.cuh>

#include <vector>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <functional>
#include <cmath>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>

using namespace std;

__global__ void
square_sum(int* d_in, int* d_out, int* d_count, int N, int start, int end)
{
    int tid = start + threadIdx.x + blockIdx.x * blockDim.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= end || tid >= N)
        return;

    if (d_in[tid] % 5 == 0) {
        // d_out[gid] = d_in[tid] * d_in[tid];
        int i = atomicAdd(d_count, 1);
        d_out[i] = d_in[tid] * d_in[tid];
    } else {
        // int i = atomicAdd(d_count, 1);
        // d_out[i] = 5;
    }
}

void merge_local(
    const tbb::enumerable_thread_specific<vector<int>>& storages,
    std::vector<int>& overlaps)
{
    overlaps.clear();
    size_t num_overlaps = overlaps.size();
    for (const auto& local_overlaps : storages) {
        num_overlaps += local_overlaps.size();
    }
    // serial merge!
    overlaps.reserve(num_overlaps);
    for (const auto& local_overlaps : storages) {
        overlaps.insert(
            overlaps.end(), local_overlaps.begin(), local_overlaps.end());
    }
}

void run_sweep_multigpu(int N, int devcount)
{
    vector<int> squareSums;

    int in[N];
    for (int i = 0; i < N; i++)
        in[i] = N - i;

    cout << "default threads " << tbb::info::default_concurrency() << endl;
    // tbb::global_control
    // thread_limiter(tbb::global_control::max_allowed_parallelism, 2);
    tbb::enumerable_thread_specific<vector<int>> storages;

    int device_init_id = 0;

    // int smemSize;
    // setup(device_init_id, smemSize, threads, nbox);

    cudaSetDevice(device_init_id);

    int* d_in;

    cudaMalloc((void**)&d_in, sizeof(int) * N);

    cudaMemcpy(d_in, in, sizeof(int) * N, cudaMemcpyHostToDevice);

    int threads = 1024;
    dim3 block(threads);
    int grid_dim_1d = (N / threads + 1);
    dim3 grid(grid_dim_1d);

    try {
        thrust::sort(thrust::device, d_in, d_in + N);
    } catch (thrust::system_error& e) {
        printf("Error: %s \n", e.what());
    }
    cudaDeviceSynchronize();

    int devices_count;
    cudaGetDeviceCount(&devices_count);
    // devices_count-=2;
    devices_count = devcount ? devcount : devices_count;
    int range = ceil((float)N / devices_count);
    printf("range: %i\n", range);

    tbb::parallel_for(0, devices_count, 1, [&](int& device_id) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        printf(
            "%s -> unifiedAddressing = %d\n", prop.name,
            prop.unifiedAddressing);

        cudaSetDevice(device_id);

        int is_able;

        for (int i = 0; i < devices_count; i++) {
            cudaDeviceCanAccessPeer(&is_able, device_id, i);
            if (is_able) {
                cudaDeviceEnablePeerAccess(i, 0);
            } else if (i != device_id)
                printf("Device %i cant access Device %i\n", device_id, i);
        }

        gpuErrchk(cudaGetLastError());

        int range_start = range * device_id;
        int range_end = range * (device_id + 1);
        printf("device_id: %i [%i, %i)\n", device_id, range_start, range_end);

        int* d_in_solo;
        cudaMalloc((void**)&d_in_solo, sizeof(int) * N);
        // if (device_id == device_init_id )
        cudaMemcpy(d_in_solo, d_in, sizeof(int) * N, cudaMemcpyDefault);

        // // turn off peer access for write variables
        sleep(1);
        for (int i = 0; i < devices_count; i++) {
            cudaDeviceCanAccessPeer(&is_able, device_id, i);
            if (is_able) {
                cudaDeviceDisablePeerAccess(i);
            } else if (i != device_id)
                printf("Device %i cant access Device %i\n", device_id, i);
        }
        sleep(1);

        int* d_out;
        cudaMalloc((void**)&d_out, sizeof(int) * range);
        cudaMemset(d_out, 0, sizeof(int) * range);

        int* d_count;
        cudaMalloc((void**)&d_count, sizeof(int) * 1);
        cudaMemset(d_count, 0, sizeof(int) * 1);

        square_sum<<<grid, block>>>(
            d_in_solo, d_out, d_count, N, range_start, range_end);
        gpuErrchk(cudaDeviceSynchronize());

        int count;
        gpuErrchk(
            cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        printf("count for device %i : %i\n", device_id, count);
        cudaFree(d_out);
        cudaMalloc((void**)&d_out, sizeof(int) * count);
        cudaMemset(d_out, -1, sizeof(int) * count);

        cudaMemset(d_count, 0, sizeof(int) * 1);

        square_sum<<<grid, block>>>(
            d_in_solo, d_out, d_count, N, range_start, range_end);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(
            cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        printf("count2 for device %i : %i\n", device_id, count);

        int* out = (int*)malloc(sizeof(int) * count);
        gpuErrchk(cudaMemcpy(
            out, d_out, sizeof(int) * count, cudaMemcpyDeviceToHost));

        auto& local_overlaps = storages.local();

        for (size_t i = 0; i < count; i++) {
            local_overlaps.emplace_back(out[i]);
        }

        printf(
            "Total(filt.) overlaps for devid %i: %i\n", device_id,
            local_overlaps.size());
        // delete [] overlaps;
        // free(overlaps);

        // // free(counter);
        // // free(counter);
        // cudaFree(d_overlaps);
        // cudaFree(d_count);
        // // cudaFree(d_b);
        // // cudaFree(d_r);
        // cudaDeviceReset();
    }); // end tbb for loop

    merge_local(storages, squareSums);

    int sum = accumulate(squareSums.begin(), squareSums.end(), 0);
    printf("\nFinal result: %i\n", sum);
    printf("Final result size: %i\n", squareSums.size());
    printf("\n");
    for (int i = 0; i < squareSums.size(); i++) {
        printf("%i ", squareSums[i]);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    int N = 1;
    int devcount = 0;

    int o;
    while ((o = getopt(argc, argv, "n:d:")) != -1) {
        switch (o) {
        case 'n':
            N = atoi(optarg);
            break;
        case 'd':
            devcount = atoi(optarg);
            break;
        }
    }

    run_sweep_multigpu(N, devcount);
}
