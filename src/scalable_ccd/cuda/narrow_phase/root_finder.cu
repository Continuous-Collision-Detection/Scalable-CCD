#include "root_finder.cuh"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/narrow_phase/ccd_buffer.cuh>
#include <scalable_ccd/cuda/narrow_phase/ccd_config.cuh>
#include <scalable_ccd/cuda/narrow_phase/ccd_data.cuh>
#include <scalable_ccd/cuda/narrow_phase/interval.cuh>
#include <scalable_ccd/cuda/utils/assert.cuh>
#include <scalable_ccd/cuda/utils/atomic_min_float.cuh>
#include <scalable_ccd/cuda/utils/device_variable.cuh>
#include <scalable_ccd/utils/logger.hpp>

#include <vector>

namespace scalable_ccd::cuda {

namespace {
    // Allocate this in constant memory for faster access
    __device__ __constant__ CCDConfig CONFIG;

    __device__ inline bool
    sum_less_than_one(const Scalar& num1, const Scalar& num2)
    {
#ifdef SCALABLE_CCD_USE_DOUBLE
        return num1 + num2 <= 1 / (1 - DBL_EPSILON);
#else
        return num1 + num2 <= 1 / (1 - FLT_EPSILON);
#endif
    }

    __device__ inline Scalar max_Linf_4(
        const Vector3& p1,
        const Vector3& p2,
        const Vector3& p3,
        const Vector3& p4,
        const Vector3& p1e,
        const Vector3& p2e,
        const Vector3& p3e,
        const Vector3& p4e)
    {
        return max(
            max((p1e - p1).lpNorm<Eigen::Infinity>(),
                (p2e - p2).lpNorm<Eigen::Infinity>()),
            max((p3e - p3).lpNorm<Eigen::Infinity>(),
                (p4e - p4).lpNorm<Eigen::Infinity>()));
    }

    __device__ void compute_face_vertex_tolerance(CCDData& data_in)
    {
        const Vector3 p000 = data_in.v0s - data_in.v1s;
        const Vector3 p001 = data_in.v0s - data_in.v3s;
        const Vector3 p011 =
            data_in.v0s - (data_in.v2s + data_in.v3s - data_in.v1s);
        const Vector3 p010 = data_in.v0s - data_in.v2s;
        const Vector3 p100 = data_in.v0e - data_in.v1e;
        const Vector3 p101 = data_in.v0e - data_in.v3e;
        const Vector3 p111 =
            data_in.v0e - (data_in.v2e + data_in.v3e - data_in.v1e);
        const Vector3 p110 = data_in.v0e - data_in.v2e;

        data_in.tol[0] = CONFIG.co_domain_tolerance
            / (3 * max_Linf_4(p000, p001, p011, p010, p100, p101, p111, p110));
        data_in.tol[1] = CONFIG.co_domain_tolerance
            / (3 * max_Linf_4(p000, p100, p101, p001, p010, p110, p111, p011));
        data_in.tol[2] = CONFIG.co_domain_tolerance
            / (3 * max_Linf_4(p000, p100, p110, p010, p001, p101, p111, p011));
    }

    __device__ void compute_edge_edge_tolerance(CCDData& data_in)
    {
        // WARNING: This differs from the original implementation in
        // Tight-Inclusion! This may be a bug, but results in better accuracy.
        const Vector3 p000 = data_in.v0s - data_in.v2s;
        const Vector3 p001 = data_in.v0s - data_in.v3s;
        const Vector3 p010 = data_in.v1s - data_in.v2s;
        const Vector3 p011 = data_in.v1s - data_in.v3s;
        const Vector3 p100 = data_in.v0e - data_in.v2e;
        const Vector3 p101 = data_in.v0e - data_in.v3e;
        const Vector3 p110 = data_in.v1e - data_in.v2e;
        const Vector3 p111 = data_in.v1e - data_in.v3e;

        data_in.tol[0] = CONFIG.co_domain_tolerance
            / (3 * max_Linf_4(p000, p001, p011, p010, p100, p101, p111, p110));
        data_in.tol[1] = CONFIG.co_domain_tolerance
            / (3 * max_Linf_4(p000, p001, p011, p010, p100, p101, p111, p110));
        data_in.tol[2] = CONFIG.co_domain_tolerance
            / (3 * max_Linf_4(p000, p100, p101, p001, p010, p110, p111, p011));
    }

    template <bool is_vf>
    __device__ __host__ void get_numerical_error(CCDData& data_in, bool use_ms)
    {
        Scalar filter;
        if (!use_ms) {
#ifdef SCALABLE_CCD_USE_DOUBLE
            if constexpr (is_vf) {
                filter = 6.661338147750939e-15;
            } else {
                filter = 6.217248937900877e-15;
            }
#else
            if constexpr (is_vf) {
                filter = 3.576279e-06;
            } else {
                filter = 3.337861e-06;
            }
#endif
        } else {
#ifdef SCALABLE_CCD_USE_DOUBLE
            if constexpr (is_vf) {
                filter = 7.549516567451064e-15;
            } else {
                filter = 7.105427357601002e-15;
            }
#else
            if constexpr (is_vf) {
                filter = 4.053116e-06;
            } else {
                filter = 3.814698e-06;
            }
#endif
        }

        const Vector3 max = data_in.v0s.cwiseAbs()
                                .cwiseMax(data_in.v1s.cwiseAbs())
                                .cwiseMax(data_in.v2s.cwiseAbs())
                                .cwiseMax(data_in.v3s.cwiseAbs())
                                .cwiseMax(data_in.v0e.cwiseAbs())
                                .cwiseMax(data_in.v1e.cwiseAbs())
                                .cwiseMax(data_in.v2e.cwiseAbs())
                                .cwiseMax(data_in.v3e.cwiseAbs())
                                .cwiseMax(Vector3::Ones());

        data_in.err = max.array() * max.array() * max.array() * filter;
    }

    __device__ Scalar
    calculate_vf(const CCDData& data_in, const BoxPrimatives& bp)
    {
        const Scalar v = (data_in.v0e[bp.dim] - data_in.v0s[bp.dim]) * bp.t
            + data_in.v0s[bp.dim];
        const Scalar t0 = (data_in.v1e[bp.dim] - data_in.v1s[bp.dim]) * bp.t
            + data_in.v1s[bp.dim];
        const Scalar t1 = (data_in.v2e[bp.dim] - data_in.v2s[bp.dim]) * bp.t
            + data_in.v2s[bp.dim];
        const Scalar t2 = (data_in.v3e[bp.dim] - data_in.v3s[bp.dim]) * bp.t
            + data_in.v3s[bp.dim];
        return v - (t1 - t0) * bp.u - (t2 - t0) * bp.v - t0;
    }

    __device__ Scalar
    calculate_ee(const CCDData& data_in, const BoxPrimatives& bp)
    {
        const Scalar ea0 = (data_in.v0e[bp.dim] - data_in.v0s[bp.dim]) * bp.t
            + data_in.v0s[bp.dim];
        const Scalar ea1 = (data_in.v1e[bp.dim] - data_in.v1s[bp.dim]) * bp.t
            + data_in.v1s[bp.dim];
        const Scalar eb0 = (data_in.v2e[bp.dim] - data_in.v2s[bp.dim]) * bp.t
            + data_in.v2s[bp.dim];
        const Scalar eb1 = (data_in.v3e[bp.dim] - data_in.v3s[bp.dim]) * bp.t
            + data_in.v3s[bp.dim];
        return ((ea1 - ea0) * bp.u + ea0) - ((eb1 - eb0) * bp.v + eb0);
    }

    template <bool is_vf>
    __device__ bool origin_in_inclusion_function(
        const CCDData& data_in,
        const CCDDomain& domain,
        Scalar& true_tol,
        bool& box_in)
    {
        box_in = true;
        true_tol = 0.0;
        BoxPrimatives bp;
        Scalar vmin = SCALAR_MAX;
        Scalar vmax = -SCALAR_MAX;
        for (bp.dim = 0; bp.dim < 3; bp.dim++) {
            vmin = SCALAR_MAX;
            vmax = -SCALAR_MAX;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        bp.b[0] = i;
                        bp.b[1] = j;
                        bp.b[2] = k;
                        bp.calculate_tuv(domain);

                        Scalar value;
                        if constexpr (is_vf) {
                            value = calculate_vf(data_in, bp);
                        } else {
                            value = calculate_ee(data_in, bp);
                        }

                        vmin = min(vmin, value);
                        vmax = max(vmax, value);
                    }
                }
            }

            // get the min and max in one dimension
            true_tol = max(true_tol, vmax - vmin);

            if (vmin - data_in.ms > data_in.err[bp.dim]
                || vmax + data_in.ms < -data_in.err[bp.dim]) {
                return false;
            }

            if (vmin + data_in.ms < -data_in.err[bp.dim]
                || vmax - data_in.ms > data_in.err[bp.dim]) {
                box_in = false;
            }
        }
        return true;
    }

    __device__ int split_dimension(const CCDData& data, Array3 width)
    {
        const Array3 res = width / data.tol;
        if (res[0] >= res[1] && res[0] >= res[2]) {
            return 0;
        } else if (res[1] >= res[0] && res[1] >= res[2]) {
            return 1;
        } else {
            assert(res[2] >= res[0] && res[2] >= res[1]);
            return 2;
        }
    }

    template <bool is_vf>
    __device__ inline bool bisect(
        const CCDDomain& domain,
        const int split,
        const Scalar* const toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
        Scalar data_toi,
#endif
        CCDBuffer* const buffer)
    {
        const SplitInterval halves(domain.tuv[split]);

        if (halves.first.lower >= halves.first.upper
            || halves.second.lower >= halves.second.upper) {
            return true;
        }

        buffer->push(domain).tuv[split] = halves.first;

        if (split == 0) {
            if (halves.second.lower <= *toi) {
                buffer->push(domain).tuv[0] = halves.second;
            }
        } else {
            if constexpr (is_vf) {
                if (split == 1) {
                    // check if u+v<=1
                    if (sum_less_than_one(
                            halves.second.lower, domain.tuv[2].lower)) {
                        buffer->push(domain).tuv[1] = halves.second;
                    }
                } else if (split == 2) {
                    // check if u+v<=1
                    if (sum_less_than_one(
                            halves.second.lower, domain.tuv[1].lower)) {
                        buffer->push(domain).tuv[2] = halves.second;
                    }
                }
            } else {
                buffer->push(domain).tuv[split] = halves.second;
            }
        }

        return false;
    }

} // namespace

// === the memory pool method =================================================

template <bool is_vf>
__global__ void compute_tolerance(CCDData* data, const int query_size)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= query_size)
        return;

    if constexpr (is_vf) {
        compute_face_vertex_tolerance(data[tx]);
    } else {
        compute_edge_edge_tolerance(data[tx]);
    }

    data[tx].nbr_checks = 0;
    get_numerical_error<is_vf>(data[tx], CONFIG.use_ms);
}

template <bool is_vf>
__global__ void
ccd_kernel(CCDBuffer* const buffer, CCDData* const data, Scalar* const toi)
{
    const int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= buffer->starting_size())
        return;

    // Get the tx element from the buffer without advancing the head.
    // This allows for better coalescing of memory access compared to pop().
    const CCDDomain domain_in = (*buffer)[tx];
    const int box_id = domain_in.query_id;
    const CCDData data_in = data[box_id];
    atomicAdd(&data[box_id].nbr_checks, 1);

    const Scalar min_t = domain_in.tuv[0].lower; // the time of this domain

#ifndef SCALABLE_CCD_TOI_PER_QUERY
    // if the time is larger than toi, return
    if (min_t >= *toi) {
#else
    if (min_t >= data_in.toi) {
#endif
        return;
    }

    // Check if exceeds max checks
    if (CONFIG.max_iter >= 0 && data_in.nbr_checks > CONFIG.max_iter) {
        return;
    }
    // else if (buffer.mp_remaining > buffer->capacity() / 2) { // overflow
    //     atomicCAS(&buffer->overflow_flag, 0, 1);
    //     return;
    // }

    Scalar true_tol = 0; // set by origin_in_inclusion_function
    bool box_in;         // set by origin_in_inclusion_function
    if (origin_in_inclusion_function<is_vf>(
            data_in, domain_in, true_tol, box_in)) {

        const Array3 widths(
            domain_in.tuv[0].upper - domain_in.tuv[0].lower,
            domain_in.tuv[1].upper - domain_in.tuv[1].lower,
            domain_in.tuv[2].upper - domain_in.tuv[2].lower);

        // Condition 1: the domain is smaller than the tolerance.
        if ((widths <= data_in.tol).all()) {
            atomicMin(toi, min_t);
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            atomicMin(&data[box_id].toi, min_t);
#endif
            return;
        }

        // Condition 2: the box is inside the epsilon box
        if (box_in && (CONFIG.allow_zero_toi || min_t > 0)) {
            atomicMin(toi, min_t);
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            atomicMin(&data[box_id].toi, min_t);
#endif
            return;
        }

        // Condition 3: real tolerance is smaller than the int tolerance
        if (true_tol <= CONFIG.co_domain_tolerance
            && (CONFIG.allow_zero_toi || min_t > 0)) {
            atomicMin(toi, min_t);
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            atomicMin(&data[box_id].toi, min_t);
#endif
            return;
        }

        // Get the next dimension to split
        const int split = split_dimension(data_in, widths);

        const bool sure_in = bisect<is_vf>(
            domain_in, split,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            &data_in.toi,
#else
            toi,
#endif
            buffer);

        // Condition 4 (rare): the interval is too small that overflow happens
        if (sure_in) {
            atomicMin(toi, min_t);
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            atomicMin(&data[box_id].toi, min_t);
#endif
            return;
        }
    }
}

template <bool is_vf>
bool ccd(
    thrust::device_vector<CCDData>& d_data,
    const std::shared_ptr<MemoryHandler> memory_handler,
    const int parallel_nbr,
    const int max_iter,
    const Scalar tol,
    const bool use_ms,
    const bool allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<int>& result_list,
#endif
    Scalar& toi)
{
    const int nbr = d_data.size();

    CCDBuffer* d_buffer;
    {
        const size_t unit_size = memory_handler->MAX_UNIT_SIZE;

        logger().trace(
            "CCD Buffer of size {:d} ({:g} GB)", unit_size,
            sizeof(CCDDomain) * unit_size / 1e9);

        CCDBuffer buffer;
        gpuErrchk(cudaMalloc(&buffer.m_data, sizeof(CCDDomain) * unit_size));
        buffer.m_starting_size = nbr;
        buffer.m_capacity = unit_size;
        buffer.m_head = 0;
        buffer.m_tail = nbr;
        buffer.m_overflow_flag = 0;

        gpuErrchk(cudaMalloc(&d_buffer, sizeof(CCDBuffer)));
        gpuErrchk(cudaMemcpy(
            d_buffer, &buffer, sizeof(CCDBuffer), cudaMemcpyHostToDevice));

        initialize_buffer<<<nbr / parallel_nbr + 1, parallel_nbr>>>(d_buffer);
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Initialize the global configuration variable
    {
        CCDConfig config;
        config.co_domain_tolerance = tol;
        config.use_ms = use_ms;
        config.allow_zero_toi = allow_zero_toi;
        config.max_iter = max_iter;
        gpuErrchk(cudaMemcpyToSymbol(CONFIG, &config, sizeof(CCDConfig)));
    }

    DeviceVariable d_toi(toi);

    // ---

    compute_tolerance<is_vf><<<nbr / parallel_nbr + 1, parallel_nbr>>>(
        thrust::raw_pointer_cast(d_data.data()), nbr);
    gpuErrchk(cudaDeviceSynchronize());

    logger().trace("Max queries: {:d}", memory_handler->MAX_QUERIES);

    int nbr_per_loop = nbr;
    logger().trace("Queue size t0: {:d}", nbr_per_loop);
    while (nbr_per_loop > 0) {
        ccd_kernel<is_vf><<<nbr_per_loop / parallel_nbr + 1, parallel_nbr>>>(
            d_buffer, thrust::raw_pointer_cast(d_data.data()), &d_toi);
        gpuErrchk(cudaDeviceSynchronize());

        // Update the starting index to reflect that the starting_size was
        // procesed.
        shift_queue_start<<<1, 1>>>(d_buffer);
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy(
            &nbr_per_loop, &(d_buffer->m_starting_size), sizeof(unsigned),
            cudaMemcpyDeviceToHost));

        logger().trace("Queue size: {:d}", nbr_per_loop);
        logger().trace("toi={:g}", *d_toi);
    }
    gpuErrchk(cudaDeviceSynchronize());

    toi = d_toi;

    int overflow;
    gpuErrchk(cudaMemcpy(
        &overflow, &(d_buffer->m_overflow_flag), sizeof(int),
        cudaMemcpyDeviceToHost));
    if (overflow) {
        return true;
    }

#ifdef SCALABLE_CCD_TOI_PER_QUERY
    CCDData* data_list = new CCDData[d_data.size()];
    // CCDConfig CONFIG = new CCDConfig[1];
    gpuErrchk(cudaMemcpy(
        data_list, d_data, sizeof(CCDData) * d_data.size(),
        cudaMemcpyDeviceToHost));
    // std::vector<std::pair<std::string, std::string>> symbolic_tois;
    int tpq_cnt = 0;
    for (size_t i = 0; i < d_data.size(); i++) {
        cuda::Rational ra(data_list[i].toi);
        if (data_list[i].toi > 1)
            continue;
        tpq_cnt++;
        // symbolic_tois.emplace_back(ra.get_numerator_str(),
        //                            ra.get_denominator_str());
        // auto pair = make_pair(ra.get_numerator_str(),
        // ra.get_denominator_str());
        std::string triple[4] = { std::to_string(data_list[i].aid),
                                  std::to_string(data_list[i].bid),
                                  ra.get_numerator_str(),
                                  ra.get_denominator_str() };
        // if (data_list[i].toi <= .00000382)
        //   printf("not one toi %s, %s, %e\n", triple[0].c_str(),
        //   triple[1].c_str(),
        //          data_list[i].toi);
        r.j_object["toi_per_query"].push_back(triple);
    }
    logger().trace("tpq_cnt: {:d}", tpq_cnt);
    free(data_list);
    gpuErrchk(cudaDeviceSynchronize());
    // json jtmp(symbolic_tois.begin(), symbolic_tois.end());
    // std::cout << jtmp.dump(4) << std::endl;
    // r.j_object.insert(jtmp.begin(), jtmp.end());
    // r.j_object.push_back(r.j_object.end(), jtmp.begin(), jtmp.end());
    // r.j_object.push_back(symbolic_tois);
    //  symbolic_tois.end());

    // json j_vec(falseNegativePairs);
    // r.j_object.insert(r.j_object.end(), symbolic_tois.begin(),
    //                   symbolic_tois.end());

    // std::ofstream o(outputFilePath);
    // o << std::setw(4) << j_vec << std::endl;
    // auto outputFilename = std::filesystem::path(std::to_string(iter) +
    // ".json"); outputFilename = outputFolder / outputFilename; std::ofstream
    // o(outputFilename); o << std::setw(4) << j << std::endl;
#endif

    return false;
}

// === Template instantiation ==================================================

template __global__ void compute_tolerance<false>(CCDData*, const int);
template __global__ void compute_tolerance<true>(CCDData*, const int);

template __global__ void
ccd_kernel<false>(CCDBuffer* const, CCDData* const, Scalar* const);
template __global__ void
ccd_kernel<true>(CCDBuffer* const, CCDData* const, Scalar* const);

// clang-format off
template bool ccd<false>(
    thrust::device_vector<CCDData>&, const std::shared_ptr<MemoryHandler>,
    const int, const int, const Scalar, const bool, const bool,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<int>&,
#endif
    Scalar&);
template bool ccd<true>(
    thrust::device_vector<CCDData>&, const std::shared_ptr<MemoryHandler>,
    const int, const int, const Scalar, const bool, const bool,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<int>&,
#endif
    Scalar&);
// clang-format on

} // namespace scalable_ccd::cuda
