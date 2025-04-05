#pragma once

#include <scalable_ccd/scalar.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

namespace scalable_ccd::cuda {

#ifdef SCALABLE_CCD_USE_DOUBLE

__device__ constexpr double SCALAR_MAX = scalable_ccd::SCALAR_MAX;

using Scalar2 = double2;
using Scalar3 = double3;

__host__ __device__ inline Scalar2
make_Scalar2(const Scalar& a, const Scalar& b)
{
    return make_double2(a, b);
}

__host__ __device__ inline Scalar3
make_Scalar3(const Scalar& a, const Scalar& b, const Scalar& c)
{
    return make_double3(a, b, c);
}

#else

__device__ constexpr float SCALAR_MAX = scalable_ccd::SCALAR_MAX;

using Scalar2 = float2;
using Scalar3 = float3;

__host__ __device__ inline Scalar2
make_Scalar2(const Scalar& a, const Scalar& b)
{
    return make_float2(a, b);
}

__host__ __device__ inline Scalar3
make_Scalar3(const Scalar& a, const Scalar& b, const Scalar& c)
{
    return make_float3(a, b, c);
}

#endif

// __host__ __device__ struct half3 {
//     __half x;
//     __half y;
//     __half z;
// };

// __host__ __device__ inline half3 make_half3(__half x, __half y, __half z)
// {
//     half3 t;
//     t.x = x;
//     t.y = y;
//     t.z = z;
//     return t;
// }

// __host__ __device__ inline half3 make_half3(float x, float y, float z)
// {
//     half3 t;
//     t.x = __float2half(x);
//     t.y = __float2half(y);
//     t.z = __float2half(z);
//     return t;
// }

__device__ Scalar3 operator+(const Scalar3& a, const Scalar3& b);

__device__ Scalar3 operator-(const Scalar3& a, const Scalar3& b);

__device__ Scalar3 __fdividef(const Scalar3& a, const Scalar b);

__device__ Scalar3 __powf(const Scalar3& a, const Scalar b);

__device__ Scalar3 abs(const Scalar3& a);

} // namespace scalable_ccd::cuda