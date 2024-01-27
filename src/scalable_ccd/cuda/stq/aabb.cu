#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/stq/aabb.cuh>

#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tbb/parallel_for.h>

namespace scalable_ccd::cuda::stq {

__host__ __device__ bool is_face(const Aabb& x) { return x.vertexIds.z >= 0; };

__host__ __device__ bool is_face(const int3& vids) { return vids.z >= 0; };

__host__ __device__ bool is_edge(const Aabb& x)
{
    return x.vertexIds.z < 0 && x.vertexIds.y >= 0;
};

__host__ __device__ bool is_edge(const int3& vids)
{
    return vids.z < 0 && vids.y >= 0;
};

__host__ __device__ bool is_vertex(const Aabb& x)
{
    return x.vertexIds.z < 0 && x.vertexIds.y < 0;
};

__host__ __device__ bool is_vertex(const int3& vids)
{
    return vids.z < 0 && vids.y < 0;
};

__host__ __device__ bool is_valid_pair(const Aabb& a, const Aabb& b)
{
    return (is_vertex(a) && is_face(b)) || (is_face(a) && is_vertex(b))
        || (is_edge(a) && is_edge(b));
};

__host__ __device__ bool is_valid_pair(const int3& a, const int3& b)
{
    return (is_vertex(a) && is_face(b)) || (is_face(a) && is_vertex(b))
        || (is_edge(a) && is_edge(b));
};

void merge_local_boxes(
    const tbb::enumerable_thread_specific<std::vector<Aabb>>& storages,
    std::vector<Aabb>& boxes)
{
    size_t num_boxes = boxes.size();
    for (const auto& local_boxes : storages) {
        num_boxes += local_boxes.size();
    }
    // serial merge!
    boxes.reserve(num_boxes);
    for (const auto& local_boxes : storages) {
        boxes.insert(boxes.end(), local_boxes.begin(), local_boxes.end());
    }
}

float nextafter_up(float x)
{
    return nextafterf(x, x + std::numeric_limits<float>::max());
};
float nextafter_down(float x)
{
    return nextafterf(x, x - std::numeric_limits<float>::max());
};

void addEdges(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    Scalar inflation_radius,
    std::vector<Aabb>& boxes)
{
    tbb::enumerable_thread_specific<std::vector<Aabb>> storages;
    tbb::parallel_for(0, static_cast<int>(edges.rows()), 1, [&](int& i) {
        // for (unsigned long i = 0; i < edges.rows(); i++) {
        Eigen::MatrixXd edge_vertex0_t0 = vertices_t0.row(edges(i, 0));
        Eigen::MatrixXd edge_vertex1_t0 = vertices_t0.row(edges(i, 1));
        Eigen::MatrixXd edge_vertex0_t1 = vertices_t1.row(edges(i, 0));
        Eigen::MatrixXd edge_vertex1_t1 = vertices_t1.row(edges(i, 1));

        Eigen::MatrixXd points(4, edge_vertex0_t0.size());
        points.row(0) = edge_vertex0_t0;
        points.row(1) = edge_vertex1_t0;
        points.row(2) = edge_vertex0_t1;
        points.row(3) = edge_vertex1_t1;

        int vertexIds[3] = { edges(i, 0), edges(i, 1), -edges(i, 0) - 1 };
#ifdef SCALABLE_CCD_WITH_DOUBLE
        Eigen::Vector3d lower_bound =
            points.colwise().minCoeff().array() - inflation_radius;
        Eigen::Vector3d upper_bound =
            points.colwise().maxCoeff().array() + inflation_radius;
#else

    Eigen::MatrixXf lower_bound =
        points.colwise().minCoeff().unaryExpr(&nextafter_down).array() - nextafter_up(inflation_radius);
    Eigen::MatrixXf upper_bound =
        points.colwise().maxCoeff().unaryExpr(&nextafter_up).array() + nextafter_up(inflation_radius);
#endif
        auto& local_boxes = storages.local();
        local_boxes.emplace_back(
            boxes.size() + i, i, vertexIds, lower_bound.data(),
            upper_bound.data());
    });
    merge_local_boxes(storages, boxes);
}

void addVertices(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    Scalar inflation_radius,
    std::vector<Aabb>& boxes)
{
    tbb::enumerable_thread_specific<std::vector<Aabb>> storages;
    tbb::parallel_for(0, static_cast<int>(vertices_t0.rows()), 1, [&](int& i) {
        // for (unsigned long i = 0; i < vertices_t0.rows(); i++) {
        Eigen::MatrixXd vertex_t0 = vertices_t0.row(i);
        Eigen::MatrixXd vertex_t1 = vertices_t1.row(i);

        Eigen::MatrixXd points(2, vertex_t0.size());
        points.row(0) = vertex_t0;
        points.row(1) = vertex_t1;

        int vertexIds[3] = { i, -i - 1, -i - 1 };

#ifdef SCALABLE_CCD_WITH_DOUBLE
        Eigen::MatrixXd lower_bound =
            points.colwise().minCoeff().array() - inflation_radius;
        Eigen::MatrixXd upper_bound =
            points.colwise().maxCoeff().array() + inflation_radius;
#else

    Eigen::MatrixXf lower_bound =
        points.colwise().minCoeff().unaryExpr(&nextafter_down).array() - nextafter_up(inflation_radius);;
    Eigen::MatrixXf upper_bound =
    points.colwise().maxCoeff().unaryExpr(&nextafter_up).array() +  nextafter_up(inflation_radius);;
#endif
        auto& local_boxes = storages.local();
        local_boxes.emplace_back(
            boxes.size() + i, i, vertexIds, lower_bound.data(),
            upper_bound.data());
    });
    merge_local_boxes(storages, boxes);
}

void addFaces(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& faces,
    Scalar inflation_radius,
    std::vector<Aabb>& boxes)
{
    tbb::enumerable_thread_specific<std::vector<Aabb>> storages;
    tbb::parallel_for(0, static_cast<int>(faces.rows()), 1, [&](int& i) {
        // for (unsigned long i = 0; i < faces.rows(); i++) {
        Eigen::MatrixXd face_vertex0_t0 = vertices_t0.row(faces(i, 0));
        Eigen::MatrixXd face_vertex1_t0 = vertices_t0.row(faces(i, 1));
        Eigen::MatrixXd face_vertex2_t0 = vertices_t0.row(faces(i, 2));
        Eigen::MatrixXd face_vertex0_t1 = vertices_t1.row(faces(i, 0));
        Eigen::MatrixXd face_vertex1_t1 = vertices_t1.row(faces(i, 1));
        Eigen::MatrixXd face_vertex2_t1 = vertices_t1.row(faces(i, 2));

        Eigen::MatrixXd points(6, face_vertex0_t0.size());
        points.row(0) = face_vertex0_t0;
        points.row(1) = face_vertex1_t0;
        points.row(2) = face_vertex2_t0;
        points.row(3) = face_vertex0_t1;
        points.row(4) = face_vertex1_t1;
        points.row(5) = face_vertex2_t1;

        int vertexIds[3] = { faces(i, 0), faces(i, 1), faces(i, 2) };

#ifdef SCALABLE_CCD_WITH_DOUBLE
        Eigen::Vector3d lower_bound = points.colwise().minCoeff().array()
            - static_cast<double>(inflation_radius);
        Eigen::Vector3d upper_bound = points.colwise().maxCoeff().array()
            + static_cast<double>(inflation_radius);
#else

    Eigen::MatrixXf lower_bound =
        points.colwise().minCoeff().unaryExpr(&nextafter_down).array() - nextafter_up(inflation_radius);;
    Eigen::MatrixXf upper_bound =
        points.colwise().maxCoeff().unaryExpr(&nextafter_up).array() + nextafter_up(inflation_radius);;
#endif
        auto& local_boxes = storages.local();
        local_boxes.emplace_back(
            boxes.size() + i, i, vertexIds, lower_bound.data(),
            upper_bound.data());
    });
    merge_local_boxes(storages, boxes);
};

} // namespace scalable_ccd::cuda::stq