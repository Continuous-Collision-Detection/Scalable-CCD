#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/broad_phase/utils.cuh>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/utils/pca.hpp>
#include <scalable_ccd/utils/logger.hpp>

#include <igl/write_triangle_mesh.h>

#include <fstream>
#include <unistd.h>
#include <filesystem>

bool file_exists(const char* fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

int main(int argc, char** argv)
{
    using namespace scalable_ccd;
    using namespace scalable_ccd::cuda;

    logger().set_level(spdlog::level::trace);
    std::vector<char*> compare;

    MemoryHandler* memhandle = new MemoryHandler();

    char* filet0;
    char* filet1;

    filet0 = argv[1];
    if (file_exists(argv[2]))
        filet1 = argv[2];
    else
        filet1 = argv[1];

    std::vector<scalable_ccd::cuda::AABB> boxes;
    Eigen::MatrixXd vertices_t0;
    Eigen::MatrixXd vertices_t1;
    Eigen::MatrixXd pca_vertices_t0;
    Eigen::MatrixXd pca_vertices_t1;
    Eigen::MatrixXi faces;
    Eigen::MatrixXi edges;

    int nbox = 0;
    int parallel = 0;
    // bool evenworkload = false;
    int devcount = 1;
    // bool pairing = false;
    // bool sharedqueue_mgpu = false;
    // bool bigworkerqueue = false;
    bool pca = false;

    int memlimit = 0;

    int o;
    while ((o = getopt(argc, argv, "c:n:b:p:d:v:WPQZ")) != -1) {
        switch (o) {
        case 'c':
            optind--;
            for (; optind < argc && *argv[optind] != '-'; optind++) {
                compare.push_back(argv[optind]);
            }
            break;
        // case 'n':
        //   N = atoi(optarg);
        //   break;
        case 'b':
            nbox = atoi(optarg);
            break;
        case 'v':
            memlimit = atoi(optarg);
            break;
        case 'p':
            parallel = std::stoi(optarg);
            break;
        case 'd':
            devcount = atoi(optarg);
            break;
        case 'P':
            pca = true;
            break;
        }
    }

    parse_mesh(filet0, filet1, vertices_t0, vertices_t1, faces, edges);
    logger().trace(
        "vertices_t0 : {:d} x {:d}", vertices_t0.rows(), vertices_t0.cols());
    if (pca) {
        scalable_ccd::nipals_pca(vertices_t0, vertices_t1);

        std::string filet0Str(filet0);
        std::filesystem::path p(filet0Str);
        std::filesystem::path filename = p.filename();
        std::string ext = filet0Str.substr(filet0Str.rfind('.') + 1);
        std::filesystem::path current_path = std::filesystem::current_path();
        std::string outname = current_path.parent_path().string() + "/"
            + filename.stem().string() + "_pca." + ext;
        igl::write_triangle_mesh(outname, vertices_t0, faces);
    }
    constructBoxes(vertices_t0, vertices_t1, edges, faces, boxes);
    size_t N = boxes.size();

    std::vector<std::pair<int, int>> overlaps;
    int2* d_overlaps; // device
    int* d_count;     // device
    int tidstart = 0;

    if (devcount == 1)
        runBroadPhase(
            boxes.data(), memhandle, N, nbox, overlaps, d_overlaps, d_count,
            parallel, tidstart, devcount, memlimit);
    else
        runBroadPhaseMultiGPU(
            boxes.data(), N, nbox, overlaps, parallel, devcount);

    logger().debug("Final CPU overlaps size: {:d}", overlaps.size());

    for (auto compFile : compare) {
        compare_mathematica(overlaps, compFile);
    }
}