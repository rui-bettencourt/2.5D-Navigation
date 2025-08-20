#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include "mesh_nav/SDFfromPLY.h"
#include <omp.h>
#include <atomic>


// Helper to linearize 3D index
inline size_t idx(int i, int j, int k, int ny, int nz)
{
    return static_cast<size_t>(i) * ny * nz + static_cast<size_t>(j) * nz + static_cast<size_t>(k);
}

int main()
{
    omp_set_num_threads(omp_get_max_threads());
    // std::string mesh_file = "/home/rui/ds/testscilindros/easy/obstacles.ply";
    // std::string output_file = "/home/rui/ds/testscilindros/easy/sdf2.bin";
    std::string mesh_file = "/home/rui/ds/testsiros2025/obstacles.ply";
    std::string output_file = "/home/rui/ds/testsiros2025/sdf.bin";

    SDFfromPLY sdfFromPLY;
    if (!sdfFromPLY.loadMesh(mesh_file))
    {
        std::cerr << "Failed to load mesh." << std::endl;
        return -1;
    }

    // Compute bounding box with margin (5% of bbox size)
    Eigen::Vector3d bbox_min = sdfFromPLY.getVertices().colwise().minCoeff();
    Eigen::Vector3d bbox_max = sdfFromPLY.getVertices().colwise().maxCoeff();
    Eigen::Vector3d bbox_size = bbox_max - bbox_min;
    double margin = 0.2; // 20% margin
    bbox_min -= margin * bbox_size;
    bbox_max += margin * bbox_size;

    // Compute center of bounding box
    Eigen::Vector3d bbox_center = 0.5 * (bbox_min + bbox_max);

    // Set voxel size (0.01 to 0.05 meters)
    double voxel_size = 0.1;

    // Compute grid dimensions
    int nx = static_cast<int>((bbox_max.x() - bbox_min.x()) / voxel_size) + 1;
    int ny = static_cast<int>((bbox_max.y() - bbox_min.y()) / voxel_size) + 1;
    int nz = static_cast<int>((bbox_max.z() - bbox_min.z()) / voxel_size) + 1;

    std::cout << "Grid dims: " << nx << " x " << ny << " x " << nz << std::endl;

    // Compute new grid origin: so that grid is centered on the mesh
    Eigen::Vector3d grid_origin = bbox_center - 0.5 * Eigen::Vector3d((nx - 1) * voxel_size,
                                                                    (ny - 1) * voxel_size,
                                                                    (nz - 1) * voxel_size);

    // Allocate arrays
    std::vector<float> sdf_grid(static_cast<size_t>(nx) * ny * nz, 0.f);
    std::vector<float> vec_grid(static_cast<size_t>(nx) * ny * nz * 3, 0.f);

    // Iterate over grid voxels
    size_t total_voxels = static_cast<size_t>(nx) * ny * nz;
    // size_t counter = 0;
    std::atomic<size_t> counter(0);

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                double x = grid_origin.x() + i * voxel_size;
                double y = grid_origin.y() + j * voxel_size;
                double z = grid_origin.z() + k * voxel_size;

                Eigen::RowVector3d pt(x, y, z);
                double dist;
                Eigen::RowVector3d closest_pt;

                bool ok = sdfFromPLY.queryDistance(pt, dist, closest_pt);
                if (!ok)
                {
                    dist = 0.0;
                    closest_pt = pt;
                }

                float signed_dist = static_cast<float>(dist);
                Eigen::RowVector3f vec_to_surface = (pt - closest_pt).cast<float>();

                size_t lin_idx = idx(i, j, k, ny, nz);
                sdf_grid[lin_idx] = signed_dist;
                vec_grid[3 * lin_idx + 0] = vec_to_surface.x();
                vec_grid[3 * lin_idx + 1] = vec_to_surface.y();
                vec_grid[3 * lin_idx + 2] = vec_to_surface.z();

                // Thread-safe increment
                size_t count = ++counter;
                if (count % 100000 == 0) // Print less often
                    std::cout << "\rProcessed voxels: " << count << " / " << total_voxels << std::flush;
            }
        }
    }

    std::cout << "\nVoxel grid generation done." << std::endl;

    // Save metadata
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs)
    {
        std::cerr << "Failed to open output file." << std::endl;
        return -1;
    }

    // ⚠️ Save new origin, not bbox_min
    ofs.write(reinterpret_cast<const char*>(&grid_origin.x()), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&grid_origin.y()), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&grid_origin.z()), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&voxel_size), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&nx), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&ny), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&nz), sizeof(int));

    // Save data
    ofs.write(reinterpret_cast<const char*>(sdf_grid.data()), sdf_grid.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(vec_grid.data()), vec_grid.size() * sizeof(float));
    ofs.close();

    std::cout << "Voxel grid saved to " << output_file << std::endl;


    return 0;
}
