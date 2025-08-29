#include "mesh_nav/VoxelSDF.h"
#include "mesh_nav/SDFfromPLY.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>

// This function dumps a slice at fixed z into a CSV file for easy plotting
void dumpSliceAtZ(const VoxelGrid& grid, float z_world, const std::string& filename)
{
    int nx = grid.getDims()[0];
    int ny = grid.getDims()[1];
    float voxel_size = grid.getResolution();
    Eigen::Vector3f origin = grid.getOrigin();

    // Prepare 2D matrix to hold distances (x,y)
    std::vector<std::vector<float>> slice_data(nx, std::vector<float>(ny, 0.f));

    // For each x,y index, convert to world coord (center of voxel) + given z
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            // Compute world coordinate at voxel center
            float x = origin[0] + (ix + 0.5f) * voxel_size;
            float y = origin[1] + (iy + 0.5f) * voxel_size;
            Eigen::Vector3f pt(x, y, z_world);

            // Query signed distance at that point
            float dist = grid.getDistanceAtPoint(pt);
            slice_data[ix][iy] = dist;
        }
    }

    // Write to CSV file: first row is y coords, first column is x coords
    std::ofstream out(filename);
    if (!out.is_open())
    {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Header: y coords
    out << "x\\y";
    for (int iy = 0; iy < ny; ++iy)
    {
        float y = origin[1] + (iy + 0.5f) * voxel_size;
        out << "," << y;
    }
    out << "\n";

    // Data rows
    for (int ix = 0; ix < nx; ++ix)
    {
        float x = origin[0] + (ix + 0.5f) * voxel_size;
        out << x;
        for (int iy = 0; iy < ny; ++iy)
        {
            out << "," << std::fixed << std::setprecision(6) << slice_data[ix][iy];
        }
        out << "\n";
    }

    std::cout << "Slice at z=" << z_world << " dumped to " << filename << std::endl;
}

int main()
{
    VoxelGrid vg;

    // if (!vg.loadFromFile("/home/rui/ds/testscilindros/easy/sdf2.bin"))
    // if (!vg.loadFromFile("/home/rui/ds/testsiros2025/sdf.bin"))
    if (!vg.loadFromFile("/home/rui/ds/testsiros2025/robot_sdf_002.bin"))
    {
        std::cerr << "Failed to load SDF file.\n";
        return 1;
    }

    std::cout << "Loaded the SDF!" << std::endl;

    SDFfromPLY sdfFromPLY;
    // sdfFromPLY.loadMesh("/home/rui/ds/testscilindros/easy/obstacles.ply");
    sdfFromPLY.loadMesh("/home/rui/ds/testsiros2025/sdf.bin");

    for (int i = 1; i <= 10; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        Eigen::Vector3f pt(1+i*0.5, 0.0f, 0.5f);
        float d = vg.getDistanceAtPoint(pt);
        Eigen::Vector3f vec = vg.getVectorAtPoint(pt);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = t2 - t1;
        std::cout << "Query time for x " << 1+i*0.5 << " : " << duration.count() * 1000 << " ms" << std::endl;


        // std::cout << "Distance: " << d << "\n";
        // std::cout << "Vector: " << vec.transpose() << "\n";

        // Ground truth from mesh
        double dist;
        Eigen::RowVector3d closest_pt;
        sdfFromPLY.queryDistance(pt.cast<double>(), dist, closest_pt);

        Eigen::Vector3f gt_vec = pt - closest_pt.transpose().cast<float>();

        std::cout << "SDF distance: " << d << "\n";
        std::cout << "GT distance:  " << std::abs(dist) << "\n";
        std::cout << "SDF vector:   " << vec.transpose() << "\n";
        std::cout << "GT vector:    " << gt_vec.transpose() << "\n";
        std::cout << "Error (distance): " << std::abs(d - std::abs(dist)) << "\n";
        std::cout << "Error (vector mag): " << (vec - gt_vec).norm() << "\n";
    }

    // Dump slice at z=0.5m
    dumpSliceAtZ(vg, 1.0f, "/home/rui/ds/testsiros2025/slice_robot_z_0_5.csv");

    return 0;
}