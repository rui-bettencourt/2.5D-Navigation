#include "mesh_nav/VoxelSDF.h"
#include <cassert>
#include <fstream>
#include <iostream>

bool VoxelGrid::loadFromFile(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    double origin_x, origin_y, origin_z;
    double voxel_size;

    // Read origin (bbox_min) and voxel size
    in.read(reinterpret_cast<char*>(&origin_x), sizeof(double));
    in.read(reinterpret_cast<char*>(&origin_y), sizeof(double));
    in.read(reinterpret_cast<char*>(&origin_z), sizeof(double));
    in.read(reinterpret_cast<char*>(&voxel_size), sizeof(double));

    origin_ = Eigen::Vector3f(origin_x, origin_y, origin_z);
    resolution_ = static_cast<float>(voxel_size);
    inv_resolution_ = 1.0f / resolution_;

    // Read dimensions
    in.read(reinterpret_cast<char*>(&nx_), sizeof(int));
    in.read(reinterpret_cast<char*>(&ny_), sizeof(int));
    in.read(reinterpret_cast<char*>(&nz_), sizeof(int));
    dims_ = Eigen::Vector3i(nx_, ny_, nz_);

    std::cout << "Loading SDF from: " << filename << std::endl;
    std::cout << "Origin: " << origin_.transpose() << std::endl;
    std::cout << "Resolution: " << resolution_ << std::endl;
    std::cout << "Grid Size: " << nx_ << " x " << ny_ << " x " << nz_ << std::endl;

    const size_t n = static_cast<size_t>(nx_) * ny_ * nz_;

    sdf_.resize(n);
    gradients_.resize(n);

    // Read sdf and gradient data
    in.read(reinterpret_cast<char*>(sdf_.data()), n * sizeof(float));
    // First, read into a temporary float buffer
    std::vector<float> gradient_floats(3 * n);
    in.read(reinterpret_cast<char*>(gradient_floats.data()), 3 * n * sizeof(float));

    // Now convert to Eigen::Vector3f
    gradients_.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        gradients_[i] = Eigen::Vector3f(
            gradient_floats[i * 3 + 0],
            gradient_floats[i * 3 + 1],
            gradient_floats[i * 3 + 2]
        );
    }

    return true;
}

int VoxelGrid::index(int i, int j, int k) const {
    return i * dims_[1] * dims_[2] + j * dims_[2] + k;
}

VoxelGrid::VoxelQuery VoxelGrid::computeVoxelQuery(const Eigen::Vector3f& pt) const
{
    Eigen::Vector3f local = (pt - origin_) * inv_resolution_;
    Eigen::Vector3i idx = local.array().floor().cast<int>();
    Eigen::Vector3f frac = local - idx.cast<float>();
    return {idx, frac};
}

float VoxelGrid::getDistanceAtPoint(const Eigen::Vector3f& pt) const
{
    auto q = computeVoxelQuery(pt);

    // Check if inside safe region for interpolation
    if (q.index.x() < 0 || q.index.y() < 0 || q.index.z() < 0 ||
        q.index.x() >= dims_.x() - 1 ||
        q.index.y() >= dims_.y() - 1 ||
        q.index.z() >= dims_.z() - 1)
    {
        // Outside safe region — use nearest voxel value
        Eigen::Vector3i clamped = q.index.cwiseMax(0).cwiseMin(dims_ - Eigen::Vector3i(1, 1, 1));
        return sdf_[index(clamped.x(), clamped.y(), clamped.z())];
    }

    // Inside safe region — interpolate
    return trilinearInterpolate(q.index, q.weights);
}


Eigen::Vector3f VoxelGrid::getVectorAtPoint(const Eigen::Vector3f& pt) const
{
    auto q = computeVoxelQuery(pt);

    if (q.index.x() < 0 || q.index.y() < 0 || q.index.z() < 0 ||
        q.index.x() >= dims_.x() - 1 ||
        q.index.y() >= dims_.y() - 1 ||
        q.index.z() >= dims_.z() - 1)
    {
        Eigen::Vector3i clamped = q.index.cwiseMax(0).cwiseMin(dims_ - Eigen::Vector3i(1, 1, 1));
        return gradients_[index(clamped.x(), clamped.y(), clamped.z())];
    }

    return trilinearInterpolateVector(q.index, q.weights);
}


float VoxelGrid::trilinearInterpolate(const Eigen::Vector3i& base, const Eigen::Vector3f& w) const
{
    float result = 0.0f;

    for (int dx = 0; dx <= 1; ++dx)
    for (int dy = 0; dy <= 1; ++dy)
    for (int dz = 0; dz <= 1; ++dz)
    {
        int i = base[0] + dx;
        int j = base[1] + dy;
        int k = base[2] + dz;

        if (i < 0 || i >= dims_[0] || j < 0 || j >= dims_[1] || k < 0 || k >= dims_[2])
            continue;

        float weight = ((dx ? w[0] : 1 - w[0]) *
                        (dy ? w[1] : 1 - w[1]) *
                        (dz ? w[2] : 1 - w[2]));

        result += weight * sdf_[index(i, j, k)];
    }

    return result;
}

Eigen::Vector3f VoxelGrid::trilinearInterpolateVector(const Eigen::Vector3i& base, const Eigen::Vector3f& w) const
{
    Eigen::Vector3f result(0, 0, 0);

    for (int dx = 0; dx <= 1; ++dx)
    for (int dy = 0; dy <= 1; ++dy)
    for (int dz = 0; dz <= 1; ++dz)
    {
        int i = base[0] + dx;
        int j = base[1] + dy;
        int k = base[2] + dz;

        if (i < 0 || i >= dims_[0] || j < 0 || j >= dims_[1] || k < 0 || k >= dims_[2])
            continue;

        float weight = ((dx ? w[0] : 1 - w[0]) *
                        (dy ? w[1] : 1 - w[1]) *
                        (dz ? w[2] : 1 - w[2]));

        result += weight * gradients_[index(i, j, k)];
    }

    return result;
}
