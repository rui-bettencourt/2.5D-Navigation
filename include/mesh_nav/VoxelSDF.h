#pragma once

#include <Eigen/Dense>
#include <vector>

class VoxelGrid
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct VoxelQuery
    {
        Eigen::Vector3i index;   // Base voxel (i,j,k)
        Eigen::Vector3f weights; // Fractional offset (fx, fy, fz)
    };

    VoxelGrid() = default;

    bool loadFromFile(const std::string& filename);

    float getDistanceAtPoint(const Eigen::Vector3f& pt) const;
    Eigen::Vector3f getVectorAtPoint(const Eigen::Vector3f& pt) const;

    const Eigen::Vector3i& getDims() const { return dims_; }
    const Eigen::Vector3f& getOrigin() const { return origin_; }
    const float& getResolution() const { return resolution_; }

private:
    int nx_, ny_, nz_;                     // Grid dimensions
    float resolution_;                     // Voxel size
    Eigen::Vector3f origin_;               // Grid origin
    std::vector<float> sdf_;              // Distance field values
    std::vector<Eigen::Vector3f> gradients_; // Gradient vectors
    float inv_resolution_;
    Eigen::Vector3i dims_;

    inline int index(int i, int j, int k) const;

    VoxelQuery computeVoxelQuery(const Eigen::Vector3f& pt) const;

    float trilinearInterpolate(const Eigen::Vector3i& base, const Eigen::Vector3f& w) const;
    Eigen::Vector3f trilinearInterpolateVector(const Eigen::Vector3i& base, const Eigen::Vector3f& w) const;
};
