#include <igl/read_triangle_mesh.h>
#include <igl/signed_distance.h>
#include <igl/AABB.h>
#include <igl/fast_winding_number.h>

#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <chrono>

class SDFfromPLY
{
public:
    SDFfromPLY() = default;

    // Loads the mesh and initializes Fast Winding structures
    bool loadMesh(const std::string& ply_filename);

    // Query signed distance and closest point for a single query point
    // Returns true if query successful, false otherwise
    bool queryDistance(
        const Eigen::RowVector3d& query_point,
        double& distance,
        Eigen::RowVector3d& closest_point) const;

    const Eigen::MatrixXd& getVertices() const { return V_; }

private:
    Eigen::MatrixXd V_;
    Eigen::MatrixXi F_;
    igl::AABB<Eigen::MatrixXd, 3> tree_;
    igl::FastWindingNumberBVH fwn_bvh_;
};
