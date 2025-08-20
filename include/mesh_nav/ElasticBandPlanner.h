#ifndef ELASTIC_BAND_PLANNER_H
#define ELASTIC_BAND_PLANNER_H

#include <vector>
#include <string>
#include <memory>

#include "mesh_nav/VoxelSDF.h"

namespace ebp {

using Vec3 = std::vector<double>; // expected size 3 for positions/forces
using Row = std::vector<double>;
using PathMatrix = std::vector<Row>;
using MeshHandle = std::any; // opaque handle, can later be replaced with your own robot model representation

// ------------------------ Planner ------------------------
class ElasticBandPlanner {
public:
    ElasticBandPlanner(double k_attraction_base=0.2,
                       double k_repulsion_base=0.2,
                       double k_attraction_joints=0.1,
                       double k_repulsion_joints=0.1,
                       double k_repulsion_robot_joints=0.1,
                       double k_update_joints=0.1,
                       double k_orientation=0.1,
                       double k_orientation_from_base=0.1,
                       double k_position_from_orientation=0.0,
                       double k_safety_joints=0.05,
                       double obstacle_threshold=1.5);

    // Initialize the planner with a voxel grid (from .bin file) and default robot points
    bool initialize(const std::string& sdf_bin_path);

    // Update path (currently does not use RobotModel)
    PathMatrix update_path(const PathMatrix& path,
                           int iterations=100, double convergence_threshold=1e-3);

    // Optional: setters/getters
    void set_dynamic_safety(bool on) { dynamic_safety = on; }

private:
    VoxelGrid* voxel_grid_ = nullptr;

    // gains
    double k_attraction_base, k_repulsion_base, k_repulsion_joints, k_repulsion_robot_joints;
    double k_attraction_joints, k_update_joints, k_orientation, k_orientation_from_base, k_position_from_orientation;
    double k_safety_joints;
    double obstacle_threshold;
    double weak_torque_threshold;
    double min_distance_to_obstacle = 0.05;
    bool dynamic_safety;
    double center_activation_safety;
    double radius_joint;
    int number_points_robot;
    std::vector<std::vector<double>> robot_points_;

    // internal state used during computation
    PathMatrix path_matrix;
    std::vector<PathMatrix> history;

    // core helpers
    static std::vector<double> compute_attractive_force(const std::vector<double>& prev_pos,
                                                        const std::vector<double>& current_pos,
                                                        const std::vector<double>& next_pos,
                                                        double obstacle_threshold);

    static std::vector<double> compute_force_from_dist(double min_distance,
                                                       const std::vector<double>& direction_vector,
                                                       double min_distance_to_obstacle,
                                                       double max_threshold);

    std::vector<double> compute_repulsive_force(const std::vector<std::vector<double>>& robot_points);
    std::pair<double, std::vector<double>> compute_distance_sdf(const std::vector<std::vector<double>>& robot_points);

    std::vector<double> compute_repulsive_force_joints(const std::vector<double>& point,
                                                       const MeshHandle& obstacle_mesh,
                                                       double max_threshold=-1.0);

    // the per-joint and per-waypoint workers
    std::vector<double> thread_joint(const std::string& joint_name,
                                     const std::vector<double>& prev_joint_positions,
                                     const std::vector<double>& joint_positions,
                                     const std::vector<double>& next_joint_positions,
                                     const MeshHandle& robot_mesh, int i);

    Row thread_waypoint(int i);

    static double wrap_angle(double a);
    static Row bound_height(const Row& position);
};

} // namespace ebp

#endif // ELASTIC_BAND_PLANNER_H