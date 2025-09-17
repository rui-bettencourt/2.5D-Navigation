#ifndef ELASTIC_BAND_PLANNER_H
#define ELASTIC_BAND_PLANNER_H

#include <string>
#include <memory>
#include <any>
#include <vector>
#include <Eigen/Dense>
#include <mutex>
#include <unordered_set>


#include "mesh_nav/VoxelSDF.h"
#include "mesh_nav/RobotKinematics.h"

namespace ebp {

using Vec3       = Eigen::Vector3d;
using Row        = Eigen::VectorXd;   // one waypoint: [x y z roll pitch yaw q1 ... qN]
using PathMatrix = Eigen::MatrixXd;   // waypoints by rows

class ElasticBandPlanner {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ElasticBandPlanner(double k_attraction_base=0.5,
                       double k_repulsion_base=0.0,
                       double k_attraction_joints=0.0,
                       double k_repulsion_joints=0.0,
                       double k_repulsion_robot_joints=0.0,
                       double k_update_joints=0.0,
                       double k_orientation=0.0,
                       double k_orientation_from_base=0.0,
                       double k_position_from_orientation=0.0,
                       double k_safety_joints=0.00,
                       double obstacle_threshold=1.5);

    // Load SDF and default body points
    bool initialize(const std::string& sdf_bin_path, const std::string& robot_sdf_bin_path);

    // Create/own the RobotKinematics instance
    bool attachRobot(const std::string& urdf_path,
                     const std::vector<std::string>& joint_names);
    void setRobotUrdfPath(const std::string& path) { robot_urdf_path_ = path; }
    void setJointNames(const std::vector<std::string>& names, const std::string& joint_limits_path) { 
        joint_names_ = names; 
        joints_limits_path_ = joint_limits_path;
    }
    void setRepulsiveJointIndices(const std::vector<int>& indices) {
        repulsive_joint_indices_ = indices;
        repulsive_joint_indices_set_.clear();
        repulsive_joint_indices_set_.reserve(repulsive_joint_indices_.size());
        for (int i : repulsive_joint_indices_) repulsive_joint_indices_set_.insert(i);
    }
    void setRobotBasePoints(const std::vector<Vec3>& pts) { robot_points_base_ = pts; }

    inline bool hasRobot() const { return static_cast<bool>(robot_); }
    inline int  dof() const      { return dof_; }

    // Main solver
    PathMatrix update_path(const PathMatrix& path,
                           int iterations=100, double convergence_threshold=1e-3);

    void set_dynamic_safety(bool on) { dynamic_safety = on; }
    void setSafeConfig(const std::vector<double>& qsafe) { safe_config_ = qsafe; }
    void setDynamicSafety(bool enable, double center=0.8) {
    dynamic_safety_ = enable; center_activation_safety_ = center;
    }

private:
    // ---------- Robot and environment ----------
    std::string robot_urdf_path_;
    std::string joints_limits_path_;
    std::unique_ptr<VoxelGrid> voxel_grid_;
    std::unique_ptr<VoxelGrid> voxel_grid_robot_;
    std::unique_ptr<RobotKinematics> robot_;
    std::vector<std::string> joint_names_;
    int dof_ = 0;
    bool debug_ = true;
    double min_distance_to_obstacle = 0.05;
    double clip_length_attractive_forces_ = 200.0;
    std::vector<double> safe_config_;        // size = dof_, NaN/empty => ignore
    bool   dynamic_safety_ = false;
    double center_activation_safety_ = 0.8;  // same meaning as python
    int    cur_wp_idx_ = 0;                  // current waypoint index
    int    path_len_   = 1;                  // total waypoints
    std::vector<int> repulsive_joint_indices_{ /* default */ 3, 6 }; // 0-based: joints 4 and 7
    std::unordered_set<int> repulsive_joint_indices_set_{3, 6};      // fast membership checks

    // ---------- Gains / params ----------
    double k_attraction_base, k_repulsion_base, k_repulsion_joints, k_repulsion_robot_joints;
    double k_attraction_joints, k_update_joints, k_orientation, k_orientation_from_base, k_position_from_orientation;
    double k_safety_joints;
    double obstacle_threshold;
    double weak_torque_threshold;
    bool   dynamic_safety;
    double center_activation_safety;
    double radius_joint;
    int    number_points_robot;

    std::mutex sdf_mutex_;

    // A few points to approximate robot body for SDF forces (in base frame)
    std::vector<Vec3> robot_points_base_;

    // ---------- Internal state ----------
    PathMatrix path_matrix;
    std::vector<PathMatrix> history;

    // ---------- Helpers ----------
    static Vec3 compute_attractive_force(const Vec3& prev_pos,
                                         const Vec3& current_pos,
                                         const Vec3& next_pos,
                                         double clip_len);

    static Vec3 compute_force_from_dist(double min_distance,
                                        const Vec3& direction_vector,
                                        double min_distance_to_obstacle,
                                        double max_threshold);

    Vec3 compute_repulsive_force(const std::vector<Vec3>& robot_points_world, const VoxelGrid* grid);
    Vec3 compute_repulsive_force_both_sdf(const std::vector<Vec3>& robot_points_world);

     // Query min distance + direction against a specific SDF grid
    std::pair<double, Vec3> compute_distance_sdf_on(const VoxelGrid* grid, const std::vector<Vec3>& robot_points_world) const;


    // Per-joint worker: returns full torque vector (size = dof_)
    // Eigen::VectorXd thread_joint(int joint_index,
    //                              const Vec3 prev_joint_pos,
    //                              const Vec3 joint_pos,
    //                              const Vec3 next_joint_pos,
    //                              const Eigen::VectorXd& q_cur);
    Eigen::VectorXd thread_joint(int joint_index,
                                const Eigen::Vector3d prev_joint_pos,
                                const Eigen::Vector3d joint_pos,
                                const Eigen::Vector3d next_joint_pos,
                                const Eigen::VectorXd& q_cur,
                                RobotKinematics& rk,
                                int wp_idx,
                                int path_len);

    // Per-waypoint update
    Row thread_waypoint(int i, RobotKinematics& rk);

    static double wrap_angle(double a);
    static Vec3   bound_height(const Vec3& p);

    static Eigen::Matrix3d rpyToRot(double r, double p, double y);
    static std::vector<Vec3> transformBodyPoints(const Vec3& base_xyz,
                                                 const Eigen::Matrix3d& Rwb,
                                                 const std::vector<Vec3>& pts_base);

    // Convenience: get a row segment as a column vector
    inline Eigen::VectorXd rowSegmentVec(int row, int start, int len) const {
        return path_matrix.block(row, start, 1, len).transpose();
    }
};

} // namespace ebp

#endif // ELASTIC_BAND_PLANNER_H
