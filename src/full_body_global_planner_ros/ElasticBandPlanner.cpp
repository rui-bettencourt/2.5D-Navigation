#include "mesh_nav/ElasticBandPlanner.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#ifdef EBP_USE_OPENMP
    #include <omp.h>
    #include <atomic>
#endif

namespace ebp {

// Make a TLS clone of RobotKinematics per source pointer
static inline RobotKinematics& tlsRobot(const RobotKinematics* src) {
    struct Slot {
        const RobotKinematics* key = nullptr;
        std::unique_ptr<RobotKinematics> inst;
    };
    thread_local Slot slot;

    if (!src) throw std::runtime_error("tlsRobot: null RobotKinematics*");

    if (slot.key != src || !slot.inst) {
        slot.key  = src;
        slot.inst = std::make_unique<RobotKinematics>(*src);   // uses your copy-ctor
    }
    return *slot.inst;
}

// ----------------- ctor -----------------
ElasticBandPlanner::ElasticBandPlanner(double k_attraction_base,
                                       double k_repulsion_base,
                                       double k_attraction_joints,
                                       double k_repulsion_joints,
                                       double k_repulsion_robot_joints,
                                       double k_update_joints,
                                       double k_orientation,
                                       double k_orientation_from_base,
                                       double k_position_from_orientation,
                                       double k_safety_joints,
                                       double obstacle_threshold)
    : k_attraction_base(k_attraction_base), k_repulsion_base(k_repulsion_base),
      k_repulsion_joints(k_repulsion_joints), k_repulsion_robot_joints(k_repulsion_robot_joints),
      k_attraction_joints(k_attraction_joints), k_update_joints(k_update_joints),
      k_orientation(k_orientation), k_orientation_from_base(k_orientation_from_base),
      k_position_from_orientation(k_position_from_orientation),
      k_safety_joints(k_safety_joints), obstacle_threshold(obstacle_threshold)
{

}

// ----------------- environment init -----------------
bool ElasticBandPlanner::initialize(const std::string& sdf_bin_path, const std::string& robot_sdf_bin_path) {
    // Load environment SDF
    voxel_grid_ = std::make_unique<VoxelGrid>();
    if (!voxel_grid_->loadFromFile(sdf_bin_path)) {
        std::cerr << "Failed to load SDF from " << sdf_bin_path << std::endl;
        voxel_grid_.reset();
        return false;
    }

    //Load robot SDF
    voxel_grid_robot_ = std::make_unique<VoxelGrid>();
    if (!voxel_grid_robot_->loadFromFile(robot_sdf_bin_path)) {
        std::cerr << "Failed to load robot SDF from " << robot_sdf_bin_path << std::endl;
        voxel_grid_robot_.reset();
        return false;
    }

    // A very light body approximation (base frame)
    if (robot_points_base_.empty()) {
        robot_points_base_.clear();
        robot_points_base_.push_back(Vec3(0.0, 0.0, 0.0)); // TODO
        robot_points_base_.push_back(Vec3(0.0, 0.0, 0.5));
        robot_points_base_.push_back(Vec3(0.0, 0.0, 0.8));
    }

    attachRobot(robot_urdf_path_, joint_names_);

    std::cout << "ElasticBandPlanner initialized with SDF, Kinematics and robot body points.\n";
    return true;
}

// ----------------- robot attach -----------------
bool ElasticBandPlanner::attachRobot(const std::string& urdf_path,
                                     const std::vector<std::string>& joint_names)
{
    try {
        robot_ = std::make_unique<RobotKinematics>(urdf_path, joint_names, joints_limits_path_);
        joint_names_ = joint_names;
        dof_ = static_cast<int>(robot_->getDOF());
        if (dof_ != static_cast<int>(joint_names_.size())) {
            std::cerr << "[ElasticBandPlanner] DOF (" << dof_ << ") != joint_names size ("
                      << joint_names_.size() << ")\n";
        }
        std::cout << "[ElasticBandPlanner] RobotKinematics created at " << robot_.get()
          << " with DOF " << dof_ << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "attachRobot error: " << e.what() << std::endl;
        robot_.reset();
        dof_ = 0;
        return false;
    }
}

// ----------------- math helpers -----------------
Eigen::Matrix3d ElasticBandPlanner::rpyToRot(double r, double p, double y) {
    Eigen::AngleAxisd Rx(r, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(p, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(y, Eigen::Vector3d::UnitZ());
    // Z * Y * X (roll around x, then pitch around y, then yaw around z)
    Eigen::Quaterniond q = Rz * Ry * Rx;
    return q.toRotationMatrix();
}

std::vector<Vec3> ElasticBandPlanner::transformBodyPoints(const Vec3& base_xyz,
                                                          const Eigen::Matrix3d& Rwb,
                                                          const std::vector<Vec3>& pts_base)
{
    std::vector<Vec3> out;
    out.reserve(pts_base.size());
    for (const auto& p_b : pts_base)
        out.emplace_back(Rwb * p_b + base_xyz);
    return out;
}

double ElasticBandPlanner::wrap_angle(double a) {
    while (a >  M_PI) a -= 2.0*M_PI;
    while (a <= -M_PI) a += 2.0*M_PI;
    return a;
}

Vec3 ElasticBandPlanner::bound_height(const Vec3& p) {
    Vec3 r = p;
    if (std::isfinite(r.z())) r.z() = 0.0;
    return r;
}

// ----------------- forces -----------------
Vec3 ElasticBandPlanner::compute_attractive_force(const Vec3& prev_pos,
                                                  const Vec3& current_pos,
                                                  const Vec3& next_pos,
                                                  double clip_len)
{
    const Vec3 dir_prev = prev_pos - current_pos;
    const Vec3 dir_next = next_pos - current_pos;

    const double mp = dir_prev.norm();
    const double mn = dir_next.norm();

    const Vec3 vprev = (mp > clip_len && mp > 1e-12) ? (dir_prev / mp) * clip_len : dir_prev;
    const Vec3 vnext = (mn > clip_len && mn > 1e-12) ? (dir_next / mn) * clip_len : dir_next;

    return vprev + vnext;
}

Vec3 ElasticBandPlanner::compute_force_from_dist(double min_distance,
                                                 const Vec3& direction_vector,
                                                 double min_distance_to_obstacle,
                                                 double max_threshold)
{
    if (min_distance < max_threshold) {
        const double denom = std::max(min_distance, min_distance_to_obstacle);
        const double gain  = 1.0 / (denom * denom);
        return gain * direction_vector;
    }
    return Vec3::Zero();
}

std::pair<double, Vec3> ElasticBandPlanner::compute_distance_sdf_on(const VoxelGrid* grid, const std::vector<Vec3>& robot_points_world) const
{
  if (!grid || robot_points_world.empty()) {
    return {std::numeric_limits<double>::infinity(), Vec3(0,0,0)};
  }

  double best_d = std::numeric_limits<double>::infinity();
  Vec3   best_dir(0,0,0);

  for (const auto& p : robot_points_world) {
    Eigen::Vector3f pf = p.cast<float>();
    float dist = grid->getDistanceAtPoint(pf);
    if (dist < best_d) { 
        best_d = dist;
        best_dir = (grid->getVectorAtPoint(pf)).cast<double>();
    }
  }

  return {best_d, best_dir};
}

Vec3 ElasticBandPlanner::compute_repulsive_force(const std::vector<Vec3>& robot_points_world, const VoxelGrid* grid)
{
    double d_env   = std::numeric_limits<double>::infinity();
    Vec3   dir_env(0,0,0);

    {
        std::lock_guard<std::mutex> lk(sdf_mutex_);
        std::tie(d_env, dir_env) = compute_distance_sdf_on(grid, robot_points_world);
    }
    return compute_force_from_dist(d_env, dir_env, min_distance_to_obstacle, obstacle_threshold);
}


Vec3 ElasticBandPlanner::compute_repulsive_force_both_sdf(const std::vector<Vec3>& robot_points_world)
{
    double d_env   = std::numeric_limits<double>::infinity();
    double d_robot = std::numeric_limits<double>::infinity();
    Vec3   dir_env(0,0,0), dir_robot(0,0,0);

    {
        std::lock_guard<std::mutex> lk(sdf_mutex_);
        if (voxel_grid_) {
            std::tie(d_env, dir_env) = compute_distance_sdf_on(voxel_grid_.get(), robot_points_world);
        }
        if (voxel_grid_robot_) {
            std::tie(d_robot, dir_robot) = compute_distance_sdf_on(voxel_grid_robot_.get(), robot_points_world);
        }
    }

    // Pick the *closest* source and its direction
    double d_min = d_env;
    Vec3   dir_min = dir_env;
    if (d_robot < d_min) {
        d_min  = d_robot;
        dir_min = dir_robot;
    }

    if (!std::isfinite(d_min)) {
        return Vec3(0,0,0);
    }

    return compute_force_from_dist(d_min, dir_min,
                                    min_distance_to_obstacle,
                                    obstacle_threshold);
}

Eigen::VectorXd ElasticBandPlanner::thread_joint(int joint_index,
                                                 const Eigen::Vector3d prev_joint_pos,
                                                 const Eigen::Vector3d joint_pos,
                                                 const Eigen::Vector3d next_joint_pos,
                                                 const Eigen::VectorXd& q_cur,
                                                 RobotKinematics& rk,
                                                 int wp_idx,
                                                 int path_len)
{
    Eigen::VectorXd torques = Eigen::VectorXd::Zero(dof_);
    if (!robot_) return torques;

    // Attractive force (world)
    Eigen::Vector3d f_attr_world = compute_attractive_force(prev_joint_pos, joint_pos, next_joint_pos,
                                                            clip_length_attractive_forces_);

    // Jacobian for this joint (6 x joint_index)
    Eigen::MatrixXd J = rk.computeJacobian(q_cur, joint_index);

    // Tau from attractive force
    Eigen::VectorXd tau_attr = rk.calculateJointTorques(f_attr_world, joint_index, q_cur, &J);

    if(repulsive_joint_indices_set_.count(joint_index) > 0){   //only compute for joint 4 and 7
        // Repulsive force (world) using the joint position as the probe
        Eigen::Vector3d f_rep_world = compute_repulsive_force(std::vector<Vec3>{joint_pos}, voxel_grid_.get());
        Eigen::Vector3d f_rep_robot = compute_repulsive_force(std::vector<Vec3>{joint_pos}, voxel_grid_robot_.get());

        // Tau from repulsive force
        Eigen::VectorXd tau_rep = rk.calculateJointTorques(f_rep_world, joint_index, q_cur, &J);

        if(k_repulsion_robot_joints>0.0){
            Eigen::VectorXd tau_rep_robot = rk.calculateJointTorques(f_rep_robot, joint_index, q_cur, &J);

            torques += (k_repulsion_robot_joints * tau_rep_robot);
        }
        torques += (k_repulsion_joints  * tau_rep);
    }

    // Combine (attractive + repulsive)
    torques += (k_attraction_joints * tau_attr);

    // --- Safety (attract joint toward safe configuration) ---
    if (!safe_config_.empty()) {
        const int j = std::clamp(joint_index - 1, 0, dof_ - 1);
        if (j < static_cast<int>(safe_config_.size())) {
            const double target = safe_config_[j];
            if (std::isfinite(target)) {
                const double delta = target - q_cur(j);

                double k_safety = k_safety_joints; // constant by default
                if (dynamic_safety_) {
                    const double L      = static_cast<double>(path_len);
                    const double center = center_activation_safety_ * L;
                    // Same shape you used in Python:
                    // k = k_attraction_joints * (1 - tanh(i - center))/2
                    k_safety = k_attraction_joints *
                               (1.0 - std::tanh(static_cast<double>(wp_idx) - center)) * 0.5;
                }

                torques(j) += k_safety * delta;
            }
        }
    }

    return torques;
}

// ----------------- per-waypoint worker -----------------
Row ElasticBandPlanner::thread_waypoint(int i, RobotKinematics& rk)
{
    // Current, previous, next base states
    const Vec3 cur_xyz  ( path_matrix(i,0), path_matrix(i,1), path_matrix(i,2) );
    const Vec3 prev_xyz ( path_matrix(i-1,0), path_matrix(i-1,1), path_matrix(i-1,2) );
    const Vec3 next_xyz ( path_matrix(i+1,0), path_matrix(i+1,1), path_matrix(i+1,2) );

    const double cur_r = path_matrix(i,3), cur_p = path_matrix(i,4), cur_y = path_matrix(i,5);
    const double prv_r = path_matrix(i-1,3), prv_p = path_matrix(i-1,4), prv_y = path_matrix(i-1,5);
    const double nxt_r = path_matrix(i+1,3), nxt_p = path_matrix(i+1,4), nxt_y = path_matrix(i+1,5);

    // Prepare output row
    Row new_wp = path_matrix.row(i).transpose();

    // Set base pose for this waypoint in this thread-local kinematics
    rk.setRobotPose(cur_xyz.x(), cur_xyz.y(), cur_xyz.z(), cur_r, cur_p, cur_y);

    // ---------- Joint update ----------
    if (robot_ && dof_ > 0) {
        const Eigen::VectorXd q_prev = rowSegmentVec(i-1, 6, dof_);
        const Eigen::VectorXd q_cur  = rowSegmentVec(i,   6, dof_);
        const Eigen::VectorXd q_next = rowSegmentVec(i+1, 6, dof_);

        // Joint positions for prev/cur/next in world
        auto prev_positions = rk.getJointPositions(q_prev);
        auto cur_positions  = rk.getJointPositions(q_cur);
        auto next_positions = rk.getJointPositions(q_next);

        // Compute per-joint torques in parallel (write to separate slots, no races)
        // std::vector<Eigen::VectorXd> tau_list(dof_);
        // #ifdef EBP_USE_OPENMP
        // #pragma omp parallel for schedule(static) if(!omp_in_parallel())
        // #endif
        // for (int j = 0; j < dof_; ++j) {
        //     tau_list[j] = thread_joint(j+1,
        //                                prev_positions[j],
        //                                cur_positions[j],
        //                                next_positions[j],
        //                                q_cur,
        //                                rk,
        //                                /*wp_idx*/ i,
        //                                /*path_len*/ static_cast<int>(path_matrix.rows()));
        // }

        // // Sum per-joint torques
        // Eigen::VectorXd total_torques = Eigen::VectorXd::Zero(dof_);
        // for (int j = 0; j < dof_; ++j) total_torques += tau_list[j];
        // --- Keep inner loop SERIAL ---
        Eigen::VectorXd total_torques = Eigen::VectorXd::Zero(dof_);
        for (int j = 0; j < dof_; ++j) {
            auto tau = thread_joint(j+1, prev_positions[j], cur_positions[j], next_positions[j], q_cur, rk, i, static_cast<int>(path_matrix.rows()));
            total_torques += tau;
        }

        // Update joints and clamp
        Eigen::VectorXd q_new = rk.clampToLimits(q_cur + k_update_joints * total_torques);

        // Write back joints
        new_wp.segment(6, dof_) = q_new;
    }

    // ---------- Base update ----------
    // Attractive force on base (world frame)
    Vec3 f_attr = compute_attractive_force(prev_xyz, cur_xyz, next_xyz, clip_length_attractive_forces_);

    // Repulsive force from body points (world frame)
    const Eigen::Matrix3d Rwb = rpyToRot(cur_r, cur_p, cur_y);
    const auto body_world_pts = transformBodyPoints(cur_xyz, Rwb, robot_points_base_);
    Vec3 f_rep = compute_repulsive_force(body_world_pts, voxel_grid_.get());

    // Combine
    Vec3 f_total_world = k_attraction_base * f_attr + k_repulsion_base * f_rep;

    // Move base in world, keep z = 0
    Vec3 new_xyz = bound_height(cur_xyz + f_total_world);

    // Heading from force (in base frame)
    Vec3 f_total_local = Rwb.transpose() * f_total_world;
    double base_yaw_from_force = std::atan2(f_total_local.y(), f_total_local.x());

    // Smooth orientation
    double corr_r = wrap_angle(prv_r - cur_r) + wrap_angle(nxt_r - cur_r);
    double corr_p = wrap_angle(prv_p - cur_p) + wrap_angle(nxt_p - cur_p);
    double corr_y = wrap_angle(prv_y - cur_y) + wrap_angle(nxt_y - cur_y);

    double new_r = wrap_angle(cur_r + k_orientation * corr_r);
    double new_p = wrap_angle(cur_p + k_orientation * corr_p);
    double new_y = wrap_angle(cur_y + k_orientation * corr_y
                              + k_orientation_from_base * base_yaw_from_force);

    // Write back base
    new_wp(0) = new_xyz.x();
    new_wp(1) = new_xyz.y();
    new_wp(2) = new_xyz.z();
    new_wp(3) = new_r;
    new_wp(4) = new_p;
    new_wp(5) = new_y;

    return new_wp;
}

// ----------------- solver loop -----------------
PathMatrix ElasticBandPlanner::update_path(const PathMatrix& path,
                                           int iterations, double convergence_threshold)
{
    path_matrix = path;
    const int N = static_cast<int>(path_matrix.rows());
    if (N < 3) return path_matrix;
    if (!robot_) throw std::runtime_error("Robot not attached before update_path!");

    if (hasRobot() && path_matrix.cols() != (6 + dof_)) {
        std::cerr << "[ElasticBandPlanner] Path columns (" << path_matrix.cols()
                  << ") != 6 + dof (" << (6 + dof_) << ").\n";
    }

    // Avoid Eigen spawning its own threads inside the OMP region (oversubscription)
    #ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
    #endif
    Eigen::setNbThreads(1);

    for (int it = 0; it < iterations; ++it) {
        double max_change = 0.0;

        std::vector<Row> new_rows;
        new_rows.resize(std::max(0, N - 2));

        // Parallelize across waypoints. Each thread must fetch its **own** TLS RobotKinematics.
        // Also, do not write any shared state from inside the loop.
        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= N - 2; ++i) {
            try {
                // IMPORTANT: get thread-local RK *inside* the parallel region
                RobotKinematics& rk_tls = tlsRobot(robot_.get());
                new_rows[i - 1] = thread_waypoint(i, rk_tls);
            } catch (const std::exception& e) {
                // Keep exceptions local to the iteration; log and produce a no-op row
                #pragma omp critical
                {
                    std::cerr << "[ElasticBandPlanner] thread_waypoint(" << i
                              << ") threw: " << e.what() << std::endl;
                }
                new_rows[i - 1] = path_matrix.row(i).transpose(); // fallback: keep old row
            } catch (...) {
                #pragma omp critical
                {
                    std::cerr << "[ElasticBandPlanner] thread_waypoint(" << i
                              << ") threw unknown exception\n";
                }
                new_rows[i - 1] = path_matrix.row(i).transpose();
            }
        }

        // Apply updates (serial), measure convergence
        for (int idx = 0; idx < static_cast<int>(new_rows.size()); ++idx) {
            const int i = 1 + idx;
            Row oldv = path_matrix.row(i).transpose();
            Row newv = new_rows[idx];

            const double change = (newv - oldv).norm();
            if (change > max_change) max_change = change;

            path_matrix.row(i) = newv.transpose();
        }

        history.push_back(path_matrix);
        if (max_change < convergence_threshold) break;
    }
    return path_matrix;
}

} // namespace ebp
