#include "mesh_nav/ElasticBandPlanner.h"

#include <cmath>
#include <algorithm>
#include <future>
#include <numeric>
#include <limits>

namespace ebp {

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
      k_orientation(k_orientation), k_orientation_from_base(k_orientation_from_base), k_position_from_orientation(k_position_from_orientation),
      k_safety_joints(k_safety_joints), obstacle_threshold(obstacle_threshold)
{
    weak_torque_threshold = 0.1;
    min_distance_to_obstacle = 0.1;
    dynamic_safety = true;
    center_activation_safety = 0.8;
    radius_joint = 0.01;
    number_points_robot = 200;
}

// Small vector helpers
static inline Row vec_add(const Row& a, const Row& b){ Row r(a.size()); for(size_t i=0;i<a.size();++i) r[i]=a[i]+b[i]; return r; }
static inline Row vec_sub(const Row& a, const Row& b){ Row r(a.size()); for(size_t i=0;i<a.size();++i) r[i]=a[i]-b[i]; return r; }
static inline Row vec_scale(const Row& a, double s){ Row r(a.size()); for(size_t i=0;i<a.size();++i) r[i]=a[i]*s; return r; }
static inline double vec_norm(const Row& a){ double s=0.0; for(double v: a) s+=v*v; return std::sqrt(s); }

bool ElasticBandPlanner::initialize(const std::string& sdf_bin_path) {
    voxel_grid_ = std::make_unique<VoxelGrid>();
    if (!voxel_grid_->loadFromFile(sdf_bin_path)) {
        std::cerr << "Failed to load SDF from " << sdf_bin_path << std::endl;
        voxel_grid_.reset();
        return false;
    }

    // Define robot representative points (example: two points)
    robot_points_.clear();
    robot_points_.push_back({0.0, 0.0, 0.0});
    robot_points_.push_back({0.0, 0.0, 1.0});

    std::cout << "ElasticBandPlanner initialized with SDF and robot points.\n";
    return true;
}

std::vector<double> ElasticBandPlanner::compute_attractive_force(const std::vector<double>& prev_pos, const std::vector<double>& current_pos, const std::vector<double>& next_pos, double obstacle_threshold){
    Row cur = current_pos; Row next = next_pos; Row prev = prev_pos;
    Row dir_prev = vec_sub(prev, cur); Row dir_next = vec_sub(next, cur);
    double mag_prev = vec_norm(dir_prev); double mag_next = vec_norm(dir_next);
    if(mag_prev>0.0){ for(auto &v: dir_prev) v /= mag_prev; }
    if(mag_next>0.0){ for(auto &v: dir_next) v /= mag_next; }
    Row out(3,0.0);
    if(mag_prev < obstacle_threshold && mag_next < obstacle_threshold){
        out = vec_add(vec_sub(prev,cur), vec_sub(next,cur));
    } else if(mag_prev >= obstacle_threshold && mag_next < obstacle_threshold){
        out = vec_add(vec_scale(dir_prev, obstacle_threshold), vec_sub(next,cur));
    } else if(mag_next >= obstacle_threshold){
        out = vec_add(vec_sub(prev,cur), vec_scale(dir_next, obstacle_threshold));
    } else {
        out = vec_add(vec_scale(dir_prev, obstacle_threshold), vec_scale(dir_next, obstacle_threshold));
    }
    return out;
}

std::vector<double> ElasticBandPlanner::compute_force_from_dist(double min_distance, const std::vector<double>& direction_vector, double min_distance_to_obstacle, double max_threshold){
    if(max_threshold < 0) max_threshold = obstacle_threshold;
    if(min_distance < max_threshold){
        double denom = std::max(min_distance, min_distance_to_obstacle);
        double force_magnitude = 1.0 / (denom*denom);
        Row dir = direction_vector;
        for(auto &v: dir) v *= force_magnitude;
        return dir;
    } else {
        return Row{0.0,0.0,0.0};
    }
}
// Compute min distance and vector over a set of robot points
std::pair<double, std::vector<double>> ElasticBandPlanner::compute_distance_sdf(
    const std::vector<std::vector<double>>& robot_points)
{
    if (!voxel_grid_) {
        throw std::runtime_error("VoxelGrid not initialized!");
    }

    double min_dist = std::numeric_limits<double>::max();
    std::vector<double> best_vec(3, 0.0);

    for (const auto& p : robot_points) {
        Eigen::Vector3f pt(p[0], p[1], p[2]);

        float dist = voxel_grid_->getDistanceAtPoint(pt);
        Eigen::Vector3f vec = voxel_grid_->getVectorAtPoint(pt);

        if (dist < min_dist) {
            min_dist = dist;
            best_vec = {static_cast<double>(vec[0]),
                        static_cast<double>(vec[1]),
                        static_cast<double>(vec[2])};
        }
    }

    return {min_dist, best_vec};
}

std::vector<double> ElasticBandPlanner::compute_repulsive_force(
    const std::vector<std::vector<double>>& robot_points)
{
    auto [min_distance, direction] = compute_distance_sdf(robot_points);
    return compute_force_from_dist(min_distance, direction,
                                   min_distance_to_obstacle, obstacle_threshold);
}

std::vector<double> ElasticBandPlanner::compute_repulsive_force_joints(const std::vector<double>& point, const MeshHandle& obstacle_mesh, double /*max_threshold*/){
    (void)point; (void)obstacle_mesh;
    // TODO: create a small sphere around 'point' and compute distance to obstacle_mesh
    return std::vector<double>{0.0,0.0,0.0};
}

std::vector<double> ElasticBandPlanner::thread_joint(const std::string& joint_name, const std::vector<double>& prev_joint_positions, const std::vector<double>& joint_positions, const std::vector<double>& next_joint_positions, const MeshHandle& /*robot_mesh*/, int i){
    int joint_id = std::stoi(joint_name.substr(1));
    int joint_index = joint_id + 5;
    int dof = robot->get_dof();
    std::vector<double> joints_torques(dof, 0.0);

    // Attractive joint force
    auto attractive_joint_force = compute_attractive_force(prev_joint_positions, joint_positions, next_joint_positions, obstacle_threshold);
    auto local_attractive_joint_force = robot->world_to_local(attractive_joint_force, true);
    std::vector<double> joint_angles(path_matrix[i].begin()+6, path_matrix[i].end());
    auto [attractive_torques, J_joint] = robot->calculate_joint_torques(local_attractive_joint_force, joint_id, joint_angles);
    for(size_t k=0;k<joints_torques.size() && k<attractive_torques.size(); ++k) joints_torques[k] += k_attraction_joints * attractive_torques[k];

    // Repulsive joint force
    std::vector<double> repulsive_joint_force = compute_repulsive_force_joints(joint_positions, obs_mesh_handle);
    auto local_repulsive_joint_force = robot->world_to_local(repulsive_joint_force, true);
    auto [repulsive_torques, _] = robot->calculate_joint_torques(local_repulsive_joint_force, joint_id, joint_angles, &J_joint);
    for(size_t k=0;k<joints_torques.size() && k<repulsive_torques.size(); ++k) joints_torques[k] += k_repulsion_joints * repulsive_torques[k];

    return joints_torques;
}

Row ElasticBandPlanner::thread_waypoint(int i){
    Row new_waypoint = path_matrix[i];
    Row current_pos(path_matrix[i].begin(), path_matrix[i].begin()+6);
    Row next_pos(path_matrix[i+1].begin(), path_matrix[i+1].begin()+6);
    Row prev_pos(path_matrix[i-1].begin(), path_matrix[i-1].begin()+6);

    // TODO: Here
    robot->set_robot_pose(current_pos[0], current_pos[1], current_pos[2], current_pos[3], current_pos[4], current_pos[5]);

    if(robot->get_dof() > 0){
        auto prev_joint_positions_map = robot->get_arm_endpoints(false, current_pos, std::vector<double>(path_matrix[i-1].begin()+6, path_matrix[i-1].end()));
        auto next_joint_positions_map = robot->get_arm_endpoints(false, current_pos, std::vector<double>(path_matrix[i+1].begin()+6, path_matrix[i+1].end()));
        auto joint_positions_map = robot->get_arm_endpoints(false, current_pos, std::vector<double>(path_matrix[i].begin()+6, path_matrix[i].end()));

        std::vector<std::string> joint_keys;
        for(auto &kv: joint_positions_map) joint_keys.push_back(kv.first);
        std::sort(joint_keys.begin(), joint_keys.end());

        std::vector<std::future<std::vector<double>>> futures;
        futures.reserve(joint_keys.size());
        for(const auto &jk: joint_keys){
            futures.emplace_back(std::async(std::launch::async, &ElasticBandPlanner::thread_joint, this, jk, prev_joint_positions_map[jk], joint_positions_map[jk], next_joint_positions_map[jk], MeshHandle(), i));
        }
        int dof = robot->get_dof();
        std::vector<double> joints_torques(dof,0.0);
        for(auto &f: futures){ auto t = f.get(); for(int k=0;k<dof;++k) joints_torques[k]+=t[k]; }

        // update joint configs
        std::vector<double> new_joints_config(dof);
        for(int k=0;k<dof;++k){ new_joints_config[k] = path_matrix[i][6+k] + k_update_joints * joints_torques[k]; }
        auto limits = robot->get_joints_limits();
        for(int k=0;k<dof;++k){ std::string qname = std::string("q") + std::to_string(k+1); if(limits.count(qname)){ auto pr = limits.at(qname); if(new_joints_config[k] < pr.first) new_joints_config[k] = pr.first; if(new_joints_config[k] > pr.second) new_joints_config[k] = pr.second; } }
        for(int k=0;k<dof;++k) new_waypoint[6+k] = new_joints_config[k];
    }

    // Body-level computations
    MeshHandle global_bbs = robot_model->simulate_move_joints(MeshHandle(), std::string("base_link"), current_pos);
    MeshHandle robot_mesh = robot_model->convert_bbs_to_mesh(global_bbs);

    auto attractive_force = compute_attractive_force(prev_pos, current_pos, next_pos, obstacle_threshold);
    auto repulsive_force_world = compute_repulsive_force(robot_mesh);
    auto repulsive_force = robot->world_to_local(repulsive_force_world, true);
    Row total_force = vec_add(vec_scale(attractive_force, k_attraction_base), vec_scale(repulsive_force, k_repulsion_base));

    Row new_pos = bound_height(vec_add(Row{current_pos[0],current_pos[1],current_pos[2]}, total_force));

    double base_torque_from_total_force = std::atan2(total_force[1], total_force[0]);
    double orientation_correction_roll = wrap_angle(path_matrix[i-1][3] - path_matrix[i][3]) + wrap_angle(path_matrix[i+1][3] - path_matrix[i][3]);
    double orientation_correction_pitch = wrap_angle(path_matrix[i-1][4] - path_matrix[i][4]) + wrap_angle(path_matrix[i+1][4] - path_matrix[i][4]);
    double orientation_correction_yaw = wrap_angle(path_matrix[i-1][5] - path_matrix[i][5]) + wrap_angle(path_matrix[i+1][5] - path_matrix[i][5]);

    double new_roll = wrap_angle(current_pos[3] + k_orientation * orientation_correction_roll);
    double new_pitch = wrap_angle(current_pos[4] + k_orientation * orientation_correction_pitch);
    double new_theta = wrap_angle(current_pos[5] + k_orientation * orientation_correction_yaw + k_orientation_from_base * base_torque_from_total_force);

    new_waypoint[0]=new_pos[0]; new_waypoint[1]=new_pos[1]; new_waypoint[2]=new_pos[2];
    new_waypoint[3]=new_roll; new_waypoint[4]=new_pitch; new_waypoint[5]=new_theta;

    return new_waypoint;
}

Row ElasticBandPlanner::bound_height(const Row& position){ Row p = position; if(p.size()>2) p[2]=0.0; return p; }

double ElasticBandPlanner::wrap_angle(double a){ while(a>M_PI) a-=2.0*M_PI; while(a<=-M_PI) a+=2.0*M_PI; return a; }


ElasticBandPlanner::PathMatrix ElasticBandPlanner::update_path(const PathMatrix& path, int iterations, double convergence_threshold){
    path_matrix = path;
    int N = static_cast<int>(path_matrix.size());
    if(N < 3) return path_matrix;

    for(int iteration=0; iteration<iterations; ++iteration){
        double max_change = 0.0;
        std::vector<std::future<Row>> futures;
        futures.reserve(N-2);
        for(int i=1;i<=N-2;++i) futures.emplace_back(std::async(std::launch::async, &ElasticBandPlanner::thread_waypoint, this, i));
        std::vector<Row> new_points; new_points.reserve(futures.size());
        for(auto &f: futures) new_points.push_back(f.get());
        for(size_t idx=0; idx<new_points.size(); ++idx){
            int i = 1 + static_cast<int>(idx);
            double change = 0.0;
            const Row &oldv = path_matrix[i];
            const Row &newv = new_points[idx];
            for(size_t k=0;k<oldv.size();++k){ double d = newv[k]-oldv[k]; change += d*d; }
            change = std::sqrt(change);
            if(change>max_change) max_change=change;
            path_matrix[i] = newv;
        }
        history.push_back(path_matrix);
        if(max_change < convergence_threshold) break;
    }
    return path_matrix;
}

} // namespace ebp