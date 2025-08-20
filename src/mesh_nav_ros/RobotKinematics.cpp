#include "mesh_nav/RobotKinematics.h"
#include <fstream>
#include <nlohmann/json.hpp>   // requires: nlohmann_json library

using json = nlohmann::json;
using namespace pinocchio;

RobotKinematics::RobotKinematics(const std::string& urdf_path, const std::string& base_link, const std::string& ee_link, const std::vector<std::string>& joint_names)
    : base_frame(base_link), end_effector(ee_link)
{
    if(use_pinocchio){
        pinocchio::urdf::buildModel(urdf_path, model);
        std::cout << "model name: " << model.name << std::endl;
        // Create data required by the algorithms
        data = pinocchio::Data(model);
        updateJointsNames(joint_names);
    }else{
        Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
        std::vector<Eigen::Isometry3d> transforms;
        createManualTransforms(transforms, q);
    }
    num_joints =joint_names.size();

    // Load joint limits from JSON file
    loadJointLimits("data/joints_limits.json", joint_names);
}

void RobotKinematics::updateJointsNames(const std::vector<std::string>& joint_names)
{
    std::cout << "Updating joints names" << std::endl;
    joint_ids.clear();
    for (const auto& name : joint_names)
    {
        pinocchio::JointIndex joint_id = model.getJointId(name);

        if (joint_id == 0 || joint_id >= model.njoints)
            throw std::invalid_argument("Invalid joint name: " + name);

        joint_ids.push_back(joint_id);
    }
    std::cout << "Finished update" << std::endl;
}

std::vector<Eigen::Vector3d> RobotKinematics::getJointPositions(
    const Eigen::VectorXd& q)
{
    if(use_pinocchio){
        return forwardKinematicsPinocchio(q);
    }else{
        return forwardKinematicsManual(q);
    }
}

std::vector<Eigen::MatrixXd> RobotKinematics::computeJacobians(const Eigen::VectorXd& q){
    std::vector<Eigen::MatrixXd> Js;

    for(int id=1; id<=num_joints; ++id){
        Js.push_back(computeJacobian(q,id));
    }

    return Js;
}

Eigen::MatrixXd RobotKinematics::computeJacobian(const Eigen::VectorXd& q, int target_joint_idx){
    if(use_pinocchio){
        return computeJacobiansPinocchio(q, target_joint_idx);
    }else{
        return computeJacobiansManual(q, target_joint_idx);
    }

}

Eigen::MatrixXd RobotKinematics::computeJacobiansPinocchio(const Eigen::VectorXd& q, int target_joint_idx)
{
    // Make sure FK and placements are up-to-date
    // pinocchio::forwardKinematics(model, data, q);
    // pinocchio::updateFramePlacements(model, data);


    Eigen::MatrixXd J(6, model.nv);
    // pinocchio::computeFrameJacobian(model, data, q, frame_id, J, pinocchio::LOCAL_WORLD_ALIGNED);
    pinocchio::computeJointJacobian(model, data, q, target_joint_idx, J);

    std::cout << "J shape: " << J.rows() << " x " << J.cols() << std::endl;
    return J;
}

size_t RobotKinematics::getDOF() const
{
    if(use_pinocchio){
        return model.nq;  // or model.nv
    }else{
        return 7;
    }
}

std::vector<Eigen::Vector3d> RobotKinematics::forwardKinematicsPinocchio(
    const Eigen::VectorXd& q)
{
    std::cout << "Starting forward kinematics" << std::endl;
    // Compute forward kinematics once
    pinocchio::forwardKinematics(model, data, q);

    std::cout << "Initializing positions vector" << std::endl;
    std::vector<Eigen::Vector3d> positions;
    positions.reserve(joint_ids.size());  // avoid reallocations

    std::cout << "Pushing results" << std::endl;
    for (auto joint_id : joint_ids)
        positions.push_back(data.oMi[joint_id].translation());

    return positions;
}


std::vector<Eigen::Vector3d> RobotKinematics::forwardKinematicsManual(const Eigen::VectorXd& q) {
    std::vector<Eigen::Isometry3d> transforms;
    createManualTransforms(transforms, q);

    std::vector<Eigen::Vector3d> positions;
    positions.reserve(transforms.size());

    Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
    for (const auto& T : transforms) {
        current = current * T;
        positions.push_back(current.translation());
    }

    return positions;
}

// Preallocated transformation matrices
void RobotKinematics::createManualTransforms(std::vector<Eigen::Isometry3d>& transforms, const Eigen::VectorXd& q) {
    transforms.clear();
    transforms.reserve(8); // 8 segments

    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();

    // 1. base_footprint -> torso_lift
    tf.translation() << -0.062, 0.0, 0.8885 + q(0);
    transforms.push_back(tf);

    // 2. q1 rotation about z (and translation)
    tf = Eigen::Isometry3d::Identity();
    double theta1 = q1z + q(1);
    tf.linear() = Eigen::AngleAxisd(theta1, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    tf.translation() << 0.15505, 0.014, -0.151;
    transforms.push_back(tf);

    // 3. q2 rotation (custom composite rotation)
    tf = Eigen::Isometry3d::Identity();
    double theta2 = q2y - q(2);
    Eigen::Matrix3d R2;
    R2 << std::cos(theta2), std::sin(theta2)*std::sin(q2x), std::sin(theta2)*std::cos(q2x),
          0,                std::cos(q2x),                 -std::sin(q2x),
         -std::sin(theta2), std::cos(theta2)*std::sin(q2x), std::cos(theta2)*std::cos(q2x);
    tf.linear() = R2;
    tf.translation() << 0.125, 0.0195, -0.031;
    transforms.push_back(tf);

    // 4. q3 rotation
    tf = Eigen::Isometry3d::Identity();
    double c3 = std::cos(q(3)), s3 = std::sin(q(3));
    Eigen::Matrix3d R3;
    R3 << std::cos(q3z)*c3, std::cos(q3z)*s3*std::sin(q3x) - std::sin(q3z)*std::cos(q3x), std::cos(q3z)*s3*std::cos(q3x) + std::sin(q3z)*std::sin(q3x),
          std::sin(q3z)*c3, std::sin(q3z)*s3*std::sin(q3x) + std::cos(q3z)*std::cos(q3x), std::sin(q3z)*s3*std::cos(q3x) - std::cos(q3z)*std::sin(q3x),
          -s3,              c3*std::sin(q3x),                                             c3*std::cos(q3x);
    tf.linear() = R3;
    tf.translation() << 0.0895, 0.0, -0.0015;
    transforms.push_back(tf);

    // 5. q4 rotation
    tf = Eigen::Isometry3d::Identity();
    double theta4 = q4y + q(4);
    Eigen::Matrix3d R4;
    R4 << std::cos(theta4), std::sin(theta4)*std::sin(q4x), std::sin(theta4)*std::cos(q4x),
          0,                std::cos(q4x),                 -std::sin(q4x),
         -std::sin(theta4), std::cos(theta4)*std::sin(q4x), std::cos(theta4)*std::cos(q4x);
    tf.linear() = R4;
    tf.translation() << -0.02, -0.027, -0.222;
    transforms.push_back(tf);

    // 6. q5 rotation
    tf = Eigen::Isometry3d::Identity();
    double theta5 = q5y + q(5);
    Eigen::Matrix3d R5;
    R5 << std::cos(q5z)*std::cos(theta5), std::cos(q5z)*std::sin(theta5)*std::sin(q5x) - std::sin(q5z)*std::cos(q5x), std::cos(q5z)*std::sin(theta5)*std::cos(q5x) + std::sin(q5z)*std::sin(q5x),
          std::sin(q5z)*std::cos(theta5), std::sin(q5z)*std::sin(theta5)*std::sin(q5x) + std::cos(q5z)*std::cos(q5x), std::sin(q5z)*std::sin(theta5)*std::cos(q5x) - std::cos(q5z)*std::sin(q5x),
         -std::sin(theta5),               std::cos(theta5)*std::sin(q5x),                                           std::cos(theta5)*std::cos(q5x);
    tf.linear() = R5;
    tf.translation() << -0.162, 0.02, 0.027;
    transforms.push_back(tf);

    // 7. q6 rotation
    tf = Eigen::Isometry3d::Identity();
    double theta6 = q6y + q(6);
    Eigen::Matrix3d R6;
    R6 << std::cos(theta6), std::sin(theta6)*std::sin(q6x), std::sin(theta6)*std::cos(q6x),
          0,                std::cos(q6x),                 -std::sin(q6x),
         -std::sin(theta6), std::cos(theta6)*std::sin(q6x), std::cos(theta6)*std::cos(q6x);
    tf.linear() = R6;
    tf.translation() << 0.0, 0.0, 0.15;
    transforms.push_back(tf);

    // 8. Final identity (tool frame)
    transforms.push_back(Eigen::Isometry3d::Identity());
}

Eigen::MatrixXd RobotKinematics::computeJacobiansManual(const Eigen::VectorXd& q, int target_joint_idx) {
    // IMPORTANT: this function computes geometric Jacobian through the forward kinematic, not using derivatives. For more accurate Jacobian i can derive the analytical jacobian.
        // Safety check
    if (target_joint_idx < 0 || target_joint_idx > num_joints) {
        throw std::invalid_argument("Invalid target_joint_idx");
    }

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, target_joint_idx);
    J.setZero();

    // Get forward transforms T_0_i for each joint i
    std::vector<Eigen::Isometry3d> T;
    createManualTransforms(T, q);

    Eigen::Vector3d p_target = T[target_joint_idx-1].translation();  // position of the target link

    for (int i = 1; i < target_joint_idx; ++i) {
        Eigen::Vector3d zi = T[i].rotation().col(2);     // z-axis of joint i
        Eigen::Vector3d pi = T[i].translation();         // position of joint i
        Eigen::Vector3d diff = p_target - pi;

        J.block<3,1>(0, i) = zi.cross(diff);             // linear velocity
        J.block<3,1>(3, i) = zi;                         // angular velocity
    }

    return J;
}

Eigen::Vector3d RobotKinematics::worldToLocal(const Eigen::Vector3d& vec, bool normalize) const
{
    Eigen::Vector3d local = base_pose.rotation().transpose() * vec;
    if (normalize && local.norm() > 1e-9)
        local.normalize();
    return local;
}

Eigen::VectorXd RobotKinematics::calculateJointTorques(const Eigen::Vector3d& force,
                                                       int joint_id,
                                                       const Eigen::VectorXd& q,
                                                       const Eigen::MatrixXd* J_ptr)
{
    // Compute Jacobian for the target joint
    Eigen::MatrixXd J;
    if (J_ptr)
        J = *J_ptr;
    else
        J = computeJacobian(q, joint_id);

    // Only use translational rows (top 3 rows of J)
    Eigen::MatrixXd Jv = J.block(0, 0, 3, J.cols());

    // tau = J^T * F
    Eigen::VectorXd tau = Jv.transpose() * force;
    return tau;
}

std::map<std::string, std::pair<double,double>> RobotKinematics::getJointsLimits() const
{
    // If we successfully loaded limits from file, return them
    if (!joint_limits.empty()) {
        return joint_limits;
    }

    std::map<std::string, std::pair<double,double>> limits;

    if (use_pinocchio) {
        for (const auto& jid : joint_ids) {
            const auto& jmodel = model.joints[jid];
            std::string name = model.names[jid];
            Eigen::VectorXd lower = model.lowerPositionLimit.segment(model.idx_qs[jid], jmodel.nq());
            Eigen::VectorXd upper = model.upperPositionLimit.segment(model.idx_qs[jid], jmodel.nq());
            limits[name] = { lower[0], upper[0] };
        }
    } else {
        // Dummy limits for manual mode (e.g. [-pi, pi])
        for (int i = 0; i < num_joints; ++i) {
            std::string name = "joint_" + std::to_string(i);
            limits[name] = { -M_PI, M_PI };
        }
    }

    return limits;
}

void RobotKinematics::setRobotPose(double x, double y, double z,
                                   double roll, double pitch, double yaw)
{
    Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());

    base_pose = Eigen::Isometry3d::Identity();
    base_pose.linear() = (Rz * Ry * Rx).toRotationMatrix();
    base_pose.translation() = Eigen::Vector3d(x, y, z);
}

// ================= Joint limits =================
void RobotKinematics::loadJointLimits(const std::string& filepath, 
                                      const std::vector<std::string>& joint_names)
{
    std::ifstream file(filepath);
    if(!file.is_open()){
        throw std::runtime_error("Could not open joint limits file: " + filepath);
    }

    json j;
    file >> j;

    joint_limits.clear();
    for(size_t i=0; i<joint_names.size(); ++i){
        std::string jname = "q" + std::to_string(i+1);
        if(j.contains(jname)){
            double minv = j[jname][0];
            double maxv = j[jname][1];
            joint_limits[joint_names[i]] = {minv, maxv};
        }else{
            throw std::runtime_error("Joint " + jname + " not found in joint_limits.json");
        }
    }

    if(debug){
        std::cout << "Loaded joint limits:" << std::endl;
        for(auto& [name, lim] : joint_limits){
            std::cout << " " << name << ": [" << lim.first << ", " << lim.second << "]" << std::endl;
        }
    }
}


bool RobotKinematics::isWithinLimits(const Eigen::VectorXd& q) const
{
    if(q.size() != joint_limits.size())
        throw std::invalid_argument("q size does not match number of joints");

    int idx=0;
    for(auto& [name, lim] : joint_limits){
        if(q(idx) < lim.first || q(idx) > lim.second)
            return false;
        idx++;
    }
    return true;
}

Eigen::VectorXd RobotKinematics::clampToLimits(const Eigen::VectorXd& q) const
{
    if(q.size() != joint_limits.size())
        throw std::invalid_argument("q size does not match number of joints");

    Eigen::VectorXd q_clamped = q;
    int idx=0;
    for(auto& [name, lim] : joint_limits){
        q_clamped(idx) = std::min(std::max(q(idx), lim.first), lim.second);
        idx++;
    }
    return q_clamped;
}

std::pair<double,double> RobotKinematics::getJointLimit(const std::string& joint_name) const
{
    auto it = joint_limits.find(joint_name);
    if(it == joint_limits.end())
        throw std::invalid_argument("Joint " + joint_name + " not found in limits");
    return it->second;
}