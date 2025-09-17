#include "mesh_nav/RobotKinematics.h"
#include <fstream>
#include <nlohmann/json.hpp>   // requires: nlohmann_json library

using json = nlohmann::json;
using namespace pinocchio;

RobotKinematics::RobotKinematics(const std::string& urdf_path, const std::vector<std::string>& joint_names)
{
    if(use_pinocchio){
        model = std::make_unique<pinocchio::Model>();
        pinocchio::urdf::buildModel(urdf_path, *model);
        std::cout << "model name: " << model->name << std::endl;
        // Create data required by the algorithms
        data = std::make_unique<pinocchio::Data>(*model);
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

// --- deep copy ctor ---
RobotKinematics::RobotKinematics(const RobotKinematics& o)
: joint_ids(o.joint_ids)
, num_joints(o.num_joints)
, base_pose(o.base_pose)
, joint_limits(o.joint_limits)
{
  // Deep-copy Pinocchio model/data
  if (o.model) {
    model = std::make_unique<pinocchio::Model>(*o.model);
    data  = std::make_unique<pinocchio::Data>(*model);
  } else {
    model.reset();
    data.reset();
  }

  // If you have any other caches/ids derived from the model, rebuild them here.
}

// --- deep assignment ---
RobotKinematics& RobotKinematics::operator=(const RobotKinematics& o)
{
  if (this == &o) return *this;

  joint_ids   = o.joint_ids;
  num_joints  = o.num_joints;
  base_pose   = o.base_pose;
  joint_limits= o.joint_limits;

  if (o.model) {
    model = std::make_unique<pinocchio::Model>(*o.model);
    data  = std::make_unique<pinocchio::Data>(*model);
  } else {
    model.reset();
    data.reset();
  }

  // Rebuild any derived caches here if you maintain them.

  return *this;
}

void RobotKinematics::updateJointsNames(const std::vector<std::string>& joint_names)
{
    std::cout << "Updating joints names" << std::endl;
    joint_ids.clear();
    for (const auto& name : joint_names)
    {
        pinocchio::JointIndex joint_id = model->getJointId(name);

        if (joint_id == 0 || joint_id >= model->njoints)
            throw std::invalid_argument("Invalid joint name: " + name);

        joint_ids.push_back(joint_id);
    }
    std::cout << "Finished update" << std::endl;
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> RobotKinematics::getJointPositions(
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

Eigen::MatrixXd RobotKinematics::computeJacobiansPinocchio(const Eigen::VectorXd& q_manip,
                                                           int target_joint_idx)
{
    if(!use_pinocchio)
        throw std::runtime_error("computeJacobiansPinocchio called in manual mode");

    if (target_joint_idx < 1 || target_joint_idx > static_cast<int>(joint_ids.size()))
        throw std::invalid_argument("Invalid target_joint_idx");

    // 1) Map 7-DoF manip vector -> full model configuration
    Eigen::VectorXd q_full = packToModelConfig(q_manip);

    // 2) Update kinematics
    pinocchio::forwardKinematics(*model, *data, q_full);

    // 3) Compute all joint Jacobians (fills data->JS)
    pinocchio::computeJointJacobians(*model, *data, q_full);

    // 4) Update placements (your version uses updateGlobalPlacements)
    pinocchio::updateGlobalPlacements(*model, *data);

    // 5) Extract 6×nv Jacobian for the *target joint* in WORLD frame
    Eigen::MatrixXd J_full(6, model->nv);
    J_full.setZero();
    const pinocchio::JointIndex jid_target = joint_ids[target_joint_idx - 1];
    pinocchio::getJointJacobian(*model, *data, jid_target, pinocchio::ReferenceFrame::WORLD, J_full);

    // 6) Reduce to 6 × target_joint_idx (your manipulator joints only)
    Eigen::MatrixXd J_red = sliceToManipulator(J_full, target_joint_idx);

    return J_red;
}


size_t RobotKinematics::getDOF() const
{
    if(use_pinocchio){
        return num_joints;  // or model.nv
    }else{
        return 7;
    }
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> RobotKinematics::forwardKinematicsPinocchio(
    const Eigen::VectorXd& q)
{
    Eigen::VectorXd q_full = packToModelConfig(q);
    // std::cout << "Starting forward kinematics" << std::endl;
    // Compute forward kinematics once
    pinocchio::forwardKinematics(*model, *data, q_full);

    // std::cout << "Initializing positions vector" << std::endl;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> positions;
    positions.reserve(joint_ids.size());  // avoid reallocations

    // std::cout << "Pushing results" << std::endl;
    for (auto joint_id : joint_ids)
        positions.push_back(data->oMi[joint_id].translation());

    return positions;
}

// Build a full-size configuration from a 7-DoF manip vector.
// Assumes each manip joint has nq()==1 (revolute/prismatic). Adjust if needed.
Eigen::VectorXd RobotKinematics::packToModelConfig(const Eigen::VectorXd& q_manip) const
{
    if(!use_pinocchio) return q_manip; // manual branch uses 7-dim already

    if(static_cast<int>(q_manip.size()) != static_cast<int>(joint_ids.size()))
        throw std::invalid_argument("q_manip size != number of manip joints");

    // Start from neutral (or cached default) so all non-manip joints are well-defined
    Eigen::VectorXd q_full = pinocchio::neutral(*model);

    for (int k = 0; k < static_cast<int>(joint_ids.size()); ++k) {
        const pinocchio::JointIndex jid = joint_ids[k];
        const auto &jmodel = model->joints[jid];
        const int iq = model->idx_qs[jid];
        const int nqj = jmodel.nq();            // expected 1
        if (nqj != 1) {
            throw std::runtime_error("packToModelConfig: joint " + model->names[jid] +
                                     " has nq=" + std::to_string(nqj) +
                                     " (only nq==1 supported in this mapper)");
        }
        q_full[iq] = q_manip[k];
    }
    return q_full;
}

// Slice a 6×nv Jacobian (WORLD frame) down to 6×cols, taking only the columns
// for manipulator joints 0..cols-1 (by joint_ids order). Assumes nv(j)=1.
Eigen::MatrixXd RobotKinematics::sliceToManipulator(const Eigen::MatrixXd &J_full,
                                                    int cols) const
{
    Eigen::MatrixXd J_red(6, cols);
    for (int k = 0; k < cols; ++k) {
        const pinocchio::JointIndex jid = joint_ids[k];
        const auto &jmodel = model->joints[jid];
        const int iv = model->idx_vs[jid];
        const int nvj = jmodel.nv();            // expected 1
        if (nvj != 1) {
            throw std::runtime_error("sliceToManipulator: joint " + model->names[jid] +
                                     " has nv=" + std::to_string(nvj) +
                                     " (only nv==1 supported here)");
        }
        // Take the single column for that joint’s velocity DoF
        J_red.col(k) = J_full.col(iv);
    }
    return J_red;
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> RobotKinematics::forwardKinematicsManual(const Eigen::VectorXd& q) {
    std::vector<Eigen::Isometry3d> transforms;
    createManualTransforms(transforms, q);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> positions;
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

Eigen::MatrixXd RobotKinematics::computeJacobiansManual(const Eigen::VectorXd& q,
                                                        int target_joint_idx)
{
    // target_joint_idx is 1..num_joints (your API uses 1-based here)
    if (target_joint_idx < 1 || target_joint_idx > num_joints)
        throw std::invalid_argument("Invalid target_joint_idx");

    const int cols = target_joint_idx; // number of active joints to the target
    Eigen::Matrix<double,6,Eigen::Dynamic> J(6, cols);
    J.setZero();

    // Forward kinematics to each joint frame (0..num_joints-1)
    std::vector<Eigen::Isometry3d> T;
    createManualTransforms(T, q);
    // T.size() should be >= num_joints, with T[j] = world->joint_j

    const Eigen::Vector3d p_target = T[target_joint_idx - 1].translation();

    for (int j = 0; j < cols; ++j) {
        const Eigen::Vector3d z_j = T[j].rotation().col(2);  // world z-axis of joint j
        const Eigen::Vector3d p_j = T[j].translation();      // world position of joint j
        const Eigen::Vector3d r   = p_target - p_j;

        J.block<3,1>(0, j) = z_j.cross(r); // linear part
        J.block<3,1>(3, j) = z_j;          // angular part
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
    const int dof = static_cast<int>(getDOF());
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(dof);
    if (dof <= 0 || joint_id <= 0) return tau;

    // Manual mode: 6 × joint_id; Pinocchio: 6 × nv
    Eigen::MatrixXd J = J_ptr ? *J_ptr : computeJacobian(q, joint_id);
    if (J.rows() != 6) return tau; // robust guard

    const int jcols_int = static_cast<int>(J.cols());
    const int ncols = std::min(jcols_int, std::min(joint_id, dof));
    if (ncols <= 0) return tau;

    // Spatial force: linear part only
    Eigen::Matrix<double,6,1> F;
    F << force, 0.0, 0.0, 0.0;

    // Torques for joints [1..ncols] only
    const Eigen::VectorXd tau_partial = J.leftCols(ncols).transpose() * F; // (ncols×1)
    tau.head(ncols) = tau_partial;  // pad to DOF


    // std::cout << "[calculateJointTorques] joint_id=" << joint_id
    //         << " J=(" << J.rows() << "x" << J.cols() << ")"
    //         << " ncols=" << ncols
    //         << " tau_partial.size()=" << tau_partial.size()
    //         << " tau.size()=" << tau.size()
    //         << std::endl;

    const int show = std::min<int>(dof, 10);
    // std::cout << "tau[0.." << show-1 << "]: ";
    // for (int i = 0; i < show; ++i) std::cout << tau[i] << ' ';
    //     std::cout << std::endl;

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
            const auto& jmodel = model->joints[jid];
            std::string name = model->names[jid];
            Eigen::VectorXd lower = model->lowerPositionLimit.segment(model->idx_qs[jid], jmodel.nq());
            Eigen::VectorXd upper = model->upperPositionLimit.segment(model->idx_qs[jid], jmodel.nq());
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

void RobotKinematics::debugPrint(const char* tag) const {
  // Adjust these to your real members
  const int dof = getDOF();
  std::cout << "[RK " << (tag ? tag : "") << "] dof=" << dof
            << " model@" << modelAddress()
            << " data@"  << dataAddress()
            << "\n";
  // If you have names/limits, print sizes (don’t spam names)
  // std::cout << "  limits: lower=" << lower_.size() << " upper=" << upper_.size() << "\n";
}

const void* RobotKinematics::modelAddress() const {
  // If you store a unique_ptr, return pointer value; if by value, return &model
  // return static_cast<const void*>(model_.get());
  // or:
  // return static_cast<const void*>(&model);
  // Replace with your actual member:
  return model.get(); // TODO: replace with real
}

const void* RobotKinematics::dataAddress() const {
  // return static_cast<const void*>(data_.get());
  // or:
  // return static_cast<const void*>(&data);
  return data.get(); // TODO: replace with real
}