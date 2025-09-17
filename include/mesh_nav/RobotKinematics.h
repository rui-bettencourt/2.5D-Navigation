#pragma once

#include <pinocchio/multibody/fcl.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

class RobotKinematics {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    RobotKinematics(const std::string& urdf_path, const std::vector<std::string>& joint_names);

    // Deep-copy semantics (needed for TLS clones)
    RobotKinematics(const RobotKinematics& o);
  RobotKinematics& operator=(const RobotKinematics& o);

    RobotKinematics(RobotKinematics&&) noexcept = default;
    RobotKinematics& operator=(RobotKinematics&&) noexcept = default;

    std::unique_ptr<RobotKinematics> clone() const {
        return std::make_unique<RobotKinematics>(*this);
    }

    // Compute forward kinematics
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> getJointPositions(const Eigen::VectorXd& q);

    // Pinocchio function
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> forwardKinematicsPinocchio(const Eigen::VectorXd& q);
    // Manual version from homogeneous transformations
    void createManualTransforms(std::vector<Eigen::Isometry3d>& transforms, const Eigen::VectorXd& q);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> forwardKinematicsManual(const Eigen::VectorXd& q);

    // Compute Jacobian
    std::vector<Eigen::MatrixXd> computeJacobians(const Eigen::VectorXd& q);
    Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& q, int target_joint_idx);
    // from Pinocchio
    Eigen::MatrixXd computeJacobiansPinocchio(const Eigen::VectorXd& q_manip, int target_joint_idx);
    //from manual calculation
    Eigen::MatrixXd computeJacobiansManual(const Eigen::VectorXd& q, int target_joint_idx);

    // Update joints names
    void updateJointsNames(const std::vector<std::string>& joint_names);

    size_t getDOF() const;

    // Convert world vector to local (robot base frame)
    Eigen::Vector3d worldToLocal(const Eigen::Vector3d& vec, bool normalize = false) const;

    // Compute joint torques from Cartesian force
    Eigen::VectorXd calculateJointTorques(const Eigen::Vector3d& force,
                                          int joint_id,
                                          const Eigen::VectorXd& q,
                                          const Eigen::MatrixXd* J = nullptr);

    // Return joint limits (dummy if manual mode)
    std::map<std::string, std::pair<double,double>> getJointsLimits() const;

    // Set robot base pose (just stored, doesn’t affect kinematics in manual mode)
    void setRobotPose(double x, double y, double z,
                      double roll, double pitch, double yaw);

    // Joint limits
    void loadJointLimits(const std::string& filepath, const std::vector<std::string>& joint_names);
    bool isWithinLimits(const Eigen::VectorXd& q) const;
    Eigen::VectorXd clampToLimits(const Eigen::VectorXd& q) const;
    std::pair<double,double> getJointLimit(const std::string& joint_name) const;

    Eigen::VectorXd packToModelConfig(const Eigen::VectorXd& q_manip) const;
    Eigen::MatrixXd sliceToManipulator(const Eigen::MatrixXd &J_full, int cols) const;

    // --- DEBUG HOOKS (add these) ---
    void debugPrint(const char* tag = "") const;
    const void* modelAddress() const;  // for identity checks
    const void* dataAddress()  const;

private:
    const bool debug = true;
    const bool use_pinocchio = true;
    std::unique_ptr<pinocchio::Model> model;
    std::unique_ptr<pinocchio::Data>  data;

    // configuration/state that must be cloned
    std::vector<pinocchio::JointIndex> joint_ids;
    double num_joints;
    Eigen::Isometry3d base_pose = Eigen::Isometry3d::Identity();
    std::map<std::string, std::pair<double,double>> joint_limits;

    //Constants for manual method
    // Constants
    static constexpr double q1z = -1.571;
    static constexpr double q2x = 1.571;
    static constexpr double q2y = 0.0;
    static constexpr double q3x = -M_PI / 2;
    static constexpr double q3z = M_PI / 2;
    static constexpr double q4x = -1.571;
    static constexpr double q4y = -1.571;
    static constexpr double q5x = -1.571;
    static constexpr double q5y = -1.571;
    static constexpr double q5z = 1.571;
    static constexpr double q6x = -1.571;
    static constexpr double q6y = -1.571;
};
