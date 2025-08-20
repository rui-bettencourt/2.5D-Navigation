#pragma once

#include <pinocchio/multibody/fcl.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

class RobotKinematics {
public:
    RobotKinematics(const std::string& urdf_path, const std::string& base_link, const std::string& ee_link, const std::vector<std::string>& joint_names);

    // Compute forward kinematics
    std::vector<Eigen::Vector3d> getJointPositions(const Eigen::VectorXd& q);

    // Pinocchio function
    std::vector<Eigen::Vector3d> forwardKinematicsPinocchio(const Eigen::VectorXd& q);
    // Manual version from homogeneous transformations
    void createManualTransforms(std::vector<Eigen::Isometry3d>& transforms, const Eigen::VectorXd& q);
    std::vector<Eigen::Vector3d> forwardKinematicsManual(const Eigen::VectorXd& q);

    // Compute Jacobian
    std::vector<Eigen::MatrixXd> computeJacobians(const Eigen::VectorXd& q);
    Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& q, int target_joint_idx);
    // from Pinocchio
    Eigen::MatrixXd computeJacobiansPinocchio(const Eigen::VectorXd& q, int target_joint_idx);
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

private:
    const bool debug = true;
    const bool use_pinocchio = false;
    pinocchio::Model model;
    pinocchio::Data data;
    std::vector<pinocchio::JointIndex> joint_ids;
    double num_joints;
    std::string base_frame;
    std::string end_effector;
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
