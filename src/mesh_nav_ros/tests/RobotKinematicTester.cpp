// #include <iostream>
// #include "mesh_nav/RobotKinematics.h"

// #include <chrono>
// #include <iostream>
// #include <vector>
// #include <string>

// int main() {
//     std::cout << "RobotKinematics test..." << std::endl;

//     // List of joint names to get positions for
//     std::vector<std::string> frames_names = {
//         "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"
//     };

//     RobotKinematics rk("/home/rui/socrob_ws/src/isr_tiago/simulation/mbot_simulation_environments/robots/tiago_ouster.urdf", "base_link", "arm_7_link", frames_names);

//     Eigen::VectorXd q = Eigen::VectorXd::Zero(rk.getDOF());


//     // rk.updateJointsNames(frames_names);

//     // Measure FK time
//     auto t1 = std::chrono::high_resolution_clock::now();
//     auto joint_positions = rk.getJointPositions(q);
//     auto t2 = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<double> fk_duration = t2 - t1;

//     std::cout << "Joint positions:\n";
//     for (size_t i = 0; i < joint_positions.size(); ++i) {
//         std::cout << i << ": " << joint_positions[i].transpose() << std::endl;
//     }
//     std::cout << "Forward kinematics time: " << fk_duration.count() * 1000 << " ms" << std::endl;

//     for (int i = 0; i < 10; ++i) {
//         t1 = std::chrono::high_resolution_clock::now();
//         joint_positions = rk.getJointPositions(q);
//         t2 = std::chrono::high_resolution_clock::now();

//         fk_duration = t2 - t1;

//         std::cout << "Iteration " << i + 1 << " - Forward kinematics time: " << fk_duration.count() * 1000 << " ms" << std::endl;
//     }

//     // frames_names = {
//     //     "arm_7_joint"
//     // };

//     // Measure Jacobian time
//     t1 = std::chrono::high_resolution_clock::now();
//     Eigen::MatrixXd J = rk.computeJacobian(q,7);
//     t2 = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<double> jac_duration = t2 - t1;

//     std::cout << "Jacobian for joint 7 " << ":\n" << J << std::endl << std::endl;

//     std::cout << "Jacobian computation time: " << jac_duration.count() * 1000 << " ms" << std::endl;
//     // Measure Jacobian time
//     t1 = std::chrono::high_resolution_clock::now();
//     std::vector<Eigen::MatrixXd> Js = rk.computeJacobians(q);
//     t2 = std::chrono::high_resolution_clock::now();

//     jac_duration = t2 - t1;

//     std::cout << "Number of jacobians: " << Js.size() <<std::endl;

//     for (size_t i = 0; i < Js.size(); ++i) {
//         std::cout << "Jacobian for " << frames_names[i] << ":\n" << Js[i] << std::endl << std::endl;
//     }

//     std::cout << "Jacobian computation time: " << jac_duration.count() * 1000 << " ms" << std::endl;

//     return 0;
// }

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "mesh_nav/RobotKinematics.h"

// Helper to measure average execution time over N iterations
template <typename Func>
double measureTime(Func f, int iterations = 10) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) f();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / iterations;
}

int main() {
    std::cout << "==== RobotKinematics Full Test ====" << std::endl;

    std::vector<std::string> frames_names = {"mani_1","mani_2","mani_3","mani_4","mani_5","mani_6"};

    // RobotKinematics rk(
    //     "/home/rui/socrob_ws/src/isr_tiago/simulation/mbot_simulation_environments/robots/tiago_ouster.urdf",
    //     "base_link", "arm_7_link", frames_names
    // );
    RobotKinematics rk(
        // "/home/rui/test_ws/src/REMANI-Planner/remani_planner/mm_config/meshes/FastArmer/mm_robot.urdf",
        "/home/rui/mesh_nav_ws/src/REMANI-Planner/remani_planner/mm_config/meshes/ur5/ur5.urdf",
        "mm_base", "mani_5", frames_names
    );

    Eigen::VectorXd q = Eigen::VectorXd::Zero(rk.getDOF());
    // Eigen::VectorXd q = Eigen::VectorXd::Constant(rk.getDOF(), 1.57);
    // q[2]=1.57;

    // --- Test Forward Kinematics ---
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> joint_positions;
    double fk_time = measureTime([&]() {
        joint_positions = rk.getJointPositions(q);
    }, 10);

    std::cout << "\nForward Kinematics result:" << std::endl;
    for (size_t i = 0; i < joint_positions.size() && i < frames_names.size(); ++i)
        std::cout << frames_names[i] << ": " << joint_positions[i].transpose() << std::endl;
    std::cout << "Average FK runtime: " << fk_time << " ms" << std::endl;

    // --- Test single Jacobian ---
    Eigen::MatrixXd J;
    double jac_time = measureTime([&]() {
        J = rk.computeJacobian(q, 6);
    }, 10);

    std::cout << "\nJacobian for joint 6:\n" << J << std::endl;
    std::cout << "Average single-Jacobian runtime: " << jac_time << " ms" << std::endl;

    // --- Test all Jacobians ---
    std::vector<Eigen::MatrixXd> Js;
    double jacs_time = measureTime([&]() {
        Js = rk.computeJacobians(q);
    }, 10);

    std::cout << "\nComputed " << Js.size() << " Jacobians" << std::endl;
    if (!Js.empty())
        std::cout << "Jacobian for joint 1:\n" << Js[0] << std::endl;
    std::cout << "Average all-Jacobians runtime: " << jacs_time << " ms" << std::endl;

    // --- Test Joint Limits ---
    auto limits = rk.getJointsLimits();
    std::cout << "\nJoint Limits:" << std::endl;
    for (auto& [name, lim] : limits)
        std::cout << " " << name << ": [" << lim.first << ", " << lim.second << "]" << std::endl;

    // --- Test isWithinLimits & clampToLimits ---
    Eigen::VectorXd q_test = q;
    q_test(2) = -10.0; // deliberately outside limit

    bool within = rk.isWithinLimits(q_test);
    std::cout << "\nIs q_test within limits? " << (within ? "YES" : "NO") << std::endl;

    auto q_clamped = rk.clampToLimits(q_test);
    std::cout << "Clamped q_test: " << q_clamped.transpose() << std::endl;

    // --- Test getJointLimit ---
    auto lim_q3 = rk.getJointLimit("joint3");
    std::cout << "\narm_3_joint limits: [" << lim_q3.first << ", " << lim_q3.second << "]" << std::endl;

    // --- Stress test all functions ---
    int iterations = 1000;
    double fk_stress = measureTime([&]() { rk.getJointPositions(q); }, iterations);
    double jac_stress = measureTime([&]() { rk.computeJacobian(q, 6); }, iterations);
    double jacs_stress = measureTime([&]() { rk.computeJacobians(q); }, iterations);
    double limit_check_stress = measureTime([&]() { rk.isWithinLimits(q); }, iterations);

    std::cout << "\n==== Stress Test (" << iterations << " iterations) ====" << std::endl;
    std::cout << " FK avg runtime: " << fk_stress << " ms" << std::endl;
    std::cout << " Single Jacobian avg runtime: " << jac_stress << " ms" << std::endl;
    std::cout << " All Jacobians avg runtime: " << jacs_stress << " ms" << std::endl;
    std::cout << " Limit check avg runtime: " << limit_check_stress << " ms" << std::endl;

    std::cout << "\n--- Test worldToLocal ---\n";
    Eigen::Vector3d world_point(1.0, 2.0, 0.5);
    Eigen::Vector3d local_point = rk.worldToLocal(world_point);
    std::cout << "World point: " << world_point.transpose()
            << " -> Local point: " << local_point.transpose() << std::endl;

    std::cout << "\n--- Test setRobotPose ---\n";
    Eigen::Vector3d new_base_pos(0.5, -0.5, 0.0);

    rk.setRobotPose(new_base_pos.x(), new_base_pos.y(), new_base_pos.z(),
                0, 0, M_PI/4);

    // Check FK after moving the base
    joint_positions = rk.getJointPositions(q);
    for (size_t i = 0; i < joint_positions.size() && i < frames_names.size(); ++i)
        std::cout << frames_names[i] << ": " << joint_positions[i].transpose() << std::endl;

    // Test worldToLocal again after moving the robot
    local_point = rk.worldToLocal(world_point);
    std::cout << "After moving robot, world point: " << world_point.transpose()
            << " -> Local point: " << local_point.transpose() << std::endl;

    std::cout << "\n--- Test calculateJointTorques ---\n";
    // Sample wrench: [Fx, Fy, Fz, Tx, Ty, Tz]
    Eigen::Matrix<double, 6, 1> wrench;
    wrench << 10.0, 0.0, 0.0, 0.0, 0.0, 1.0;

    Eigen::Vector3d force = wrench.head<3>();  // extract linear part
    int joint_idx = 6;                          // whichever joint you want
    Eigen::VectorXd torques = rk.calculateJointTorques(force, joint_idx, q);
    std::cout << "Joint torques for sample wrench: " << torques.transpose() << std::endl;

    std::cout << "\n==== Test Completed ====" << std::endl;
    return 0;
}

