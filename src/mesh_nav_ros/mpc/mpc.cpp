#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <nlohmann/json.hpp>
#include <ros/ros.h>

using namespace casadi;
using namespace std;
using json = nlohmann::json;

class MPC {
public:
    double T;
    int N;
    int nr_states, nr_controls;
    int nr_states_base = 8, nr_states_arm = 7;
    int nr_controls_base = 2, nr_controls_arm = 7;

    casadi::Opti opti;
    casadi::OptiSol sol;

    double alpha = 2.0, beta = 1.0, gamma = 0.1, g = 9.8, motor_strength = 1.0;
    json joints_limits;
    map<string, double> configs;

    std::vector<casadi::MX> dq;

    casadi::MX vel_x, vel_yaw;
    std::vector<casadi::MX> q;

    MPC(double T_, int N_) : T(T_), N(N_) {
        nr_states = nr_states_base + nr_states_arm;
        nr_controls = nr_controls_base + nr_controls_arm;

        // Set control limits
        configs = {
            {"max_Fx", 1.0},  {"max_Fyaw", 1.0}, {"max_vx", 0.3},  {"max_vyaw", 1.57}, {"max_vqs", 1.95},
            {"min_Fx", -1.0}, {"min_Fyaw", -1.0}, {"min_vx", -0.05}, {"min_vyaw", -1.57}, {"min_vqs", -1.95}
        };

        // Load joint limits from file
        ifstream f("/home/rui/pcl_ws/src/mesh_nav/data/joints_limits.json");
        if (f.is_open()) {
            f >> joints_limits;
        } else {
            cerr << "Warning: Could not open joints_limits.json" << endl;
        }
    }

    vector<double> interpolate(double start, double goal, int N) {
        vector<double> result(N);
        if (N <= 1) {
            result[0] = start;
            return result;
        }
        double step = (goal - start) / (N - 1);
        for (int i = 0; i < N; ++i) {
            result[i] = start + i * step;
        }
        return result;
    }

    bool solve_mpc(const map<string, double>& state, const map<string, double>& gs) {
        Opti opti;

        MX X = opti.variable(nr_states, N + 1);
        MX U = opti.variable(nr_controls, N);

        auto f = [&](MX x, MX u) -> MX {
            return casadi::MX::vertcat({
                x(6) * cos(x(5)),
                x(6) * sin(x(5)),
                MX(0), MX(0), MX(0),
                x(7),
                u(0),
                u(1),
                u(2), u(3), u(4), u(5), u(6), u(7), u(8)
            });
        };

        double dt = T / N;

        for (int k = 0; k < N; ++k) {
            MX k1 = f(X(casadi::Slice(), k), U(casadi::Slice(), k));
            MX k2 = f(X(casadi::Slice(), k) + dt/2 * k1, U(casadi::Slice(), k));
            MX k3 = f(X(casadi::Slice(), k) + dt/2 * k2, U(casadi::Slice(), k));
            MX k4 = f(X(casadi::Slice(), k) + dt * k3, U(casadi::Slice(), k));
            MX x_next = X(casadi::Slice(), k) + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4);
            opti.subject_to(X(casadi::Slice(), k + 1) == x_next);
        }

        // Objective (simplified)
        MX objective = 0;
        for (int k = 1; k <= N; ++k) {
            objective += alpha * pow(X(0, k) - gs.at("x"), 2);
            objective += alpha * pow(X(1, k) - gs.at("y"), 2);
            objective += beta * pow(X(8, k) - gs.at("q1"), 2);
        }
        opti.minimize(objective);

        // Solver options
        Dict opts;
        opts["print_time"] = false;
        opts["ipopt.print_level"] = 0;
        opts["ipopt.max_iter"] = 10000;
        opti.solver("ipopt", opts);

        try {
            sol = opti.solve();  // Store in class member
            cout << "Solved successfully.\n";

            // Optional: extract values to class members for access
            casadi::MX Fx = U(0, casadi::Slice());
            casadi::MX Fyaw = U(1, casadi::Slice());
            dq.clear();
            for (int j = 2; j < 9; ++j)
                dq.push_back(U(j, casadi::Slice()));
            vel_x = X(6, casadi::Slice());
            vel_yaw = X(7, casadi::Slice());
            q.clear();
            for (int j = 8; j < 15; ++j)
                q.push_back(X(j, casadi::Slice()));

            return true;
        } catch (std::exception& e) {
            cerr << "Solver failed: " << e.what() << endl;
            return false;
        }
    }

    // map<string, double> get_control(int i = 0) {
    //     map<string, double> control;
    //     control["Fx"] = sol.value(Fx(i)).scalar();
    //     control["Fyaw"] = sol.value(Fyaw(i)).scalar();

    //     for (int j = 0; j < 7; ++j) {
    //         string key = "dq" + to_string(j + 1);
    //         control[key] = sol.value(dq[j](i)).scalar();
    //     }

    //     return control;
    // }

    // map<string, double> get_control_pose(int i = 0) {
    //     map<string, double> control;
    //     control["vx"] = sol.value(vel_x(i)).scalar();
    //     control["vyaw"] = sol.value(vel_yaw(i)).scalar();

    //     for (int j = 0; j < 7; ++j) {
    //         string key = "q" + to_string(j + 1);
    //         control[key] = sol.value(q[j](i)).scalar();
    //     }

    //     return control;
    // }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mesh_nav_node");
    ros::NodeHandle nh;

    // Create an MPC object with required parameters (example values)
    double T_ = 10.0;  // Time horizon
    int N_ = 50;  // Number of steps

    MPC mpc(T_, N_);

    // Create mock state and goal maps for testing
    map<string, double> state = {{"x", 0.0}, {"y", 0.0}, {"q1", 0.0}};
    map<string, double> goal = {{"x", 10.0}, {"y", 10.0}, {"q1", 1.57}};

    if (mpc.solve_mpc(state, goal)) {
        cout << "MPC solved successfully!" << endl;
    } else {
        cout << "MPC solving failed." << endl;
    }

    ros::spin();
    return 0;
}