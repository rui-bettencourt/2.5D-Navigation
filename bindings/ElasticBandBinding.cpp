#include "mesh_nav/ElasticBandPlanner.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;
using namespace ebp;

// Convert Python list of dicts -> Eigen::MatrixXd
Eigen::MatrixXd path_from_py(const py::list &path, int num_joints) {
    int num_waypoints = path.size();
    Eigen::MatrixXd mat(num_waypoints, 6 + num_joints);

    for (int i = 0; i < num_waypoints; ++i) {
        py::dict wp = path[i].cast<py::dict>();

        mat(i, 0) = wp["x"].cast<double>();
        mat(i, 1) = wp["y"].cast<double>();
        mat(i, 2) = wp["z"].cast<double>();
        mat(i, 3) = wp["roll"].cast<double>();
        mat(i, 4) = wp["pitch"].cast<double>();
        mat(i, 5) = wp["yaw"].cast<double>();

        for (int j = 0; j < num_joints; ++j) {
            std::string key = "q" + std::to_string(j + 1);
            mat(i, 6 + j) = wp[py::str(key)].cast<double>();
        }
    }

    return mat;
}

// Convert Eigen::MatrixXd -> Python list of dicts
py::list path_to_py(const Eigen::MatrixXd &mat, int num_joints) {
    py::list path;

    for (int i = 0; i < mat.rows(); ++i) {
        py::dict d;

        d["x"] = mat(i, 0);
        d["y"] = mat(i, 1);
        d["z"] = mat(i, 2);
        d["roll"] = mat(i, 3);
        d["pitch"] = mat(i, 4);
        d["yaw"] = mat(i, 5);

        for (int j = 0; j < num_joints; ++j) {
            std::string key = "q" + std::to_string(j + 1);
            d[py::str(key)] = mat(i, 6 + j);
        }

        path.append(d);
    }

    return path;
}

// Convert Python safe config (dict with q1..qN OR list/tuple) -> std::vector<double>
static std::vector<double> safe_from_py(const py::handle &obj, int num_joints) {
    std::vector<double> q(num_joints, std::numeric_limits<double>::quiet_NaN());

    if (py::isinstance<py::dict>(obj)) {
        py::dict d = obj.cast<py::dict>();
        for (int j = 0; j < num_joints; ++j) {
            std::string key = "q" + std::to_string(j + 1);
            if (d.contains(py::str(key))) {
                q[j] = d[py::str(key)].cast<double>();
            }
        }
    } else if (py::isinstance<py::sequence>(obj)) {
        py::sequence seq = obj.cast<py::sequence>();
        const int L = std::min(num_joints, static_cast<int>(seq.size()));
        for (int j = 0; j < L; ++j) q[j] = seq[j].cast<double>();
    } else {
        throw std::runtime_error("safe_config must be a dict {q1..qN} or a sequence of floats");
    }
    return q;
}

PYBIND11_MODULE(elastic_band_planner_cpp, m) {
    py::class_<ElasticBandPlanner>(m, "ElasticBandPlanner")
        .def(py::init<double,double,double,double,double,double,double,double,double,double,double>(),
             py::arg("k_attraction_base")=0.2, py::arg("k_repulsion_base")=0.2,
             py::arg("k_attraction_joints")=0.1, py::arg("k_repulsion_joints")=0.1,
             py::arg("k_repulsion_robot_joints")=0.0, py::arg("k_update_joints")=0.1,
             py::arg("k_orientation")=0.0, py::arg("k_orientation_from_base")=0.0,
             py::arg("k_position_from_orientation")=0.0, py::arg("k_safety_joints")=0.0,
             py::arg("obstacle_threshold")=1.5)


        .def("initialize", &ElasticBandPlanner::initialize,
             py::arg("sdf_bin_path"), py::arg("robot_sdf_bin_path"),
             "Initialize the planner with an SDF file and default robot points.")

        .def("update_path", [](ElasticBandPlanner &self, py::list path,
                               int iterations, double convergence_threshold){
            PathMatrix pm = path_from_py(path,7);
            // Call the original C++ method using the Python obs_mesh object
            auto result = self.update_path(pm, iterations, convergence_threshold);
            return path_to_py(result,7);
        }, py::arg("path"),
           py::arg("iterations")=100, py::arg("convergence_threshold")=1e-3)

        // --- New: pass safe configuration from Python (dict or list/tuple) ---
        .def("set_safe_config",
             [](ElasticBandPlanner &self, py::object safe_cfg, int num_joints) {
                 self.setSafeConfig(safe_from_py(safe_cfg, num_joints));
             },
             py::arg("safe_config"),
             py::arg("num_joints") = 7,
             "Set a safety configuration. Accepts dict {q1..qN} or a sequence.")

        // --- New: enable/disable dynamic safety weighting (+ center position) ---
        .def("set_dynamic_safety",
             &ElasticBandPlanner::setDynamicSafety,
             py::arg("enabled") = true,
             py::arg("center_activation_safety") = 0.8,
             "Enable dynamic safety weighting and set its center along the path.");
    m.doc() = "Elastic band planner with voxel SDF integration (no RobotKinematics/RobotModel bindings).";
}
