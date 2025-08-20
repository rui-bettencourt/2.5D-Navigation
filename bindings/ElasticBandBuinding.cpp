#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mesh_nav/ElasticBandPlanner.h"

namespace py = pybind11;
using namespace ebp;

// convert python list-of-dicts to PathMatrix and back
static PathMatrix path_from_py(const py::list& path){
    if(path.empty()) throw std::runtime_error("Empty path");
    PathMatrix mat;
    py::object first = path[0];
    std::vector<std::string> joint_keys;
    for(auto item : first.attr("items")()){
        auto pair = item.cast<py::tuple>();
        std::string k = pair[0].cast<std::string>();
        if(!k.empty() && k[0]=='q') joint_keys.push_back(k);
    }
    std::sort(joint_keys.begin(), joint_keys.end());
    for(size_t i=0;i<path.size();++i){
        py::object wp = path[i];
        Row row;
        row.push_back(py::float_(wp["x"]));
        row.push_back(py::float_(wp["y"]));
        row.push_back(py::float_(wp["z"]));
        row.push_back(py::float_(wp["roll"]));
        row.push_back(py::float_(wp["pitch"]));
        row.push_back(py::float_(wp["yaw"]));
        for(auto &qk: joint_keys) row.push_back(py::float_(wp[qk.c_str()]));
        mat.push_back(row);
    }
    return mat;
}

static py::list path_to_py(const PathMatrix& mat){
    py::list out;
    if(mat.empty()) return out;
    int num_joints = static_cast<int>(mat[0].size()) - 6;
    for(const auto &row: mat){
        py::dict d;
        d["x"] = row[0]; d["y"] = row[1]; d["z"] = row[2];
        d["roll"] = row[3]; d["pitch"] = row[4]; d["yaw"] = row[5];
        for(int j=0;j<num_joints;++j) d[std::string("q") + std::to_string(j+1)] = row[6+j];
        out.append(d);
    }
    return out;
}

PYBIND11_MODULE(elastic_band_planner_cpp, m) {
    py::class_<ElasticBandPlanner>(m, "ElasticBandPlanner")
        .def(py::init<double,double,double,double,double,double,double,double,double,double,double>(),
             py::arg("k_attraction_base")=0.2, py::arg("k_repulsion_base")=0.2,
             py::arg("k_attraction_joints")=0.1, py::arg("k_repulsion_joints")=0.1,
             py::arg("k_repulsion_robot_joints")=0.1, py::arg("k_update_joints")=0.1,
             py::arg("k_orientation")=0.1, py::arg("k_orientation_from_base")=0.1,
             py::arg("k_position_from_orientation")=0.0, py::arg("k_safety_joints")=0.05,
             py::arg("obstacle_threshold")=1.5)

        .def("initialize", &ElasticBandPlanner::initialize,
             py::arg("sdf_bin_path"),
             "Initialize the planner with an SDF file and default robot points.")

        .def("update_path", [](ElasticBandPlanner &self, py::list path,
                               int iterations, double convergence_threshold){
            PathMatrix pm = path_from_py(path);
            auto result = self.update_path(pm, obs_handle, iterations, convergence_threshold);
            return path_to_py(result);
        }, py::arg("path"), py::arg("obs_mesh"),
           py::arg("iterations")=100, py::arg("convergence_threshold")=1e-3);

    m.doc() = "Elastic band planner with voxel SDF integration (no RobotKinematics/RobotModel bindings).";
}
