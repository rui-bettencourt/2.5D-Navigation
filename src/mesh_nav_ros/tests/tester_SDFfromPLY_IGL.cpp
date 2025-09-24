// #include <ros/ros.h>
// #include <visualization_msgs/Marker.h>
// #include <geometry_msgs/Point.h>
// #include <Eigen/Dense>

// int main(int argc, char** argv)
// {
//     ros::init(argc, argv, "point_cloud_and_closest_point_viz");
//     ros::NodeHandle nh;
//     ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

//     // Dummy example data: replace with your actual V, P, C
//     Eigen::MatrixXd V(5, 3);
//     V << 0,0,0,
//          1,0,0,
//          1,1,0,
//          0,1,0,
//          0.5,0.5,0;
//     Eigen::Vector3d P(0.25, 0.25, 0);
//     Eigen::Vector3d C(0.6, 0.6, 0);

//     ros::Rate r(1);
//     while (ros::ok())
//     {
//         // 1. Publish vertices as points (black)
//         visualization_msgs::Marker points;
//         points.header.frame_id = "map"; // or your fixed frame
//         points.header.stamp = ros::Time::now();
//         points.ns = "vertices";
//         points.id = 0;
//         points.type = visualization_msgs::Marker::POINTS;
//         points.action = visualization_msgs::Marker::ADD;
//         points.scale.x = 0.02; // size of points
//         points.scale.y = 0.02;
//         points.color.r = 0.0f;
//         points.color.g = 0.0f;
//         points.color.b = 0.0f;
//         points.color.a = 1.0;

//         for (int i = 0; i < V.rows(); ++i)
//         {
//             geometry_msgs::Point p;
//             p.x = V(i,0);
//             p.y = V(i,1);
//             p.z = V(i,2);
//             points.points.push_back(p);
//         }

//         // 2. Publish P as green sphere
//         visualization_msgs::Marker sphere_P;
//         sphere_P.header.frame_id = "map";
//         sphere_P.header.stamp = ros::Time::now();
//         sphere_P.ns = "point_P";
//         sphere_P.id = 1;
//         sphere_P.type = visualization_msgs::Marker::SPHERE;
//         sphere_P.action = visualization_msgs::Marker::ADD;
//         sphere_P.pose.position.x = P.x();
//         sphere_P.pose.position.y = P.y();
//         sphere_P.pose.position.z = P.z();
//         sphere_P.scale.x = 0.05; // size of sphere
//         sphere_P.scale.y = 0.05;
//         sphere_P.scale.z = 0.05;
//         sphere_P.color.r = 0.0f;
//         sphere_P.color.g = 1.0f;
//         sphere_P.color.b = 0.0f;
//         sphere_P.color.a = 1.0;

//         // 3. Publish C as red sphere
//         visualization_msgs::Marker sphere_C;
//         sphere_C.header.frame_id = "map";
//         sphere_C.header.stamp = ros::Time::now();
//         sphere_C.ns = "point_C";
//         sphere_C.id = 2;
//         sphere_C.type = visualization_msgs::Marker::SPHERE;
//         sphere_C.action = visualization_msgs::Marker::ADD;
//         sphere_C.pose.position.x = C.x();
//         sphere_C.pose.position.y = C.y();
//         sphere_C.pose.position.z = C.z();
//         sphere_C.scale.x = 0.05;
//         sphere_C.scale.y = 0.05;
//         sphere_C.scale.z = 0.05;
//         sphere_C.color.r = 1.0f;
//         sphere_C.color.g = 0.0f;
//         sphere_C.color.b = 0.0f;
//         sphere_C.color.a = 1.0;

//         // 4. Publish line between P and C (red)
//         visualization_msgs::Marker line;
//         line.header.frame_id = "map";
//         line.header.stamp = ros::Time::now();
//         line.ns = "line_PC";
//         line.id = 3;
//         line.type = visualization_msgs::Marker::LINE_STRIP;
//         line.action = visualization_msgs::Marker::ADD;
//         line.scale.x = 0.01; // line width
//         line.color.r = 1.0f;
//         line.color.g = 0.0f;
//         line.color.b = 0.0f;
//         line.color.a = 1.0;

//         geometry_msgs::Point p_start, p_end;
//         p_start.x = P.x();
//         p_start.y = P.y();
//         p_start.z = P.z();
//         p_end.x = C.x();
//         p_end.y = C.y();
//         p_end.z = C.z();

//         line.points.push_back(p_start);
//         line.points.push_back(p_end);

//         // Publish all
//         marker_pub.publish(points);
//         marker_pub.publish(sphere_P);
//         marker_pub.publish(sphere_C);
//         marker_pub.publish(line);

//         ros::spinOnce();
//         r.sleep();
//     }

//     return 0;
// }

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <cmath>
#include <cstdlib>

#include <omp.h>
#include <Eigen/Core>

#include "mesh_nav/SDFfromPLY.h"  // must provide: loadMesh(), getVertices(), queryDistance()

struct Args {
    std::string mesh;
    size_t      num   = 1000;   // number of queries
    int         threads = 1;      // 1 = single-thread
    double      margin  = 0.2;    // bbox margin as fraction
    std::string mode    = "uniform"; // "uniform" | "grid"
    unsigned    seed    = 42;
};

static void usage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " --mesh PATH [--num N] [--threads T] [--margin F] [--mode uniform|grid] [--seed S]\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i=1; i<argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* what)->const char*{
            if (i+1 >= argc) { std::cerr << "Missing value for " << what << "\n"; std::exit(2); }
            return argv[++i];
        };
        if      (k == "--mesh")    a.mesh    = need("--mesh");
        else if (k == "--num")     a.num     = std::strtoull(need("--num"), nullptr, 10);
        else if (k == "--threads") a.threads = std::atoi(need("--threads"));
        else if (k == "--margin")  a.margin  = std::atof(need("--margin"));
        else if (k == "--mode")    a.mode    = need("--mode");
        else if (k == "--seed")    a.seed    = static_cast<unsigned>(std::strtoul(need("--seed"), nullptr, 10));
        else if (k == "--help" || k == "-h") { usage(argv[0]); return false; }
        else { std::cerr << "Unknown option: " << k << "\n"; usage(argv[0]); return false; }
    }
    if (a.mesh.empty()) { usage("sdf_query_bench"); return false; }
    return true;
}

static std::string hms(double sec) {
    if (!std::isfinite(sec)) return "--:--:--";
    uint64_t s = (uint64_t)(sec + 0.5);
    uint64_t h = s / 3600; s %= 3600;
    uint64_t m = s / 60;   s %= 60;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02llu:%02llu:%02llu",
                  (unsigned long long)h, (unsigned long long)m, (unsigned long long)s);
    return buf;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    std::cout << "[INFO] Mesh: "    << args.mesh    << "\n";
    std::cout << "[INFO] Num: "     << args.num     << "\n";
    std::cout << "[INFO] Threads: " << args.threads << "\n";
    std::cout << "[INFO] Mode: "    << args.mode    << "  (margin=" << args.margin*100 << "%)\n";

    if (args.threads < 1) args.threads = 1;
    omp_set_num_threads(args.threads);

    // Load mesh & build acceleration
    SDFfromPLY sdf;
    auto t_load0 = std::chrono::steady_clock::now();
    if (!sdf.loadMesh(args.mesh)) {
        std::cerr << "[ERROR] Failed to load " << args.mesh << "\n";
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::cout << "[INFO] Mesh loaded in "
              << std::chrono::duration<double>(t_load1 - t_load0).count() << " s\n";

    // BBox with margin
    Eigen::Vector3d bmin = sdf.getVertices().colwise().minCoeff();
    Eigen::Vector3d bmax = sdf.getVertices().colwise().maxCoeff();
    Eigen::Vector3d size = bmax - bmin;
    bmin -= args.margin * size;
    bmax += args.margin * size;

    // Prepare query points
    std::vector<Eigen::RowVector3d> pts;
    pts.reserve(args.num);

    if (args.mode == "grid") {
        // approximate cubic grid count
        size_t n = (size_t)std::ceil(std::cbrt((double)args.num));
        if (n < 2) n = 2;
        for (size_t ix=0; ix<n && pts.size()<args.num; ++ix) {
            for (size_t iy=0; iy<n && pts.size()<args.num; ++iy) {
                for (size_t iz=0; iz<n && pts.size()<args.num; ++iz) {
                    double fx = (double)ix / (double)(n-1);
                    double fy = (double)iy / (double)(n-1);
                    double fz = (double)iz / (double)(n-1);
                    Eigen::RowVector3d p;
                    p.x() = bmin.x() + fx * (bmax.x() - bmin.x());
                    p.y() = bmin.y() + fy * (bmax.y() - bmin.y());
                    p.z() = bmin.z() + fz * (bmax.z() - bmin.z());
                    pts.push_back(p);
                }
            }
        }
    } else {
        std::mt19937 rng(args.seed);
        std::uniform_real_distribution<double> ux(bmin.x(), bmax.x());
        std::uniform_real_distribution<double> uy(bmin.y(), bmax.y());
        std::uniform_real_distribution<double> uz(bmin.z(), bmax.z());
        for (size_t i=0; i<args.num; ++i) {
            pts.emplace_back(ux(rng), uy(rng), uz(rng));
        }
    }
    std::cout << "[INFO] Generated " << pts.size() << " query points\n";

    // Warmup (avoid one-time costs in timed section)
    {
        double d; Eigen::RowVector3d cp;
        size_t W = std::min<size_t>(1000, pts.size());
        for (size_t i=0; i<W; ++i) (void)sdf.queryDistance(pts[i], d, cp);
    }

    // Single-thread baseline (useful even when threads>1)
    {
        double d; Eigen::RowVector3d cp;
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i=0; i<pts.size(); ++i) {
            (void)sdf.queryDistance(pts[i], d, cp);
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double qps = (sec > 0) ? pts.size() / sec : 0.0;
        double us_per = (sec > 0) ? (1e6 * sec / pts.size()) : 0.0;
        std::cout << "[1T]   total " << sec << " s  |  " << qps << " qps  |  "
                  << us_per << " us/query\n";
    }

    // Parallel (if threads > 1). NOTE: Only valid if SDFfromPLY::queryDistance is thread-safe.
    if (args.threads > 1) {
        std::vector<double> dists(pts.size());
        std::vector<Eigen::RowVector3d> cps(pts.size());

        auto t0 = std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)pts.size(); ++i) {
            double d; Eigen::RowVector3d cp;
            sdf.queryDistance(pts[(size_t)i], d, cp);
            dists[(size_t)i] = d;
            cps[(size_t)i]   = cp;
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double qps = (sec > 0) ? pts.size() / sec : 0.0;
        double us_per = (sec > 0) ? (1e6 * sec / pts.size()) : 0.0;
        std::cout << "[" << args.threads << "T] total " << sec << " s  |  "
                  << qps << " qps  |  " << us_per << " us/query\n";
    }

    std::cout << "[DONE]\n";
    return 0;
}
