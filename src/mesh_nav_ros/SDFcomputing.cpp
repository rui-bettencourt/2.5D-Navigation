#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <atomic>
#include <thread>   // <-- progress thread
#include <cstdlib>

#include <Eigen/Core>
#include <omp.h>

#include "mesh_nav/SDFfromPLY.h"

// ----------------- helpers -----------------
static inline size_t idx(int i, int j, int k, int ny, int nz) {
    return static_cast<size_t>(i) * static_cast<size_t>(ny) * static_cast<size_t>(nz)
         + static_cast<size_t>(j) * static_cast<size_t>(nz)
         + static_cast<size_t>(k);
}

static std::string human_bytes(double bytes) {
    const char* units[] = {"B","KB","MB","GB","TB"};
    int u = 0;
    while (bytes >= 1024.0 && u < 4) { bytes /= 1024.0; ++u; }
    std::ostringstream os;
    os << std::fixed << std::setprecision(bytes >= 10 ? 1 : 2) << bytes << " " << units[u];
    return os.str();
}

static std::string format_hms(double sec) {
    if (sec < 0 || !std::isfinite(sec)) return "--:--:--";
    uint64_t s = static_cast<uint64_t>(sec + 0.5);
    uint64_t h = s / 3600; s %= 3600;
    uint64_t m = s / 60;   s %= 60;
    std::ostringstream os;
    os << std::setfill('0') << std::setw(2) << h << ":"
       << std::setw(2) << m << ":" << std::setw(2) << s;
    return os.str();
}

struct Args {
    std::string mesh_file = "/home/rui/ds/small-house/obstacles.ply";
    std::string out_file  = "/home/rui/ds/small-house/sdf.bin";
    double voxel          = 0.1;
    double margin_frac    = 0.20;   // 20% bbox margin
    int threads           = 6;     // -1 -> use OMP default (usually all cores)
    double progress_secs  = 2.0;    // seconds between progress prints
};


static void usage(const char* prog) {
    std::cerr <<
      "Usage: " << prog << " [options]\n"
      "  --mesh PATH                 Input PLY mesh\n"
      "  --out  PATH                 Output SDF .bin\n"
      "  --voxel FLOAT               Voxel size (m), default 0.02\n"
      "  --margin FLOAT              Fractional bbox margin, default 0.2\n"
      "  --threads INT               Cap OpenMP threads (default: OMP default)\n"
      "  --progress-interval FLOAT   Progress print interval seconds (default 2.0)\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* what)->const char*{
            if (i+1 >= argc) { std::cerr << "Missing value for " << what << "\n"; std::exit(2); }
            return argv[++i];
        };
        if      (k == "--mesh")               a.mesh_file = need("--mesh");
        else if (k == "--out")                a.out_file  = need("--out");
        else if (k == "--voxel")              a.voxel     = std::atof(need("--voxel"));
        else if (k == "--margin")             a.margin_frac = std::atof(need("--margin"));
        else if (k == "--threads")            a.threads   = std::atoi(need("--threads"));
        else if (k == "--progress-interval")  a.progress_secs = std::atof(need("--progress-interval"));
        else if (k == "--help" || k == "-h")  { usage(argv[0]); return false; }
        else { std::cerr << "Unknown option: " << k << "\n"; usage(argv[0]); return false; }
    }
    return true;
}

int main(int argc, char** argv)
{
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    if (args.threads > 0) {
        omp_set_num_threads(args.threads);
    }
    int used_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        used_threads = omp_get_num_threads();
    }

    std::cout << "[INFO] Using " << used_threads << " OpenMP thread(s)\n";
    std::cout << "[INFO] Mesh:  " << args.mesh_file << "\n";
    std::cout << "[INFO] Out:   " << args.out_file << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "[INFO] Voxel: " << args.voxel << " m, margin: " << args.margin_frac*100.0 << "%\n";

    // Load mesh
    SDFfromPLY sdfFromPLY;
    if (!sdfFromPLY.loadMesh(args.mesh_file)) {
        std::cerr << "[ERROR] Failed to load mesh: " << args.mesh_file << "\n";
        return 1;
    }
    std::cout << "[INFO] Mesh loaded and fast winding structures ready.\n";

    // BBox with margin
    Eigen::Vector3d bbox_min = sdfFromPLY.getVertices().colwise().minCoeff();
    Eigen::Vector3d bbox_max = sdfFromPLY.getVertices().colwise().maxCoeff();
    Eigen::Vector3d bbox_size = bbox_max - bbox_min;

    bbox_min -= args.margin_frac * bbox_size;
    bbox_max += args.margin_frac * bbox_size;
    Eigen::Vector3d bbox_center = 0.5 * (bbox_min + bbox_max);

    // Grid
    const double voxel_size = args.voxel;
    const int nx = static_cast<int>((bbox_max.x() - bbox_min.x()) / voxel_size) + 1;
    const int ny = static_cast<int>((bbox_max.y() - bbox_min.y()) / voxel_size) + 1;
    const int nz = static_cast<int>((bbox_max.z() - bbox_min.z()) / voxel_size) + 1;

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        std::cerr << "[ERROR] Invalid grid dims (" << nx << "," << ny << "," << nz << ")\n";
        return 1;
    }

    std::cout << "[INFO] Grid dims: " << nx << " x " << ny << " x " << nz << "\n";
    const size_t total_voxels = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    std::cout << "[INFO] Total voxels: " << total_voxels << "\n";

    const Eigen::Vector3d grid_origin =
        bbox_center - 0.5 * Eigen::Vector3d((nx - 1) * voxel_size,
                                            (ny - 1) * voxel_size,
                                            (nz - 1) * voxel_size);

    // Allocate
    std::vector<float> sdf_grid(total_voxels, 0.f);
    std::vector<float> vec_grid(total_voxels * 3ULL, 0.f);

    const double bytes_sdf = static_cast<double>(sdf_grid.size()) * sizeof(float);
    const double bytes_vec = static_cast<double>(vec_grid.size()) * sizeof(float);
    std::cout << "[INFO] Estimated memory: SDF " << human_bytes(bytes_sdf)
              << " + VEC " << human_bytes(bytes_vec)
              << " = " << human_bytes(bytes_sdf + bytes_vec) << "\n";

    // Progress/Timing
    std::atomic<size_t> counter{0};
    std::atomic<bool>   done{false};
    auto t0 = std::chrono::steady_clock::now();
    const double progress_interval = std::max(0.2, args.progress_secs);

    // ---- Progress thread: prints every N seconds, independent of voxel stride ----
    std::thread reporter([&](){
        using namespace std::chrono;
        while (!done.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(duration<double>(progress_interval));
            size_t c = counter.load(std::memory_order_relaxed);
            auto now = steady_clock::now();
            double elapsed = duration<double>(now - t0).count();
            double rate = (elapsed > 0) ? (static_cast<double>(c) / elapsed) : 0.0; // vox/s
            double eta  = (rate > 0) ? (static_cast<double>(total_voxels - c) / rate) : NAN;
            double pct  = (100.0 * static_cast<double>(c)) / static_cast<double>(total_voxels);

            std::ostringstream line;
            line << "[PROGRESS] " << std::fixed << std::setprecision(1) << pct << "%  "
                 << c << "/" << total_voxels
                 << "  |  " << std::setprecision(0) << rate << " vox/s"
                 << "  |  elapsed " << format_hms(elapsed)
                 << "  |  ETA " << format_hms(eta);
            std::cout << "\r" << line.str() << "    " << std::flush;
        }
    });

    std::cout << "[INFO] Starting voxel sweep...\n";

    // Main sweep
    #pragma omp parallel for collapse(3) schedule(dynamic, 4)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const double x = grid_origin.x() + i * voxel_size;
                const double y = grid_origin.y() + j * voxel_size;
                const double z = grid_origin.z() + k * voxel_size;

                Eigen::RowVector3d pt(x, y, z);
                double dist;
                Eigen::RowVector3d closest_pt;

                bool ok = sdfFromPLY.queryDistance(pt, dist, closest_pt);
                if (!ok) { dist = 0.0; closest_pt = pt; }

                const float signed_dist = static_cast<float>(dist);
                const Eigen::RowVector3f vec_to_surface = (pt - closest_pt).cast<float>();

                const size_t lin = idx(i, j, k, ny, nz);
                sdf_grid[lin] = signed_dist;
                vec_grid[3 * lin + 0] = vec_to_surface.x();
                vec_grid[3 * lin + 1] = vec_to_surface.y();
                vec_grid[3 * lin + 2] = vec_to_surface.z();

                ++counter; // atomic
            }
        }
    }

    // Signal reporter and finalize print
    done.store(true, std::memory_order_relaxed);
    reporter.join();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double rate = (elapsed > 0) ? (static_cast<double>(total_voxels) / elapsed) : 0.0;

    std::cout << "\r[PROGRESS] 100.0%  " << total_voxels << "/" << total_voxels
              << "  |  " << std::fixed << std::setprecision(0) << rate << " vox/s"
              << "  |  elapsed " << format_hms(elapsed)
              << "  |  ETA 00:00:00          \n";

    std::cout << "[INFO] Voxel grid generation done. Writing output...\n";

    // Save
    std::ofstream ofs(args.out_file, std::ios::binary);
    if (!ofs) {
        std::cerr << "[ERROR] Failed to open output file: " << args.out_file << "\n";
        return 1;
    }
    ofs.write(reinterpret_cast<const char*>(&grid_origin.x()), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&grid_origin.y()), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&grid_origin.z()), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&voxel_size), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&nx), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&ny), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&nz), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(sdf_grid.data()), sdf_grid.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(vec_grid.data()), vec_grid.size() * sizeof(float));
    ofs.close();

    std::cout << "[OK] Wrote: " << args.out_file << "\n";
    return 0;
}
