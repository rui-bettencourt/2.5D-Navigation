#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <Eigen/Dense>
#include "mesh_nav/RobotKinematics.h"

// ----------- helpers: RSS reading (Linux) -----------
static long long readVmRSS_kB() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) { // starts with "VmRSS:"
      std::istringstream iss(line);
      std::string key, unit;
      long long value_kb = 0;
      iss >> key >> value_kb >> unit; // "VmRSS:" <value> "kB"
      return value_kb;                // kB
    }
  }
  return -1; // not found
}
static std::string fmtBytes(long long bytes) {
  std::ostringstream oss;
  if (bytes < 0) { oss << "N/A"; return oss.str(); }
  const double kb = bytes / 1024.0;
  const double mb = kb / 1024.0;
  if (mb >= 1.0)      oss << mb << " MB";
  else if (kb >= 1.0) oss << kb << " KB";
  else                oss << bytes << " B";
  return oss.str();
}

// ----------- your TLS helper (unchanged logic) -----------
static inline RobotKinematics& tlsRobot(const RobotKinematics* src) {
  struct Slot {
    const RobotKinematics* key = nullptr;
    std::unique_ptr<RobotKinematics> inst;
  };
  thread_local Slot slot;

  if (!src) throw std::runtime_error("tlsRobot: null RobotKinematics*");

  if (slot.key != src || !slot.inst) {
    slot.key  = src;
    slot.inst = std::make_unique<RobotKinematics>(*src); // deep copy
  }
  return *slot.inst;
}

static double maxAbsDiff(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  if (A.rows()!=B.rows() || A.cols()!=B.cols()) return 1e9;
  return (A - B).cwiseAbs().maxCoeff();
}

int main() {
  try {
    std::vector<std::string> joint_names = {
      "arm_1_joint","arm_2_joint","arm_3_joint",
      "arm_4_joint","arm_5_joint","arm_6_joint","arm_7_joint"
    };

    RobotKinematics rk_orig(
      "/home/rui/socrob_ws/src/isr_tiago/simulation/mbot_simulation_environments/robots/tiago_ouster.urdf",
      "base_link",
      "arm_7_link",
      joint_names
    );

    rk_orig.debugPrint("orig");

    const int dof = rk_orig.getDOF();
    if (dof <= 0) {
      std::cerr << "manipDof() <= 0\n";
      return 1;
    }

    // Non-trivial pose for consistency
    rk_orig.setRobotPose(0.5, -0.2, 0.0, 0.1, -0.05, 0.3);

    // Reasonable q
    Eigen::VectorXd q = Eigen::VectorXd::Zero(dof);
    for (int i=0;i<dof;i++) q[i] = 0.1*(i+1);

    const int jidx = std::min(3, dof);
    auto J_orig = rk_orig.computeJacobian(q, jidx);

    // ----- Measure TLS clone creation (first use in main thread) -----
    {
      const long long rss0_kb = readVmRSS_kB();
      auto t0 = std::chrono::steady_clock::now();

      auto& tls_main = tlsRobot(&rk_orig); // creation happens here (first call in this thread)

      auto t1 = std::chrono::steady_clock::now();
      const long long rss1_kb = readVmRSS_kB();

      // Warm up: set pose & compute Jacobian to trigger any lazy allocs
      tls_main.setRobotPose(0.5, -0.2, 0.0, 0.1, -0.05, 0.3);
      auto J_tls_main = tls_main.computeJacobian(q, jidx);
      const long long rss2_kb = readVmRSS_kB();

      std::cout << "\n== TLS copy (main thread) ==\n";
      std::cout << "Model addr: orig=" << rk_orig.modelAddress()
                << " tls_main=" << tls_main.modelAddress() << "\n";
      std::cout << "Data  addr: orig=" << rk_orig.dataAddress()
                << " tls_main=" << tls_main.dataAddress() << "\n";
      std::cout << "J diff main: " << maxAbsDiff(J_orig, J_tls_main) << "\n";

      const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      const long long delta_create_bytes = (rss1_kb - rss0_kb) * 1024LL;
      const long long delta_warm_bytes   = (rss2_kb - rss1_kb) * 1024LL;

      std::cout << "TLS create time: " << dt_ms << " ms\n";
      std::cout << "RSS +after create: " << fmtBytes(delta_create_bytes) << "\n";
      std::cout << "RSS +after warmup (J): " << fmtBytes(delta_warm_bytes) << "\n";
    }

    // ----- Also measure a direct heap copy (not TLS) -----
    {
      const long long rss0_kb = readVmRSS_kB();
      auto t0 = std::chrono::steady_clock::now();

      auto rk_copy = std::make_unique<RobotKinematics>(rk_orig); // deep copy via copy-ctor

      auto t1 = std::chrono::steady_clock::now();
      const long long rss1_kb = readVmRSS_kB();

      rk_copy->setRobotPose(0.5, -0.2, 0.0, 0.1, -0.05, 0.3);
      auto J_copy = rk_copy->computeJacobian(q, jidx);
      const long long rss2_kb = readVmRSS_kB();

      std::cout << "\n== Direct copy (heap, main thread) ==\n";
      std::cout << "Model addr: copy=" << rk_copy->modelAddress() << "\n";
      std::cout << "Data  addr: copy=" << rk_copy->dataAddress()  << "\n";
      std::cout << "J diff copy: " << maxAbsDiff(J_orig, J_copy) << "\n";

      const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      const long long delta_create_bytes = (rss1_kb - rss0_kb) * 1024LL;
      const long long delta_warm_bytes   = (rss2_kb - rss1_kb) * 1024LL;

      std::cout << "Direct copy time: " << dt_ms << " ms\n";
      std::cout << "RSS +after create: " << fmtBytes(delta_create_bytes) << "\n";
      std::cout << "RSS +after warmup (J): " << fmtBytes(delta_warm_bytes) << "\n";
    }

    // ----- Spawn another OS thread, measure its TLS clone too -----
    std::thread th([&](){
      const long long rss0_kb = readVmRSS_kB();
      auto t0 = std::chrono::steady_clock::now();

      auto& tls_th = tlsRobot(&rk_orig); // first use in this thread -> create

      auto t1 = std::chrono::steady_clock::now();
      const long long rss1_kb = readVmRSS_kB();

      tls_th.setRobotPose(0.5, -0.2, 0.0, 0.1, -0.05, 0.3);
      auto J_tls_th = tls_th.computeJacobian(q, jidx);
      const long long rss2_kb = readVmRSS_kB();

      std::cout << "\n== TLS copy (worker thread) ==\n";
      std::cout << "Model addr: tls_thread=" << tls_th.modelAddress() << "\n";
      std::cout << "Data  addr: tls_thread=" << tls_th.dataAddress()  << "\n";
      std::cout << "J diff thread: " << maxAbsDiff(J_orig, J_tls_th) << "\n";

      const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      const long long delta_create_bytes = (rss1_kb - rss0_kb) * 1024LL;
      const long long delta_warm_bytes   = (rss2_kb - rss1_kb) * 1024LL;

      std::cout << "TLS create time (thread): " << dt_ms << " ms\n";
      std::cout << "RSS +after create (thread): " << fmtBytes(delta_create_bytes) << "\n";
      std::cout << "RSS +after warmup (J) (thread): " << fmtBytes(delta_warm_bytes) << "\n";
    });
    th.join();

    std::cout << "\nOK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << "\n";
    return 1;
  }
}
