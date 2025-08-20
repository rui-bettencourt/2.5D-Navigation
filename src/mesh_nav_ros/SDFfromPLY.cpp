    #include "mesh_nav/SDFfromPLY.h"

    bool SDFfromPLY::loadMesh(const std::string& ply_filename)
    {
        if (!igl::read_triangle_mesh(ply_filename, V_, F_))
        {
            std::cerr << "Failed to load mesh: " << ply_filename << std::endl;
            return false;
        }

        // Build AABB tree for closest point queries
        tree_.init(V_, F_);

        // Initialize Fast Winding Number BVH (using default depth=2)
        igl::fast_winding_number(V_, F_, 2, fwn_bvh_);

        std::cout << "Mesh loaded and fast winding number structures initialized." << std::endl;
        return true;
    }

    bool SDFfromPLY::queryDistance(
        const Eigen::RowVector3d& query_point,
        double& distance,
        Eigen::RowVector3d& closest_point) const
    {
        // auto t1 = std::chrono::high_resolution_clock::now();
        if (V_.rows() == 0 || F_.rows() == 0)
        {
            std::cerr << "Mesh not loaded." << std::endl;
            return false;
        }
        // auto t2 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> duration = t2 - t1;
        // t1 = std::chrono::high_resolution_clock::now();

        // Wrap query_point into a 1x3 matrix for libigl calls
        // Instead of creating matrix every time, reuse a member or static matrix if possible
        Eigen::Matrix<double, 1, 3> P = query_point;

        Eigen::VectorXd S;         // Signed distance output
        Eigen::VectorXd sqrD;      // Squared distances output
        Eigen::MatrixXd C;         // Closest points output
        Eigen::VectorXi I;         // Face indices output (unused here)

        // t2 = std::chrono::high_resolution_clock::now();
        // duration = t2 - t1;
        // std::cout << "Initializations query time: " << duration.count() * 1000 << " ms" << std::endl;

        // t1 = std::chrono::high_resolution_clock::now();
        // Compute signed distance using fast winding number
        igl::signed_distance_fast_winding_number(P, V_, F_, tree_, fwn_bvh_, S);
        // t2 = std::chrono::high_resolution_clock::now();
        // duration = t2 - t1;
        // std::cout << "FWN query time: " << duration.count() * 1000 << " ms" << std::endl;

        // t1 = std::chrono::high_resolution_clock::now();
        // Compute closest point on mesh (unsigned)
        igl::point_mesh_squared_distance(P, V_, F_, sqrD, I, C);
        // t2 = std::chrono::high_resolution_clock::now();
        // duration = t2 - t1;
        // std::cout << "Find closest point query time: " << duration.count() * 1000 << " ms" << std::endl;

        distance = S(0);
        closest_point = C.row(0);

        // t2 = std::chrono::high_resolution_clock::now();
        // duration = t2 - t1;
        // std::cout << "Signed distance query time: " << duration.count() * 1000 << " ms" << std::endl;

        return true;
    }
