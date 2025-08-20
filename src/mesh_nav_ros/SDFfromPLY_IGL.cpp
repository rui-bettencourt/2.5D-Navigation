// #include <igl/signed_distance.h>
// #include <igl/read_triangle_mesh.h>
// #include <igl/writePLY.h>

// #include <Eigen/Dense>
// #include <vector>
// #include <chrono>
// #include <fstream>
// #include <array>
// #include <iostream>

// int main() {
//     Eigen::MatrixXd V;
//     Eigen::MatrixXi F;
//     double max_distance = 1;

//     // Load PLY
//     auto t1 = std::chrono::high_resolution_clock::now();
//     // loadPLY("/home/rui/ds/testsiros2025/obstacles.ply", V, F);
//     // loadPLY("/home/rui/ds/testscilindros/easy/obstacles.ply", V, F);
//     igl::read_triangle_mesh("/home/rui/ds/testsiros2025/obstacles.ply", V, F);
//     auto t2 = std::chrono::high_resolution_clock::now();
//     std::cout << "Load PLY time: " << (t2 - t1).count() * 1000 << " ms" << std::endl;

//     // Prepare a query point
//     Eigen::MatrixXd P(1, 3);
//     P << 0.0, 0.0, 1.0;

//     Eigen::VectorXd S;         // Signed distances
//     Eigen::VectorXd sqrD;      // Squared distances
//     Eigen::MatrixXd C;         // Closest points (Nx3)
//     Eigen::VectorXi I;         // Face indices
//     // Eigen::MatrixXd FN;        // Face normal vectors


//     // Measure time to compute distance
//     t1 = std::chrono::high_resolution_clock::now();
//     igl::AABB<Eigen::MatrixXd,3> tree;
//     igl::FastWindingNumberBVH fwn_bvh;

//     igl::point_mesh_squared_distance(V, V, F, sqrD, I, C);
//     max_distance = sqrt(sqrD.maxCoeff());
//     // Precompute signed distance AABB tree for Pseudonormal method
//     tree.init(V,F);

//     // Precompute vertex,edge and face normals
//     // igl::per_face_normals(V,F,FN);
//     // igl::per_vertex_normals(
//     //     V,F,igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,FN,VN);
//     // igl::per_edge_normals(
//     //     V,F,igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,FN,EN,E,EMAP);

//     // fast winding number setup (just init fwn bvh)
//     igl::fast_winding_number(V, F, 2, fwn_bvh);
//     t2 = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<double> duration = t2 - t1;
//     std::cout << "Signed distance computing time: " << duration.count() * 1000 << " ms" << std::endl;

//     // Measure time to compute distance
//     t1 = std::chrono::high_resolution_clock::now();

//     //P,V,F,sign_type,lower_bound,upper_bound,S,I,C,N)
//     // igl::signed_distance(P, V, F,
//     // igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
//     // S, sqrD, C, I);
//     igl::signed_distance_fast_winding_number(P, V, F, tree, fwn_bvh, S);


//     t2 = std::chrono::high_resolution_clock::now();

//     duration = t2 - t1;
//     std::cout << "Signed distance query time: " << duration.count() * 1000 << " ms" << std::endl;

//     std::cout << "Distance: " << S(0) << std::endl;
//     std::cout << "Closest Point: " << C(0, 0) << ", " << C(0, 1) << ", " << C(0, 2) << std::endl;

//     return 0;
// }





#include "mesh_nav/SDFfromPLY.h"
#include <iostream>

int main()
{
    const std::string ply_file = "/home/rui/ds/testsiros2025/obstacles.ply";
    SDFfromPLY sdf;

    if (!sdf.loadMesh(ply_file))
    {
        std::cerr << "Failed to load mesh." << std::endl;
        return -1;
    }

    // Example query point
    Eigen::RowVector3d query_point(0.0, 0.0, 1.0);

    double signed_distance;
    Eigen::RowVector3d closest_point;

    if (sdf.queryDistance(query_point, signed_distance, closest_point))
    {
        std::cout << "Query Point: " << query_point << std::endl;
        std::cout << "Signed Distance: " << signed_distance << std::endl;
        std::cout << "Closest Point on Mesh: " << closest_point << std::endl;
    }
    else
    {
        std::cerr << "Query failed." << std::endl;
    }

    return 0;
}