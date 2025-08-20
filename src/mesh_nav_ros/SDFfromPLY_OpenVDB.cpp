#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <tinyply.h>

#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <memory>

openvdb::FloatGrid::Ptr meshToSDF(const std::string& plyPath, float voxelSize = 0.05f, float exteriorBandWidth = 5.0f, float interiorBandWidth = 3.0f)
{
    std::cout << "[INFO] Reading PLY file: " << plyPath << std::endl;

    std::ifstream file(plyPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file: " << plyPath << std::endl;
        return nullptr;
    }

    tinyply::PlyFile plyFile;
    try {
        plyFile.parse_header(file);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to parse PLY header: " << e.what() << std::endl;
        return nullptr;
    }

    std::shared_ptr<tinyply::PlyData> vertices, faces;
    try {
        vertices = plyFile.request_properties_from_element("vertex", { "x", "y", "z" });
        faces = plyFile.request_properties_from_element("face", { "vertex_indices" });
        plyFile.read(file);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to read PLY data: " << e.what() << std::endl;
        return nullptr;
    }

    std::cout << "[INFO] Number of vertices: " << vertices->count << std::endl;

    std::vector<openvdb::Vec3s> points;
    if (vertices->t == tinyply::Type::FLOAT32) {
        const float* raw = reinterpret_cast<const float*>(vertices->buffer.get());
        for (size_t i = 0; i < vertices->count; ++i)
            points.emplace_back(raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]);
    } else if (vertices->t == tinyply::Type::FLOAT64) {
        const double* raw = reinterpret_cast<const double*>(vertices->buffer.get());
        for (size_t i = 0; i < vertices->count; ++i)
            points.emplace_back(static_cast<float>(raw[i * 3]),
                                static_cast<float>(raw[i * 3 + 1]),
                                static_cast<float>(raw[i * 3 + 2]));
    } else {
        std::cerr << "[ERROR] Unsupported vertex type." << std::endl;
        return nullptr;
    }

    std::cout << "[INFO] Total face buffer size (bytes): " << faces->buffer.size_bytes() << std::endl;

    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;  // required for meshToSignedDistanceField
    const uint8_t* ptr = faces->buffer.get();
    size_t bytesRead = 0;
    size_t triangleCount = 0;

    while (bytesRead < faces->buffer.size_bytes()) {
        uint8_t vertexCount = *ptr++; bytesRead += 1;

        if (vertexCount == 3) {
            uint32_t v0 = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
            uint32_t v1 = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
            uint32_t v2 = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
            triangles.emplace_back(v0, v1, v2);
            bytesRead += 12;
        }
        else if (vertexCount == 4) {
            uint32_t v0 = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
            uint32_t v1 = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
            uint32_t v2 = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
            uint32_t v3 = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
            quads.emplace_back(v0, v1, v2, v3);
            bytesRead += 16;
        }
        else {
            ptr += vertexCount * sizeof(uint32_t);
            bytesRead += vertexCount * sizeof(uint32_t);
        }
    }

    std::cout << "[INFO] Triangle count: " << triangles.size() << std::endl;
    std::cout << "[INFO] Quad count: " << quads.size() << std::endl;

    openvdb::initialize();

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(voxelSize);

    openvdb::FloatGrid::Ptr sdfGrid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
        *transform,
        points,
        triangles,
        quads,
        exteriorBandWidth,
        interiorBandWidth
    );

    std::cout << "[INFO] SDF grid created successfully." << std::endl;
    return sdfGrid;
}


int main() {
    std::string ply_file = "/home/rui/ds/testsiros2025/obstacles.ply";

    try {
        auto sdf = meshToSDF(ply_file);

        std::cout << "SDF successfully computed!" << std::endl;

        // Example: query SDF at a world-space location
        openvdb::Vec3d queryPoint(0.0, 0.0, 0.0);
        auto accessor = sdf->getConstAccessor();
        openvdb::Coord ijk = sdf->transform().worldToIndexCellCentered(queryPoint);
        float distance = accessor.getValue(ijk);

        std::cout << "Signed distance at " << queryPoint << " is: " << distance << std::endl;

        // Optional: Save the grid to a .vdb file
        openvdb::io::File file("output.vdb");
        openvdb::GridPtrVec grids;
        grids.push_back(sdf);
        file.write(grids);
        file.close();
        std::cout << "Saved SDF to output.vdb" << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}