#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/Interpolation.h>

#include <iostream>
#include <vector>

int main() {
    openvdb::initialize();

    using namespace openvdb;
    using Vec3s = openvdb::math::Vec3s;
    using Vec3I = openvdb::math::Vec3ui;
    using Vec4I = openvdb::math::Vec4ui;

    // 1. Simple triangle mesh: one triangle (in XY plane)
    std::vector<Vec3s> points = {
        Vec3s(0, 0, 0),
        Vec3s(1, 0, 0),
        Vec3s(0, 1, 0)
    };

    std::vector<Vec3I> triangles = {
        Vec3I(0, 1, 2)
    };

    std::vector<Vec4I> quads; // Empty, no quads

    // 2. Create transform (voxel size)
    float voxelSize = 0.05f;
    math::Transform::Ptr transform = math::Transform::createLinearTransform(voxelSize);

    // 3. Convert mesh to SDF grid
    float exterior = 50.0f, interior = 5.0f;
    FloatGrid::Ptr sdfGrid = tools::meshToSignedDistanceField<FloatGrid>(
        *transform, points, triangles, quads, exterior, interior);

    // 4. Set up interpolator
    tools::GridSampler<FloatGrid::ConstAccessor, tools::BoxSampler> sampler(
        sdfGrid->getConstAccessor(), sdfGrid->transform());

    // 5. Query a point in world space
    math::Vec3d queryPoint(0.0, 0.0, 1.0);

    float distance = sampler.wsSample(queryPoint);

    // Manually compute gradient using central differences
    double h = 1e-3;
    float dx = sampler.wsSample(queryPoint + math::Vec3d(h, 0, 0)) - sampler.wsSample(queryPoint - math::Vec3d(h, 0, 0));
    float dy = sampler.wsSample(queryPoint + math::Vec3d(0, h, 0)) - sampler.wsSample(queryPoint - math::Vec3d(0, h, 0));
    float dz = sampler.wsSample(queryPoint + math::Vec3d(0, 0, h)) - sampler.wsSample(queryPoint - math::Vec3d(0, 0, h));

    math::Vec3d gradient(dx, dy, dz);
    gradient.normalize();

    math::Vec3d vectorToSurface = -distance * gradient;

    // 6. Output
    std::cout << "Query Point: " << queryPoint << std::endl;
    std::cout << "SDF Distance: " << distance << std::endl;
    std::cout << "Gradient (approx.): " << gradient << std::endl;
    std::cout << "Vector to surface: " << vectorToSurface << std::endl;

    return 0;
}
