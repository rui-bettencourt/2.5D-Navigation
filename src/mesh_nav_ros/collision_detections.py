import numpy as np
import fcl
import time
import open3d as o3d

def is_colliding_fcl(mesh_robot, env):
    start_time = time.time()
    # conversion of open3d mesh to fcl model of the robot
    robot_model = fcl.BVHModel()
    robot_verts = np.asarray(mesh_robot.vertices)
    robot_triangles = np.asarray(mesh_robot.triangles)
    robot_model.beginModel(len(robot_verts), len(robot_triangles))
    robot_model.addSubModel(robot_verts, robot_triangles)
    robot_model.endModel()
    t_robot = fcl.Transform()
    robot_object = fcl.CollisionObject(robot_model, t_robot)


    # check if the environment is represented through mesh or octomap
    if isinstance(env, o3d.geometry.TriangleMesh):
        #conversion of open3d mesh to fcl model of the environment
        env_model = fcl.BVHModel()
        env_verts = np.asarray(env.vertices)
        env_triangles = np.asarray(env.triangles)
        env_model.beginModel(len(env_verts), len(env_triangles))
        env_model.addSubModel(env_verts, env_triangles)
        env_model.endModel()
    elif isinstance(env, o3d.geometry.PointCloud):
        resolution = 0.1
        points = np.asarray(env.points)


        env_model = fcl.OcTree(resolution, points)
    else:
        print("error: environment variable no mesh or octomap")
        exit()

    t_env = fcl.Transform()
    env_object = fcl.CollisionObject(env_model, t_env)

    # collision check
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    ret = fcl.collide(robot_object, env_object, request, result)
    print("FCL - Intersecting: " + str(bool(ret)) + " | Took " + str(time.time()-start_time) + " s")
    return ret

def is_colliding_fcl_octomap(mesh_robot, env):
    start_time = time.time()
    # conversion of open3d mesh to fcl model of the robot
    robot_model = fcl.BVHModel()
    robot_verts = np.asarray(mesh_robot.vertices)
    robot_triangles = np.asarray(mesh_robot.triangles)
    robot_model.beginModel(len(robot_verts), len(robot_triangles))
    robot_model.addSubModel(robot_verts, robot_triangles)
    robot_model.endModel()
    t_robot = fcl.Transform()
    robot_object = fcl.CollisionObject(robot_model, t_robot)


    # check if the environment is represented through mesh or octomap
    if isinstance(env, o3d.geometry.PointCloud):
        resolution = 0.09
        points = np.asarray(env.points)

        collision_objects_octomap = [robot_object]
        collision_geoms_octomap = [robot_model]
        collision_names = ['robot']
        for i, point in enumerate(points):
            b = fcl.Box(resolution,resolution,resolution)
            t_b = fcl.Transform(point)
            o_b = fcl.CollisionObject(b, t_b)
            collision_objects_octomap.append(o_b)
            collision_geoms_octomap.append(b)
            collision_names.append('octomap'+str(i))
    else:
        print("error: environment variable no mesh or octomap")
        exit()


    # Create map from geometry IDs to objects
    geom_id_to_obj = { id(geom) : obj for geom, obj in zip(collision_geoms_octomap, collision_objects_octomap) }

    # Create map from geometry IDs to string names
    geom_id_to_name = { id(geom) : name for geom, name in zip(collision_geoms_octomap, collision_names) }

    # Create manager
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(collision_objects_octomap)
    manager.setup()

    # Create collision request structure
    crequest = fcl.CollisionRequest(num_max_contacts=100, enable_contact=False)
    cdata = fcl.CollisionData(crequest, fcl.CollisionResult())

    # Run collision request
    manager.collide(cdata, fcl.defaultCollisionCallback)

    # Extract collision data from contacts and use that to infer set of
    # objects that are in collision
    objs_in_collision = set()

    collision = False

    for contact in cdata.result.contacts:
        # Extract collision geometries that are in contact
        coll_geom_0 = contact.o1
        coll_geom_1 = contact.o2

        # Get their names
        coll_names = [geom_id_to_name[id(coll_geom_0)], geom_id_to_name[id(coll_geom_1)]]
        coll_names = tuple(sorted(coll_names))
        objs_in_collision.add(coll_names)
        if 'robot' in coll_names:
            collision = True

    print("FCL octomap - Intersecting: " + str(collision) + " | Took " + str(time.time()-start_time) + " s")
    # for coll_pair in objs_in_collision:
    #     print('Object {} in collision with object {}!'.format(coll_pair[0], coll_pair[1]))
    return collision

def is_colliding_o3d(robot_mesh, obstacle_mesh):
    start_time = time.time()
    intersect = robot_mesh.is_intersecting(obstacle_mesh)
    print("O3d - Intersecting: " + str(intersect) + " | Took " + str(time.time()-start_time) + " s")
    return intersect


# if __name__ == '__main__':