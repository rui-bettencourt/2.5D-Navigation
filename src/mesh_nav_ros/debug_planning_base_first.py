#!/usr/bin/env python
# import ros and other libraries
import rospy
import random
import numpy as np
import csv
import math

# import other created files
from robot_mesh_state import RobotMeshState
from collision_detections import *
from graph_functions import GraphManager
from elastic_bands_threads import ElasticBandPlanner
from robot_kinematics import RobotKinematics
# from robot_kinematics_rob_toolbox import RobotKinematics
import time
from aux_functions import interpolate_path, sample_path
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt
import json
from open3d.visualization import rendering
import imageio.v3 as iio  # pip install imageio imageio-ffmpeg

###### variables
k_attraction_base=0.005
k_repulsion_base=0.0002
k_attraction_joints=0.2
k_repulsion_joints=0.2#09
k_repulsion_robot_joints=0.0
k_safety_joints = 0.3 #0.0
closest_obstacle_only = True
k_update_joints=0.2
k_orientation=0.02
k_orientation_from_base=0.0
obstacle_threshold=1.5
manipulator = True
#min_distance_to_obstacle = 0.05

start = {'x': 0.0, 'y': 0.0, 'z': 0.0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'q1': 1.57, 'q2': -0.0, 'q3': 0.0, 'q4': 0.0, 'q5': 0.0, 'q6': 0.0}

goal = {'x': 4.0, 'y': 0.0, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'q1': 0.0, 'q2': -0.0, 'q3': 0.0, 'q4': 0.0, 'q5': 0.0, 'q6': 0.0, 'q7': 0.0}
# safe_config = {'q1': 0.2, 'q2': -1.34, 'q3': -0.2, 'q4': 1.94, 'q5': -1.57, 'q6': 1.37, 'q7': 0.0}
safe_config = {'q1': 0.72, 'q2': -0.9, 'q3': -0.88, 'q4': 1.94, 'q5': -1.2, 'q6': 1.37}
center_activation_safety = 0.8
################

class MeshNav(object):
    def __init__(self):
        # begin node
        rospy.init_node('mesh_nav', anonymous=True)

        # variables
        # self.path = '/home/rui/ds/testscilindros/hard/'     # path for files, change this to save in the package
        self.path = '/home/rui/ds/testsiros2025/'     # path for files, change this to save in the package
        self.rate = rospy.Rate(10)              # TODO: make this an argument of launch file

        # create classes needed for navigation
        # self.Meshes = RobotMeshState()
        self.robot_kinematics = RobotKinematics(manipulator)
        self.Graph = GraphManager(self.path+"traversablegroundgraph.pkl")
        self.RM = RobotMeshState(robot_kinematics=self.robot_kinematics)
        self.EBAND = ElasticBandPlanner(robot_kinematics=self.robot_kinematics, k_attraction_base=k_attraction_base, 
                                        k_repulsion_base=k_repulsion_base, k_attraction_joints=k_attraction_joints,
                                        k_repulsion_joints=k_repulsion_joints, k_repulsion_robot_joints=k_repulsion_robot_joints, k_update_joints=k_update_joints,
                                        k_orientation=k_orientation, k_orientation_from_base=k_orientation_from_base,
                                        k_safety_joints=k_safety_joints, safe_config=safe_config,
                                        obstacle_threshold=obstacle_threshold)
        # self.joint_names = ['base_link','joint1','joint2','joint3','joint4','joint5','joint6']

    def run(self):
        # obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
        # obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'retail_visuals.ply')
        # obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67]) # pink
        obstacle_mesh = self.import_obstacles()
        execution_times = []
        number_of_attempts = 1
        print("Read obstacles!")

        while not rospy.is_shutdown() and number_of_attempts>0:
            # test path planning with random start and end
            start_coord = [start['x'], start['y'], start['z'], start['yaw']]
            end_coord = [goal['x'], goal['y'], goal['yaw']]

            start_time = time.time()

            path = self.Graph.plan(start_coord, end_coord)
            path = sample_path(path, 15) #25
            path[-1]['yaw'] = goal['yaw']
            # self.plot_path_3d(path, obstacle_mesh)
            # check for collisions all points in the path
            
            # complete the path initialization with roll, pitch and qs
            path = interpolate_path(path,start,goal)
            end_time = time.time()
            print("ORIGINAL PATH. Time: ", end_time - start_time)
            self.plot_path_3d(path, obstacle_mesh)
            # if rospy.is_shutdown():
            #     exit()

            # if there is collisions use some sort of elastic band to move away from the collision
            start_time = time.time()
            new_path = self.EBAND.update_path(path, obstacle_mesh, self.RM, 200, convergence_threshold=2e-2)
            # new_path = path
            end_time = time.time()
            print("Execution time:", end_time - start_time)
            # save execution time to a CSV file
            # with open("/home/rui/pcl_ws/src/mesh_nav/data/testscilindros/planning_times.csv", "a") as file:
            #     file.write(str(end_time - start_time) + "\n")

            # self.EBAND.animate_path_evolution(50)
            self.plot_path_3d(new_path, obstacle_mesh)
            with open("/home/rui/pcl_ws/src/mesh_nav/data//testscilindros/test_hard_path2.json", "w") as file:
                json.dump(new_path, file, indent=4)

            # print(self.RM.robot_state.get_pose(), self.RM.robot_state.joints)
            # exit()
            execution_times.append(end_time-start_time)
            number_of_attempts -= 1
            self.rate.sleep()
        execution_times = np.asarray(execution_times)
        
        print("Avg time: ",np.mean(execution_times)," | std : ", np.std(execution_times), " | min: ", np.min(execution_times), " | max: ", np.max(execution_times))

    def plot_path_3d(self, path, obstacle_mesh, save_path=None):
        all_bbs = []
        all_bbs.append(obstacle_mesh)
        colormap= plt.get_cmap('jet')
        # colormap = cm.get_cmap('jet') #('RdYlGn')
        padding = 0.8
        # then use the code of moving the robot to move to each point in the path
        for pose in path:
            # pose_array = np.array([pose['x'], pose['y'], pose['z'], pose['roll'], pose['pitch'], pose['yaw']])
            # conf_bb = self.RM.simulate_move_joints(self.RM.robot_bbs, self.joint_names[0], pose_array) # this is just moving the base to the pose. arm config is the online one
            # joint_poses = []
            # for key in pose.keys():
            #     if key.startswith('q'):
            #         joint_poses.append(pose[key])
            # conf_bb = self.RM.simulate_move_joints(conf_bb, self.joint_names[1:], joint_poses) # move the joints
            meshes = self.RM.update_robot_arm_bbs(pose, move_base = True, local_frame = False)

            # mesh= self.RM.convert_bbs_to_mesh(conf_bb)
            combined_mesh = o3d.geometry.TriangleMesh()
            for mesh in meshes.values():
                combined_mesh += mesh

            bb = combined_mesh.get_axis_aligned_bounding_box()
            bb.scale(2.0,bb.get_center())
            # simplify obstacle mesh
            obstacle_mesh_cropped = obstacle_mesh.crop(bb)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)

            if is_colliding_o3d(combined_mesh, obstacle_mesh_cropped):
                mesh_color = [1.0,0,0]
            else:
                distance, _ = self.EBAND.compute_2mesh_distance(combined_mesh, obstacle_mesh_cropped)
                # print(distance)
                norm_distance = (1-min(distance, obstacle_threshold)/(obstacle_threshold))*padding
                # print(norm_distance)
                # print('----------------')
                mesh_color = np.asarray(colormap(norm_distance))[:3]


            # conf_bb = list(meshes.values())

            # for bb in conf_bb:
            #     bb.color=mesh_color
            # all_bbs.extend(conf_bb)
            combined_mesh.paint_uniform_color(mesh_color)
            combined_mesh.compute_vertex_normals()
            all_bbs.append(combined_mesh)

        if save_path is None:
            o3d.visualization.draw_geometries(all_bbs)
        else:
            self.save_open3d_view(all_bbs, save_path)

    # def import_obstacles(self):
    #     # obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
    #     obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'retail_visuals.ply')
    #     obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67]) # pink
    #     return obstacle_mesh
    def import_obstacles(self):
        # mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
        mesh = o3d.io.read_triangle_mesh(self.path + 'retail_visuals.ply')
        if mesh.is_empty():
            return mesh

        V = np.asarray(mesh.vertices)
        z = V[:, 2]
        zmin, zmax = float(z.min()), float(z.max())

        # Normalize z -> [0,1]
        if zmax == zmin:
            s = np.ones_like(z)
        else:
            s = (z - zmin) / (zmax - zmin)

        # Shades of pink: scale brightness between 0.3 and 1.0
        base = np.array([1.0, 0.72, 0.67], dtype=float)  # pink
        brightness = 0.3 + 0.7 * s                       # 0.3 (dark) .. 1.0 (full)
        colors = np.clip(brightness[:, None] * base[None, :], 0.0, 1.0)

        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    def import_path(self, path_dir):
        with open(path_dir, "r") as file:
            path = json.load(file)
        return path

    

    def import_traj_csv(self, csv_path,
                        num_joints=6,
                        angles_in_degrees=False,
                        default_z=0.0,
                        default_roll=0.0,
                        default_pitch=0.0):
        """
        Read a trajectory CSV and return a list[dict] with keys:
        x, y, z, roll, pitch, yaw, q1..qN
        Expected headers: at least x,y,yaw and (optionally) t, q1..qN.
        Missing z/roll/pitch are filled with provided defaults.
        Missing q's are filled with 0.0.
        """
        def _to_float(s):
            try:
                return float(s)
            except Exception:
                return 0.0

        path = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            hdr = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
            have_x   = any(h.lower() == "x"   for h in hdr)
            have_y   = any(h.lower() == "y"   for h in hdr)
            have_yaw = any(h.lower() == "yaw" for h in hdr)
            if not (have_x and have_y and have_yaw):
                raise ValueError(f"CSV must contain columns x,y,yaw. Found: {hdr}")

            # normalize key lookup to be case-insensitive
            def g(row, key, default="0"):
                # exact first, then case-insensitive
                if key in row:
                    return row[key]
                for k in row.keys():
                    if k.lower() == key.lower():
                        return row[k]
                return default

            for row in reader:
                pose = {}
                pose["x"] = _to_float(g(row, "x"))
                pose["y"] = _to_float(g(row, "y"))
                pose["z"] = _to_float(g(row, "z", default=str(default_z)))

                pose["roll"]  = _to_float(g(row, "roll",  default=str(default_roll)))
                pose["pitch"] = _to_float(g(row, "pitch", default=str(default_pitch)))
                pose["yaw"]   = _to_float(g(row, "yaw"))

                # joints q1..qN
                for j in range(1, num_joints + 1):
                    key = f"q{j}"
                    pose[key] = _to_float(g(row, key, default="0"))

                # angle unit conversion (if CSV in degrees)
                if angles_in_degrees:
                    pose["roll"]  = math.radians(pose["roll"])
                    pose["pitch"] = math.radians(pose["pitch"])
                    pose["yaw"]   = math.radians(pose["yaw"])
                    for j in range(1, num_joints + 1):
                        key = f"q{j}"
                        pose[key] = math.radians(pose[key])

                path.append(pose)

        rospy.loginfo(f"[import_traj_csv] Loaded {len(path)} poses from {csv_path}")
        return path
    
    # def save_open3d_view(self, geometries, out_path="scene.png", width=1600, height=900, bg=None):
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window("Open3D", width=width, height=height, visible=True)
    #     for g in (geometries if isinstance(geometries, (list, tuple)) else [geometries]):
    #         vis.add_geometry(g)

    #     # Optional background color
    #     if bg is not None:
    #         opt = vis.get_render_option()
    #         opt.background_color = np.array(bg, dtype=float)

    #     vis.poll_events()
    #     vis.update_renderer()
    #     vis.capture_screen_image(out_path, do_render=True)
    #     vis.destroy_window()
    def save_open3d_view(self, geometries, out_path="scene.png", width=3200,
                        height=1800,
                        fov_deg=50.0,
                        tilt_deg=20.0,
                        yaw_deg=0.0,
                        pad=1.03,
                        bg_rgba=(1,1,1,1),):
        from math import radians, tan, atan
        def _union_aabb(a, b):
                # Works for legacy & CUDA AABBs
            if hasattr(a, "to_legacy"): a = a.to_legacy()
            if hasattr(b, "to_legacy"): b = b.to_legacy()
            mn = np.minimum(a.get_min_bound(), b.get_min_bound())
            mx = np.maximum(a.get_max_bound(), b.get_max_bound())
            return o3d.geometry.AxisAlignedBoundingBox(mn, mx)

        def _cam_basis_from_angles(tilt_deg=20.0, yaw_deg=45.0):
            """Start looking straight down (0,0,-1), then tilt around world X and yaw around Z."""
            ct, st = np.cos(radians(tilt_deg)), np.sin(radians(tilt_deg))
            cy, sy = np.cos(radians(yaw_deg)),  np.sin(radians(yaw_deg))
            Rx = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
            Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
            R  = Rz @ Rx
            front = (R @ np.array([0,0,-1.0])).astype(float)        # camera looks along +front
            up    = (R @ np.array([0,1.0,0])).astype(float)
            # Orthonormalize
            front = front / np.linalg.norm(front)
            right = np.cross(front, up); right /= np.linalg.norm(right)
            up    = np.cross(right, front); up /= np.linalg.norm(up)
            return front, up, right
        
        
        geoms = [obstacle_mesh] + (list(geometries) if geometries else [])
        geoms = [g for g in geoms if g is not None and not g.is_empty()]

        # --- Build union AABB (for fitting) ---
        aabb = geoms[0].get_axis_aligned_bounding_box()
        for g in geoms[1:]:
            aabb = _union_aabb(aabb, g.get_axis_aligned_bounding_box())

        # Center specifically on the obstacle mesh (your requirement)
        obs_center = obstacle_mesh.get_axis_aligned_bounding_box().get_center()
        center = np.asarray(obs_center)

        extent = np.asarray(aabb.get_extent())   # world-aligned extents
        hx, hy, hz = extent * 0.5

        # Camera orientation from angles
        front, up, right = _cam_basis_from_angles(tilt_deg=tilt_deg, yaw_deg=yaw_deg)

        # Project the AABB half-extents onto camera right/up axes for tight fit
        half_w = abs(right[0])*hx + abs(right[1])*hy + abs(right[2])*hz   # half width on image plane
        half_h = abs(up[0])   *hx + abs(up[1])   *hy + abs(up[2])   *hz   # half height on image plane

        fov_y = radians(fov_deg)
        aspect = float(width) / float(height)
        fov_x = 2.0 * atan(tan(fov_y/2.0) * aspect)

        # Distance so that box fits in both width & height
        d_w = half_w / tan(fov_x / 2.0)
        d_h = half_h / tan(fov_y / 2.0)
        dist = max(d_w, d_h) * float(pad)

        # Eye is behind center along the viewing direction
        eye = center - front * dist

        # --- Render headlessly ---
        r = rendering.OffscreenRenderer(width, height)
        r.scene.set_background(bg_rgba)
        r.scene.scene.set_sun_light(direction=[-1,-1,-2], intensity=45000, color=[1,1,1])
        r.scene.scene.enable_sun_light(True)

        def _mat(rgba=(0.8,0.8,0.8,1.0)):
            m = rendering.MaterialRecord()
            m.shader = "defaultLit"
            m.base_color = rgba
            m.base_roughness = 0.9
            m.base_reflectance = 0.02
            m.base_metallic = 0.0
            return m
        def _material_preserve_appearance(mesh, lit=True):
            """Material that keeps mesh colors/textures as-is."""
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit" if lit else "defaultUnlit"
            # Make sure we don't tint the mesh
            mat.base_color = (1.0, 1.0, 1.0, 1.0)
            mat.base_metallic = 0.0
            mat.base_roughness = 0.9
            mat.base_reflectance = 0.02

            # If the mesh has an albedo texture and UVs, use it.
            try:
                if getattr(mesh, "has_triangle_uvs", lambda: False)() and getattr(mesh, "textures", None):
                    if len(mesh.textures) > 0 and mesh.textures[0] is not None:
                        # Open3D 0.15–0.17
                        if hasattr(mat, "albedo_img"):
                            mat.albedo_img = mesh.textures[0]
                        # (Older/newer versions: if this attribute changes, this try/except just skips)
            except Exception:
                pass

            return mat

        def add_mesh_preserving_colors(scene, name, mesh, lit=True):
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            scene.add_geometry(name, mesh, _material_preserve_appearance(mesh, lit=lit))

        # Add obstacle first (slightly darker), then others
        # r.scene.add_geometry("env", obstacle_mesh, _mat((0.7,0.7,0.7,1)))
        add_mesh_preserving_colors(r.scene, "env", obstacle_mesh, lit=True)
        for i, g in enumerate(geoms[1:], 1):
            # r.scene.add_geometry(f"g{i}", g, _mat((0.2,0.55,0.95,1)))
            add_mesh_preserving_colors(r.scene, f"g{i}", g, lit=True)

        r.setup_camera(fov_deg, center, eye, up)
        img = r.render_to_image()
        o3d.io.write_image(out_path, img)
        # r.release()


    def render_path_video(
        self,
        obstacle_mesh: o3d.geometry.TriangleMesh,
        robot_meshes: list,                 # list[TriangleMesh] — one per frame
        out_mp4: str = "path.mp4",
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        bg_rgba=(1, 1, 1, 1),               # white background
        robot_color=(0.2, 0.55, 0.95, 1.0), # nice blue
        env_color=(0.7, 0.7, 0.7, 1.0),     # grey
        fov_deg: float = 60.0,
    ):
        assert isinstance(obstacle_mesh, o3d.geometry.TriangleMesh)
        robot_meshes = [m for m in robot_meshes if isinstance(m, o3d.geometry.TriangleMesh) and not m.is_empty()]
        if not robot_meshes:
            raise ValueError("robot_meshes is empty or contains no valid TriangleMesh objects.")

        # --- Prepare renderer ---
        r = rendering.OffscreenRenderer(width, height)
        r.scene.set_background(bg_rgba)

        # Materials
        def make_mat(rgba):
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.base_color = rgba
            mat.base_roughness = 0.9
            mat.base_reflectance = 0.02
            mat.base_metallic = 0.0
            return mat
        def _to_legacy_aabb(aabb):
            # If this is a tensor AABB (t.geometry), convert to legacy
            if hasattr(aabb, "to_legacy"):
                return aabb.to_legacy()
            return aabb

        def union_aabb(a, b):
            a = _to_legacy_aabb(a)
            b = _to_legacy_aabb(b)
            minb = np.minimum(np.asarray(a.get_min_bound()), np.asarray(b.get_min_bound()))
            maxb = np.maximum(np.asarray(a.get_max_bound()), np.asarray(b.get_max_bound()))
            return o3d.geometry.AxisAlignedBoundingBox(minb, maxb)

        env_mat   = make_mat(env_color)
        robot_mat = make_mat(robot_color)
        
        def _open_video_sink(path, fps, frame_shape):
            h, w = frame_shape[:2]
            # Try imageio (v2 API) with imageio-ffmpeg
            try:
                import imageio, imageio_ffmpeg  # noqa: F401
                writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                            bitrate="8M", pixelformat="yuv420p")
                def write(frame):
                    writer.append_data(frame)
                def close():
                    writer.close()
                return write, close, "imageio-ffmpeg"
            except Exception as e:
                print(f"[warn] imageio-ffmpeg unavailable: {e}")

            # Fallback: OpenCV VideoWriter
            try:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely available
                vw = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
                if not vw.isOpened():
                    raise RuntimeError("cv2.VideoWriter failed to open.")
                def write(frame):
                    # Expect RGBA or RGB uint8; convert to BGR
                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    bgr = frame[:, :, ::-1]  # RGB->BGR
                    vw.write(bgr)
                def close():
                    vw.release()
                return write, close, "opencv"
            except Exception as e:
                print(f"[warn] OpenCV VideoWriter unavailable: {e}")

            # Last resort: PNG sequence in a folder next to MP4 path
            out_dir = os.path.splitext(path)[0] + "_frames"
            os.makedirs(out_dir, exist_ok=True)
            import imageio.v2 as iio_v2  # pillow-based PNG writing
            idx = {"n": 0}
            def write(frame):
                iio_v2.imwrite(os.path.join(out_dir, f"frame_{idx['n']:05d}.png"), frame)
                idx["n"] += 1
            def close():
                print(f"[info] Wrote PNG sequence to {out_dir}. Encode with ffmpeg:\n"
                    f"ffmpeg -r {fps} -i {out_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {path}")
            return write, close, "png-seq"

        # Add environment once
        r.scene.add_geometry("env", obstacle_mesh, env_mat)

        # --- Camera: freeze viewpoint across frames to avoid flicker ---
        # Use the union of the env AABB + all robot AABBs
        aabb = _to_legacy_aabb(obstacle_mesh.get_axis_aligned_bounding_box())
        for rm in robot_meshes:
            if rm is None or rm.is_empty():
                continue
            aabb = union_aabb(aabb, rm.get_axis_aligned_bounding_box())

        center = aabb.get_center()
        extent = aabb.get_extent()
        diag = float(np.linalg.norm(extent))
        dist = max(diag * 1.8, 1e-3)
        print(dist)
        eye = center + np.array([0, 0, 15])  # oblique view
        up = np.array([0, 0, 15], dtype=float)

        r.setup_camera(fov_deg, center, eye, up)

        # Optional: a sun light for nicer shading
        r.scene.scene.set_sun_light(
            direction=[-1.0, -1.0, -2.0],
            intensity=45000,  # lux
            color=[1.0, 1.0, 1.0]
        )
        r.scene.scene.enable_sun_light(True)
        
        # first frame
        r.scene.add_geometry("robot", robot_meshes[0], robot_mat)
        img = r.render_to_image()
        frame = np.asarray(img)
        write, close, backend = _open_video_sink(out_mp4, fps, frame.shape)
        write(frame)
        r.scene.remove_geometry("robot")

        # remaining frames
        for idx, rm in enumerate(robot_meshes[1:], start=1):
            r.scene.add_geometry("robot", rm, robot_mat)
            frame = np.asarray(r.render_to_image())
            write(frame)
            r.scene.remove_geometry("robot")
        close()
        print(f"[ok] video written via {backend}: {out_mp4}")

        # --- Stream frames directly to video writer (no big memory spikes) ---
        # with iio.imopen(out_mp4, "w", plugin="ffmpeg", fps=fps, codec="h264", bitrate="8M") as writer:
        #     for idx, rm in enumerate(robot_meshes):
        #         name = "robot"
        #         # Add current robot mesh
        #         r.scene.add_geometry(name, rm, robot_mat)
        #         # Render
        #         img = r.render_to_image()
        #         # Convert to numpy (uint8, HxWx3/4). Open3D returns o3d.core.Tensor image; use .numpy()
        #         frame = np.asarray(img)
        #         writer.write(frame)
        #         # Remove the robot mesh for next frame
        #         r.scene.remove_geometry(name)
        #         if (idx + 1) % 50 == 0 or idx == len(robot_meshes) - 1:
        #             print(f"[render] {idx + 1}/{len(robot_meshes)} frames")

        # r.release()
        # print(f"[ok] wrote video: {out_mp4}")
    
    def make_video(self, path, obstacle_mesh, video_path):
        robot_meshes = []
        colormap= plt.get_cmap('jet')

        padding = 0.8
        # then use the code of moving the robot to move to each point in the path
        for pose in path:
            meshes = self.RM.update_robot_arm_bbs(pose, move_base = True, local_frame = False)

            # mesh= self.RM.convert_bbs_to_mesh(conf_bb)
            combined_mesh = o3d.geometry.TriangleMesh()
            for mesh in meshes.values():
                combined_mesh += mesh

            bb = combined_mesh.get_axis_aligned_bounding_box()
            bb.scale(2.0,bb.get_center())
            # simplify obstacle mesh
            obstacle_mesh_cropped = obstacle_mesh.crop(bb)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)

            if is_colliding_o3d(combined_mesh, obstacle_mesh_cropped):
                mesh_color = [1.0,0,0]
            else:
                distance, _ = self.EBAND.compute_2mesh_distance(combined_mesh, obstacle_mesh_cropped)
                # print(distance)
                norm_distance = (1-min(distance, obstacle_threshold)/(obstacle_threshold))*padding
                # print(norm_distance)
                # print('----------------')
                mesh_color = np.asarray(colormap(norm_distance))[:3]

            combined_mesh.paint_uniform_color(mesh_color)
            combined_mesh.compute_vertex_normals()
            robot_meshes.append(combined_mesh)
        self.render_path_video(obstacle_mesh= obstacle_mesh, robot_meshes=robot_meshes, out_mp4 = video_path, fps=2)


def generate_colorbar(obstacle_threshold=obstacle_threshold, padding=0.5, cmap='jet'):
    colormap = plt.get_cmap(cmap)
    
    # Create normalized gradient from 0 to obstacle_threshold
    gradient = np.linspace(0, obstacle_threshold, 256).reshape(-1, 1)  # Make it vertical
    norm_distance = 1 - np.clip(gradient / obstacle_threshold, 0, 1)  # Normalize between 0 and 1
    
    # Display vertical gradient colorbar
    fig, ax = plt.subplots(figsize=(2, 6))
    ax.imshow(norm_distance, aspect='auto', cmap=colormap, origin='lower')
    
    # Set labels
    ax.set_ylabel('Distance')
    ax.set_yticks([0, 128, 256])
    ax.set_yticklabels([f"{0}", f"{obstacle_threshold/2:.2f}", f"{obstacle_threshold:.2f}"])
    ax.set_xticks([])
    ax.set_title("Colormap Distance Representation")

    plt.show()




if __name__ == '__main__':
    # mn = MeshNav()
    # # mn.run()
    # obstacle_mesh = mn.import_obstacles()
    # # o3d.visualization.draw_geometries([obstacle_mesh])
    
    # # path = mn.import_path("/home/rui/pcl_ws/src/mesh_nav/data/irosusingresult1.json")
    # # path = mn.import_path("/home/rui/pcl_ws/src/mesh_nav/data/test_path_goal2.json")
    # csv_path = "/home/rui/ds/testsiros2025/testsremani_newdatasetvis/test_166.csv"  # <-- change to your CSV
    # # csv_path = "/home/rui/ds/testsiros2025/testsmeshnav_newdataset/test_166.csv"  # <-- change to your CSV
    # path = mn.import_traj_csv(
    #     csv_path,
    #     num_joints=6,               # UR5
    #     angles_in_degrees=False,    # set True if your CSV has degrees
    #     default_z=0.0,
    #     default_roll=0.0,
    #     default_pitch=0.0
    # )
    
    # mn.plot_path_3d(path, obstacle_mesh)
    # generate_colorbar()
    import argparse, os, sys
    loop = None

    parser = argparse.ArgumentParser(description="Run MeshNav with a chosen test CSV.")
    parser.add_argument(
        "-t", "--test", type=int, default=1,
        help="Test number to use (builds path .../test_<N>.csv). Default: 1"
    )
    parser.add_argument(
        "--loop", type=int, default=None,
        help="Optional: loop and print the arg runs."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Optional: full CSV path to use (overrides --test)."
    )
    parser.add_argument(
        "--video", type=bool, default=False,
        help="Optional: full CSV path to use (overrides --test)."
    )
    args = parser.parse_args()

    # Build CSV path from test number unless a full path is provided
    if args.csv:
        csv_path = args.csv
    elif args.loop:
        base_dir = "/home/rui/ds/testsiros2025/testsmeshnav_newdataset"
        loop = args.loop
    else:
        base_dir = "/home/rui/ds/testsiros2025/testsmeshnav_newdataset"
        csv_path = os.path.join(base_dir, f"test_{args.test}.csv")

    # if not os.path.isfile(csv_path):
    #     print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
    #     sys.exit(1)

    mn = MeshNav()

    obstacle_mesh = mn.import_obstacles()

    if loop is None:
        path = mn.import_traj_csv(
            csv_path,
            num_joints=6,               # UR5
            angles_in_degrees=False,    # set True if your CSV has degrees
            default_z=0.0,
            default_roll=0.0,
            default_pitch=0.0
        )
        
        if args.video:
            mn.make_video(path, obstacle_mesh, os.path.join(base_dir, f"videos/test.mp4"))
        else:
            mn.plot_path_3d(path, obstacle_mesh)
    else:
        for i in range(0,loop):
            csv_path = os.path.join(base_dir, f"test_{i}.csv")
            path = mn.import_traj_csv(
                csv_path,
                num_joints=6,               # UR5
                angles_in_degrees=False,    # set True if your CSV has degrees
                default_z=0.0,
                default_roll=0.0,
                default_pitch=0.0
            )

            if args.video:
                mn.make_video(path, obstacle_mesh, os.path.join(base_dir, f"videos/{i}.mp4"))
            else:
                mn.plot_path_3d(path, obstacle_mesh, save_path=os.path.join(base_dir, f"images_3/{i}_path.png"))
                ogpath = [path[0], path[-1]]
                mn.plot_path_3d(ogpath, obstacle_mesh, save_path=os.path.join(base_dir, f"images_3/{i}_init.png"))