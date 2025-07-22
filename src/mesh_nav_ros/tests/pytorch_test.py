import math
import rospy
import time
import numpy as np
import torch
import pytorch_kinematics as pk
from sensor_msgs.msg import JointState

# Check if CUDA is available
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print("Using GPU for computation.")
    print("Device type:", torch.cuda.get_device_name(device))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    # Set the device to CPU
    device = torch.device("cpu")
    print("GPU not available. Using CPU for computation.")


rospy.init_node('elastic_band_planner_node')
state = []
dtype = torch.float64

def joint_states_callback(msg):
    global state
        # Extract the first 7 joint positions
    joint_positions = np.asarray(msg.position[:7])
    # Update the robot joints
    state = np.append(0.0,joint_positions)

rospy.Subscriber('/joint_states', JointState, joint_states_callback)
rospy.sleep(2.0)

# can convert Chain to SerialChain by choosing end effector frame
chain = pk.build_serial_chain_from_urdf(open('/home/rui/socrob_ws/src/isr_tiago/simulation/mbot_simulation_environments/robots/tiago_ouster.urdf').read(), 'arm_7_link')
chain = chain.to(dtype=dtype, device=device)
print(chain.high.cpu().numpy())
print(chain.high.cpu().numpy()[0])
chain.print_tree()
# extract a specific serial chain such as for inverse kinematics
serial_chain = pk.SerialChain(chain, "arm_1_link", "base_footprint")
serial_chain = serial_chain.to(dtype=dtype, device=device)
serial_chain.print_tree()
th = torch.tensor(np.append(0.0,state[:1]), dtype=torch.float64).to(device)  # Move to GPU or CPU
# (1,6,7) tensor, with 7 corresponding to the DOF of the robot
start_time = time.time()
J = serial_chain.jacobian(th)
end_time = time.time()
print("Jacobian calculation time:", end_time - start_time)
print("--------------- JACOBIAN-------------------------")
print(J)
print('-------------------------------------------------')
print("Jacobian matrix shape:", J.size())
print('-------------------------------------------------')
Jnp = J.cpu().numpy()
print("Original Shape:", Jnp.shape)
Jnp = np.squeeze(Jnp, axis=0)
print("Squeezed Shape:", Jnp.shape)
print(Jnp)
# specify joint values (can do so in many forms)
#th =[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]
# do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
start_time = time.time()
ret = chain.forward_kinematics(th, end_only=False)
end_time = time.time()
print("Forward kinematics calculation time:", end_time - start_time)
print(ret)
print(np.squeeze(ret['arm_7_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0))


    # # get Jacobian in parallel and use CUDA if available
    # N = 1000
    # d = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = torch.float64

    # chain = serial_chain.to(dtype=dtype, device=d)
    # # Jacobian calculation is differentiable
    # th = torch.rand(N, 8, dtype=dtype, device=d, requires_grad=True)
    # # (N,6,7)
    # J = chain.jacobian(th)
    # print(J)
    # Spin the ROS node
rospy.spin()
