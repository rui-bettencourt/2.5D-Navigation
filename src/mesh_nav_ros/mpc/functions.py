import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import json
import matplotlib.gridspec as gridspec

class Auxiliary(object):
    def show_world(self, init, gpath, qs=False, title=None):
        """
        Takes a 2D list as input and returns a visually appealing grid plot.
        """
        self.qs = qs
        if qs:
            # self.fig, (self.ax, self.ax2) = plt.subplots(2, 1, figsize=(15, 15))  # Two subplots
            self.fig = plt.figure(figsize=(15, 8))  # Overall figure size
            gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1],height_ratios=[1,1,1])  # 3:1 ratio (left:right)
            self.ax = self.fig.add_subplot(gs[:, :2])

            # Second subplot: Smaller q1 over time plot on the right
            self.ax1 = self.fig.add_subplot(gs[0,2])
            # Second subplot: q1 over time
            self.ax1.set_title("q1 Over Time")
            self.ax1.set_xlabel("Time step")
            self.ax1.set_ylabel("q Value (rad)")
            self.ax1.grid(True)
            
            self.ax2 = self.fig.add_subplot(gs[1,2])
            # Second subplot: q1 over time
            self.ax2.set_title("q2 Over Time")
            self.ax2.set_xlabel("Time step")
            self.ax2.set_ylabel("q Value (rad)")
            self.ax2.grid(True)
            
            self.ax3 = self.fig.add_subplot(gs[2,2])
            # Second subplot: q1 over time
            self.ax3.set_title("q3 Over Time")
            self.ax3.set_xlabel("Time step")
            self.ax3.set_ylabel("q Value (rad)")
            self.ax3.grid(True)
            
            # self.ax4 = self.fig.add_subplot(gs[3,2])
            # # Second subplot: q1 over time
            # self.ax4.set_title("q4 Over Time")
            # self.ax4.set_xlabel("Time step")
            # self.ax4.set_ylabel("q Value (rad)")
            # self.ax4.grid(True)
            
            # self.ax5 = self.fig.add_subplot(gs[0,2])
            # # Second subplot: q1 over time
            # self.ax5.set_title("q1 Over Time")
            # self.ax5.set_xlabel("Time step")
            # self.ax5.set_ylabel("q Value (rad)")
            # self.ax5.grid(True)
            
            # self.ax1 = self.fig.add_subplot(gs[0,2])
            # # Second subplot: q1 over time
            # self.ax1.set_title("q1 Over Time")
            # self.ax1.set_xlabel("Time step")
            # self.ax1.set_ylabel("q Value (rad)")
            # self.ax1.grid(True)
            
            # self.ax1 = self.fig.add_subplot(gs[0,2])
            # # Second subplot: q1 over time
            # self.ax1.set_title("q1 Over Time")
            # self.ax1.set_xlabel("Time step")
            # self.ax1.set_ylabel("q Value (rad)")
            # self.ax1.grid(True)

            self.qs_goals = [[],[],[],[],[],[],[]]
        else:
            self.fig, self.ax = plt.subplots(figsize=(30,30))

        self.ax.grid(visible=False,color='k', linewidth=3)
        plt.title(title, fontsize=20)


        staticxs = [init[0]] + [point['x'] for point in gpath]
        staticys = [init[1]] + [point['y'] for point in gpath]

        linegoal, = self.ax.plot(staticxs, staticys, linestyle='-',marker='*', markersize=10)

        self.__arrows = []


        self.ax.set_xticks(np.arange(-10,-4, step=1))
        self.ax.set_yticks(np.arange(-10, -4, step=1))

        plt.ion()  # Turn on interactive mode
        plt.show()
        # plt.show(block=False)

    def update_world(self, mpc, converged=True, title=None, path=[], resolution=1.0, goal=None):
        # remove all the previous arrows
        for arrow in self.__arrows:
            arrow.remove()
        self.__arrows = []
        if self.qs:
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            q1_values = []
            q2_values = []
            q3_values = []
            if goal is not None:
                self.qs_goals[0].append(goal['q1'])
                self.qs_goals[1].append(goal['q2'])
                self.qs_goals[2].append(goal['q3'])

        # self.ax.clear()
        self.ax.set_title(title, fontsize=20)
        if len(path)!=0:
            for step in path:
                arrow = self.ax.arrow(step['x'],
                        step['y'],
                        np.cos(step['yaw'])*resolution*0.2,
                        np.sin(step['yaw'])*resolution*0.2,
                        color='blue',
                        head_width=resolution*0.1)
                self.__arrows.append(arrow)

                if self.qs:
                    q1_values.append(step['q1'])
                    q2_values.append(step['q2'])
                    q3_values.append(step['q3'])
        if converged:
            # for i in range(len(mpc.sol.value(mpc.pos_x))):
            #     xs = mpc.sol.value(mpc.pos_x)
            #     ys = mpc.sol.value(mpc.pos_y)
            #     yaws = mpc.sol.value(mpc.pos_yaw)
            for i in range(len(mpc.sol.value(mpc.X[0,:]))):
                xs = mpc.sol.value(mpc.X[0,:])
                ys = mpc.sol.value(mpc.X[1,:])
                yaws = mpc.sol.value(mpc.X[5,:])


                arrow = self.ax.arrow(xs[i],
                            ys[i],
                            np.cos(yaws[i])*resolution*0.4,
                            np.sin(yaws[i])*resolution*0.4,
                            color='red',
                            head_width=resolution*0.2)

                self.__arrows.append(arrow)

        if self.qs:
            # Update q1 plot
            time_values = np.linspace(0,len(q1_values),len(q1_values))
            self.ax1.plot(time_values, np.asarray(q1_values), marker='.', linestyle='-', color='b')
            self.ax1.plot(time_values, np.asarray(self.qs_goals[0]), marker='*', linestyle='--', color='k')
            self.ax2.plot(time_values, np.asarray(q2_values), marker='.', linestyle='-', color='r')
            self.ax2.plot(time_values, np.asarray(self.qs_goals[1]), marker='*', linestyle='--', color='k')
            self.ax3.plot(time_values, np.asarray(q3_values), marker='.', linestyle='-', color='g')
            self.ax3.plot(time_values, np.asarray(self.qs_goals[2]), marker='*', linestyle='--', color='k')


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def simulate_step(self,dt, state, control, configs, terrain_func=None, 
                                noise={'nx': 0.0, 'ny': 0.0, 'nz': 0.0, 
                                        'nroll': 0.0, 'npitch': 0.0, 'nyaw': 0.0, 
                                        'nvx': 0.0, 'nvyaw': 0.0,
                                        'nq1': 0.0, 'nq2': 0.0, 'nq3': 0.0, 'nq4': 0.0, 'nq5': 0.0, 'nq6': 0.0, 'nq7': 0.0}):
                                    # 'nx': 0.01, 'ny': 0.01, 'nz': 0.005, 
                                    #     'nroll': 0.005, 'npitch': 0.005, 'nyaw': 0.01, 
                                    #     'nvx': 0.05, 'nvyaw': 0.05}):
        """
        Simulates the next state of a ground robot in 6-DOF (x, y, z, roll, pitch, yaw)
        considering terrain elevation.
        
        Parameters:
            dt (float): Time step
            state (dict): Current robot state
            control (dict): Control inputs (Fx: acceleration, Fyaw: angular acceleration)
            configs (dict): Min/max velocity constraints
            terrain_func (callable): Function terrain_func(x, y) -> (z, roll, pitch)
            noise (dict): Noise parameters
        
        Returns:
            sim_state (dict): Updated robot state
        """
        
        sim_state = {}
        
        g = 9.81  # Gravity
        motor_strength = 1.0

        # Translational velocity update with motor & gravity compensation
        sim_state['vx'] = np.clip(
            state['vx'] + control['Fx'] * dt - g * np.sin(state['pitch']) * dt * (1.0 - motor_strength) +
            np.random.normal(scale=noise['nvx']) * dt,
            configs['min_vx'], configs['max_vx']
        )
        
        sim_state['vyaw'] = np.clip(
            state['vyaw'] + control['Fyaw'] * dt + np.random.normal(scale=noise['nvyaw']) * dt,
            configs['min_vyaw'], configs['max_vyaw']
        )

        # Update position based on velocity and orientation
        sim_state['x'] = state['x'] + sim_state['vx'] * math.cos(state['yaw']) * dt + np.random.normal(scale=noise['nx'])
        sim_state['y'] = state['y'] + sim_state['vx'] * math.sin(state['yaw']) * dt + np.random.normal(scale=noise['ny'])
        sim_state['yaw'] = state['yaw'] + sim_state['vyaw'] * dt + np.random.normal(scale=noise['nyaw'])


        # If no terrain function is provided, simulate small random variations
        sim_state['z'] = state['z'] + np.random.normal(scale=noise['nz'])
        sim_state['roll'] = state['roll'] + np.random.normal(scale=noise['nroll'])
        sim_state['pitch'] = state['pitch'] + np.random.normal(scale=noise['npitch'])

        sim_state['q1'] = state['q1'] + control['dq1'] * dt + np.random.normal(scale=noise['nq1'])
        sim_state['q2'] = state['q2'] + control['dq2'] * dt + np.random.normal(scale=noise['nq2'])
        sim_state['q3'] = state['q3'] + control['dq3'] * dt + np.random.normal(scale=noise['nq3'])
        sim_state['q4'] = state['q4'] + control['dq4'] * dt + np.random.normal(scale=noise['nq4'])
        sim_state['q5'] = state['q5'] + control['dq5'] * dt + np.random.normal(scale=noise['nq5'])
        sim_state['q6'] = state['q6'] + control['dq6'] * dt + np.random.normal(scale=noise['nq6'])
        sim_state['q7'] = state['q7'] + control['dq7'] * dt + np.random.normal(scale=noise['nq7'])

        return sim_state



    def plot_robot_and_reference(self, robot_path_file, ref_path_file):
        # Load the robot and reference path JSON files.
        with open(robot_path_file, 'r') as f:
            robot_data = json.load(f)
        with open(ref_path_file, 'r') as f:
            ref_data = json.load(f)
        
        # Extract x, y, yaw for the robot path.
        robot_x = [pt['x'] for pt in robot_data]
        robot_y = [pt['y'] for pt in robot_data]
        robot_yaw = [pt['yaw'] for pt in robot_data]
        
        # Extract x, y, yaw for the reference path.
        ref_x = [pt['x'] for pt in ref_data]
        ref_y = [pt['y'] for pt in ref_data]
        ref_yaw = [pt['yaw'] for pt in ref_data]
        
        # --- Plot the trajectory with robot poses ---
        plt.figure(figsize=(10, 8))
        
        # Plot the trajectories.
        plt.plot(robot_x, robot_y, label='Robot trajectory', color='blue')
        plt.plot(ref_x, ref_y, label='Reference trajectory', linestyle='--',marker='*', markersize=10, color='red')
        
        # Plot arrows for each robot pose (using x, y as origin and yaw for orientation).
        arrow_length = 0.2  # Adjust arrow length as needed.
        for x, y, yaw in zip(robot_x, robot_y, robot_yaw):
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)
            plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot Path with Poses')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        
        # --- Plot the evolution of q1 to q7 ---
        # Create time arrays (assuming each dictionary is one time step).
        time_robot = np.arange(len(robot_data))
        time_ref = np.arange(len(ref_data))
        
        # Prepare dictionaries to hold joint angle values.
        qs_robot = {}
        qs_ref = {}
        for i in range(1, 8):
            q_label = f"q{i}"
            qs_robot[q_label] = [pt[q_label] for pt in robot_data]
            qs_ref[q_label] = [pt[q_label] for pt in ref_data]
        
        plt.figure(figsize=(12, 8))
        
        palette = self.generate_color_palette()
        # Plot each joint angle evolution.
        for i in range(1, 8):
            q_label = f"q{i}"
            plt.plot(time_robot, qs_robot[q_label], label=f'{q_label} (robot)',color=palette[(i-1)*2,:])
            plt.plot(time_ref, qs_ref[q_label], linestyle='--', label=f'{q_label} (ref)',color=palette[(i-1)*2+1,:])
        
        plt.xlabel('Time step')
        plt.ylabel('Angle (rad)')
        plt.title('Evolution of Joint Angles (q1 to q7)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_color_palette(self):
        # Define 7 distinct base colors in RGB format (values between 0 and 1)
        base_colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [0.5, 0.5, 0.5]   # Gray
        ])

        # Initialize an empty list to store the color pairs
        color_pairs = []

        # Define factors to adjust brightness
        light_factor = 1.5
        dark_factor = 0.5

        # Generate lighter and darker variants for each base color
        for color in base_colors:
            # Ensure values stay within the valid range [0, 1]
            lighter_color = np.clip(color * light_factor, 0, 1)
            darker_color = np.clip(color * dark_factor, 0, 1)
            color_pairs.append(lighter_color)
            color_pairs.append(darker_color)

        # Convert the list to a NumPy array
        palette = np.array(color_pairs)

        return palette

if __name__ == '__main__':
    a = Auxiliary()
    # Example usage:
    # Provide the paths to your JSON files.
    robot_json_file = "/home/rui/pcl_ws/src/mesh_nav/data/tests/mpc_path.json"
    ref_json_file = "/home/rui/pcl_ws/src/mesh_nav/data/tests/mpc_path_targets.json"

    a.plot_robot_and_reference(robot_json_file, ref_json_file)
