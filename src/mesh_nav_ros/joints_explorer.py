import numpy as np
import time

# class Node:
#     def __init__(self, state, parent=None):
#         self.state = state
#         self.parent = parent

# class JointExploration:
#     def __init__(self, bounds):
#         # Define bounds for each joint (assuming each joint has a range of [-pi, pi])
#         self.bounds = bounds

#     def distance(self, state1, state2):
#         return np.linalg.norm(np.array(state1) - np.array(state2))

#     def random_state(self, bounds):
#         return [np.random.uniform(low, high) for low, high in bounds]

#     def nearest(self, tree, state):
#         return min(tree, key=lambda node: self.distance(node.state, state))

#     def steer(self, from_state, to_state, step_size):
#         direction = np.array(to_state) - np.array(from_state)
#         length = np.linalg.norm(direction)
#         direction = direction / length  # Normalize the direction
#         return np.array(from_state) + step_size * direction

#     def rrt(self, start, goal, bounds, max_iter=1000, step_size=0.1):
#         tree = [Node(start)]
#         for _ in range(max_iter):
#             rand_state = self.random_state(bounds)
#             nearest_node = self.nearest(tree, rand_state)
#             new_state = self.steer(nearest_node.state, rand_state, step_size)


#             new_node = Node(new_state, nearest_node)
#             tree.append(new_node)
            
#             if self.distance(new_state, goal) < step_size:
#                 goal_node = Node(goal, new_node)
#                 tree.append(goal_node)
#                 return tree, goal_node
        
#         return None, None

#     def extract_path(self, goal_node):
#         path = []
#         node = goal_node
#         while node:
#             path.append(node.state)
#             node = node.parent
#         return path[::-1]

#     def smooth_path(self, path, alpha=0.5, beta=0.5, tolerance=1e-2, max_iterations=2000):
#         def compute_force(current, prev, next):
#             force = np.zeros_like(current)
#             if prev is not None:
#                 force += alpha * (prev - current)
#             if next is not None:
#                 force += beta * (next - current)
#             return force

#         smoothed_path = np.array(path, copy=True)
#         for _ in range(max_iterations):
#             max_displacement = 0
#             for i in range(1, len(smoothed_path) - 1):
#                 current = smoothed_path[i]
#                 prev = smoothed_path[i - 1]
#                 next = smoothed_path[i + 1]

#                 force = compute_force(current, prev, next)
#                 smoothed_path[i] += force
#                 max_displacement = max(max_displacement, np.linalg.norm(force))

#             if max_displacement < tolerance:
#                 break

#         return smoothed_path.tolist()

#     def plan(self, start, goal, smooth=False):
#         tree, goal_node = self.rrt(start, goal, self.bounds)

#         if goal_node:
#             print("Solution found!")
#             path = self.extract_path(goal_node)
#             print("Original Path:")
#             for state in path:
#                 print(state)

#             if smooth:
#                 # Smooth the path
#                 path = self.smooth_path(path)
#                 print("Smoothed Path:")
#                 for state in path:
#                     print(state)
#             return path
#         else:
#             print("No solution found.")
#             return []

########################################################################

import networkx as nx
import matplotlib.pyplot as plt
import rospy
from graph_functions import GraphManager


class PRM(object):
    def __init__(self, Graph=None, random_position=False):
        # Configuration space parameters
        self.x_limits = (-20, 20)
        self.y_limits = (-20, 20)
        self.theta_limits = (-np.pi, np.pi)
        self.joint1_limits = (0.0, 0.35)
        self.joint2_limits = (0.07, 2.68) #TODO: check limits automatically or make it easier to change
        self.Graph = Graph
        self.prm = None
        if Graph is None:
            self.random_position = True
        else:
            self.random_position = random_position

    # Helper functions
    def random_configuration(self, random_position=False):
        # # random x and y
        if random_position:
            x = np.random.uniform(*self.x_limits)
            y = np.random.uniform(*self.y_limits)
        else:
            # instead of random
            x,y = self.Graph.random_node()
        theta = np.random.uniform(*self.theta_limits)
        joint1 = np.random.uniform(*self.joint1_limits)
        joint2 = np.random.uniform(*self.joint2_limits)
        return np.array([x, y, theta, joint1, joint2])

    def is_valid(self, configuration):
        # Add your validity checks here (e.g., collision checking) 
        return True

    def distance(self, config1, config2):
        # Euclidean distance in 5D space
        return np.linalg.norm(config1 - config2)

    def generate(self, num_samples = 1000, threshold = 0.1):
        # Build the PRM
        G = nx.Graph()

        # Sample random configurations
        configurations = []
        for _ in range(num_samples):
            while True:
                config = self.random_configuration()
                if self.is_valid(config):
                    configurations.append(config)
                    break

        # Add nodes to the graph
        for i, config in enumerate(configurations):
            G.add_node(i, configuration=config)

        # Add edges based on the threshold distance
        for i in range(len(configurations)):
            for j in range(i + 1, len(configurations)):
                if self.distance(configurations[i], configurations[j]) <= threshold:
                    G.add_edge(i, j, weight=self.distance(configurations[i], configurations[j]))
        self.prm = G

        # Save the PRM
        # import pickle
        # with open("prm_graph.pkl", "wb") as f:
        #     pickle.dump(G, f)

        # (Optional) Visualize the PRM for the base positions
        # plt.figure()
        # for (i, j) in G.edges:
        #     config1 = G.nodes[i]['configuration']
        #     config2 = G.nodes[j]['configuration']
        #     plt.plot([config1[0], config2[0]], [config1[1], config2[1]], 'k-', alpha=0.3)
        # plt.scatter([config[0] for config in configurations], [config[1] for config in configurations], c='r')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('PRM for Differential Drive Robot with 2-Joint Arm')
        # plt.show()

if __name__ == '__main__':
    rospy.init_node('debug_mesh', anonymous=True)
    path = '/home/rui/ds/testbed/'
    Graph = GraphManager(path+"traversablegroundgraph.pkl")

    mc = PRM(Graph)
    time = rospy.Time.now()
    mc.generate(num_samples=1000)
    print("Took {} seconds".format(rospy.Time.now() - time))