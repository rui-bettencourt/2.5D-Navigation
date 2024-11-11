import numpy as np
import pickle
import networkx as nx
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class GraphManager(object):
    def __init__(self, filename=None):
        if filename is not None:
            self.import_graph(filename)
        else:
            self.G = None
            self.__Gpositions = None
            self.__Gpoints = None
            self.__Gpoints_kdtree = None

    def import_graph(self, filename):
        with open(filename, "rb") as f:
            self.G = pickle.load(f)

        # Extract the positions from the graph nodes
        self.__Gpositions = nx.get_node_attributes(self.G, 'pos')
        self.__Gpoints = np.array(list(self.__Gpositions.values()))
        self.__Gpoints_kdtree = KDTree(self.__Gpoints)

    def world_to_graph(self, point):
        distance, index = self.__Gpoints_kdtree.query(point)
        node_list = list(self.__Gpositions.keys())
        return node_list[index]

    def graph_to_world(self, node):
        return self.__Gpositions[node]

    def plan(self, start_point, end_point):
        # Find the nearest nodes in the graph to these coordinates
        start_node = self.world_to_graph(start_point[:3])
        end_node = self.world_to_graph(end_point)
        print(start_node, end_node)

        # Find the shortest path between the start and end nodes
        try:
            path = nx.shortest_path(self.G, source=start_node, target=end_node, weight='weight', method='dijkstra')
            print("Shortest path:", path)
            
            # Plot the graph with the path highlighted (2D projection for visualization)
            pos = {i: (self.__Gpoints[i][0], self.__Gpoints[i][1]) for i in self.G.nodes()}

            plt.figure()
            nx.draw(self.G, pos, with_labels=True, node_size=50, node_color="red", edge_color="blue")

            # Highlight the path
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_nodes(self.G, pos, nodelist=path, node_color="green", node_size=100)
            nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, edge_color="green", width=2)
            plt.show()

            # Create a list of poses
            poses = []
            for i in range(len(path)):
                position = np.asarray(self.graph_to_world(path[i]))
                if i == 0:
                    orientation = start_point[3]  # Initial orientation can be set to 0.0 or any default value
                else:
                    prev_position = np.asarray(self.graph_to_world(path[i - 1]))
                    delta = position - prev_position
                    orientation = np.arctan2(delta[1], delta[0])
                pose = np.append(position, orientation)
                poses.append(pose)

            return poses

        except nx.NetworkXNoPath:
            print("No path found between the given coordinates.")
            return None

    def random_node(self):
        # randomly chooses a node and returns its x and y
        index = np.random.choice(self.__Gpoints.shape[0])
        x, y, _ = self.__Gpoints[index]  # Assuming z is not needed
        return x, y