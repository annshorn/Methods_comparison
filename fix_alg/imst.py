import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont

import numpy as np

from scipy.spatial.distance import pdist, squareform

import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from heapq import heappop, heappush
from itertools import combinations
import heapq
import math
import statistics
import community as community_louvain
import matplotlib.pyplot as plt



class Graph:
    def __init__(self, df):
        self.w, self.h = df['win_w'][0], df['win_h'][0]
        self.df = df
        self.vertices = []
        self.edges = []
        self.adjacency_list = {}

        for i, (x, y, timestamp) in enumerate(self.df[['x', 'y', 'timestamp']].values):
            self.add_vertex(x, y, timestamp)
            
        self.create_edges()

    def add_vertex(self, x, y, timestamp):
        vertex = (x, y, timestamp)
        self.vertices.append(vertex)
        self.adjacency_list[vertex] = []

    def calculate_weight(self, v1, v2):
        dist = math.sqrt((v1[0] / self.w - v2[0] / self.w)**2 + (v1[1] / self.h - v2[1] / self.h)**2)
        #time_diff = abs(v1[2] - v2[2])
        weight = dist#np.exp(-dist)#*time_diff
        return weight

    def add_edge(self, v1, v2):
        weight = self.calculate_weight(v1, v2)
        self.edges.append((v1, v2, weight))
        self.adjacency_list[v1].append((v2, weight))
        self.adjacency_list[v2].append((v1, weight))

    def create_edges(self):
        for v1, v2 in combinations(self.vertices, 2):
            self.add_edge(v1, v2)

    def save_vertices(self, filename):
        with open(filename, 'w') as file:
            for x, y, timestamp in self.vertices:
                file.write(f"{x} {y} {timestamp}\n")

    def save_edges(self, filename):
        with open(filename, 'w') as file:
            for (v1, v2, weight) in self.edges:
                file.write(f"{v1[0]} {v1[1]} {v1[2]} {v2[0]} {v2[1]} {v2[2]} {weight}\n")


def prim_algorithm(graph):
    # use first vertex as starting point
    start_vertex = graph.vertices[0]
    # Priority queue for storing edges (weight, start vertex, end vertex)
    priority_queue = []
    for v, weight in graph.adjacency_list[start_vertex]:
        heapq.heappush(priority_queue, (weight, start_vertex, v))
    
    in_mst = set()
    in_mst.add(start_vertex)
    mst_edges = []
    total_weight = 0

    while priority_queue and len(in_mst) < len(graph.vertices):
        weight, u, v = heapq.heappop(priority_queue)
        if v not in in_mst:
            in_mst.add(v)
            mst_edges.append((u, v, weight))
            total_weight += weight
            # Add all edges emanating from the new vertex
            for next_v, next_weight in graph.adjacency_list[v]:
                if next_v not in in_mst:
                    heapq.heappush(priority_queue, (next_weight, v, next_v))

    return total_weight, mst_edges

def prim_algorithm_networkx(graph):
    mst = nx.minimum_spanning_tree(graph, algorithm='prim')
    mst_edges = list(mst.edges(data=True))
    total_weight = sum(weight['weight'] for u, v, weight in mst_edges)
    return total_weight, mst_edges


def build_tree(mst_edges):
    tree = {}
    for u, v, w in mst_edges:
        if u not in tree:
            tree[u] = []
        if v not in tree:
            tree[v] = []
        tree[u].append(v)
        tree[v].append(u)
    return tree

def dfs(tree, start_vertex, visited=None, depth=0):
    if visited is None:
        visited = {}
    visited[start_vertex] = depth
    max_depth = depth
    for neighbor in tree.get(start_vertex, []):
        if neighbor not in visited:
            current_depth = dfs(tree, neighbor, visited, depth + 1)
            if current_depth > max_depth:
                max_depth = current_depth
    visited[start_vertex] = max(max_depth, visited[start_vertex])
    return max_depth

def find_max_depths(mst_edges):
    tree = build_tree(mst_edges)
    max_depths = {}
    for vertex in tree:
        visited = {}
        max_depths[vertex] = dfs(tree, vertex, visited)
    return max_depths

def calculate_mst_depths_stats(df):
    g = Graph(df)
    _, mst_edges = prim_algorithm(g)
    max_depths = find_max_depths(mst_edges)
    depth_values = list(max_depths.values())
    mean_depth = statistics.mean(depth_values)
    std_depth = statistics.stdev(depth_values)
    return mst_edges, max_depths, mean_depth, std_depth

def make_cluster(data, mean_points=True):
    subgraph = nx.Graph()
    subgraph.add_weighted_edges_from(data)  # assuming weights are still relevant

    # Step 2: Apply the Louvain community detection algorithm
    partition = community_louvain.best_partition(subgraph)

    # Step 3: Print the clusters
    cluster_map = {}
    for node, cluster_id in partition.items():
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(node)
    
    if mean_points is True:
        means_ = []
        for cluster_id, nodes in cluster_map.items():
            nodearray = [[node[0], node[1]] for node in nodes]
            means_.append(np.median(nodearray, axis=0))
        return means_
    else:
        return cluster_map

root_paths = [
            "/home/csn801/EyeTrackProject/data/400/06.06.2021/",
            #   "/home/csn801/EyeTrackProject/data/400/05.06.2021/",
            #   "/home/csn801/EyeTrackProject/data/400/10.04.2021/",
            #   "/home/csn801/EyeTrackProject/data/400/14.11.2021/"
            # "/home/csn801/EyeTrackProject/data/600/02.04.2022/",
            #   "/home/csn801/EyeTrackProject/data/600/12.03.2022/",
            #   "/home/csn801/EyeTrackProject/data/600/19.03.2022/",
            #   "/home/csn801/EyeTrackProject/data/600/27.03.2022/"
              ]

for root_path in root_paths:
    folders = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    k = 0
    for folder_name in folders:
        raw_path = os.path.join(root_path, folder_name, 'miccai2025', 'raw_gaze.csv')
        output_path = os.path.join(root_path, folder_name, 'miccai2025', 'imst_filtered_edges.csv')
        if os.path.exists(raw_path) and not os.path.exists(output_path):
            raw_gaze = pd.read_csv(raw_path)
            if len(raw_gaze) < 2:
                continue
            if len(raw_gaze) > 7000:
                raw_gaze = raw_gaze.iloc[::2]
            mst_edges, max_depths, mean_depth, std_depth = calculate_mst_depths_stats(raw_gaze)
            depth_threshold = mean_depth
            filtered_edges = [edge for edge in mst_edges if max_depths[edge[0]] <= depth_threshold and max_depths[edge[1]] <= depth_threshold]
            means_ = make_cluster(filtered_edges)

            df_fixations = pd.DataFrame(means_, columns=['x', 'y'])

            if len(raw_gaze) > len(df_fixations):
                    raw_gaze = raw_gaze.iloc[:len(df_fixations)]
            
            df_fixations['win_w'] = raw_gaze['win_w']
            df_fixations['win_h'] = raw_gaze['win_h']
            df_fixations['radiologist'] = raw_gaze['radiologist']
            df_fixations['rad_name'] = df_fixations['radiologist'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
            df_fixations['label'] = raw_gaze['label']
            df_fixations['case'] = raw_gaze['case']
            df_fixations['fname'] = raw_gaze['fname']
            df_fixations['gaze_path'] = raw_gaze['gaze_path']
            # df_fixations = df_fixations.drop(['z', 'std', 'end_time'], axis=1)
            df_fixations.to_csv(output_path)
            print("Processed:", k,  output_path)
