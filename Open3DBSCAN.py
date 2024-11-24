import argparse
import open3d as o3d
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Arquivo com pontos 3D e confiança')
parser.add_argument('confidence_threshold', type=float, help='Limiar de confiança')
parser.add_argument('--eps', type=float, default=0.05, help='Distância máxima entre pontos para clustering')
parser.add_argument('--min_points', type=int, default=3, help='Número mínimo de pontos por cluster')
args = parser.parse_args()

with open(args.filename) as f:
    points = []
    for line in f:
        x, y, z, confidence = map(float, line.split())
        if confidence >= args.confidence_threshold:
            points.append([x, y, z])

points = np.array(points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

labels = np.array(pcd.cluster_dbscan(eps=args.eps, min_points=args.min_points, print_progress=True))

unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
largest_cluster_label = unique_labels[np.argmax(counts)]

largest_cluster_points = points[labels == largest_cluster_label]

largest_cluster_pcd = o3d.geometry.PointCloud()
largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)

cluster_color = np.array([[1, 0, 0]]) 
point_colors = np.tile(cluster_color, (largest_cluster_points.shape[0], 1))
largest_cluster_pcd.colors = o3d.utility.Vector3dVector(point_colors)

bounding_box = largest_cluster_pcd.get_axis_aligned_bounding_box()
bounding_box.color = (0, 1, 0) 

o3d.visualization.draw_geometries([largest_cluster_pcd, bounding_box])

min_bound = bounding_box.min_bound 
max_bound = bounding_box.max_bound 
print(f"Dimensões da bounding box (em unidades do modelo):")
print(f"X: {(max_bound[0] - min_bound[0])*100} cm")
print(f"Y: {(max_bound[1] - min_bound[1])*100} cm")
print(f"Z: {(max_bound[2] - min_bound[2])*100} cm")
