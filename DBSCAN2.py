import argparse
import open3d as o3d
import open3d.core as o3c
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Configurar argumentos
parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Arquivo com pontos 3D e confiança')
parser.add_argument('confidence_threshold', type=float, help='Limiar de confiança')
parser.add_argument('--eps', type=float, default=0.05, help='Distância máxima entre pontos para clustering')
parser.add_argument('--min_samples', type=int, default=3, help='Número mínimo de pontos por cluster')
args = parser.parse_args()

# Carregar e filtrar os pontos
with open(args.filename) as f:
    points = []
    for line in f:
        x, y, z, confidence = map(float, line.split())
        if confidence >= args.confidence_threshold:
            points.append([x, y, z])

points = np.array(points)

# Aplicar DBSCAN
db = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(points)
labels = db.labels_

# Identificar o cluster com maior número de pontos (ignorar ruído: label = -1)
unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
largest_cluster_label = unique_labels[np.argmax(counts)]  # Label do maior cluster

# Filtrar pontos do maior cluster
largest_cluster_points = points[labels == largest_cluster_label]

# Criar nuvem de pontos do maior cluster
pcd = o3d.t.geometry.PointCloud(o3c.Tensor(largest_cluster_points, o3c.float32))
legacy_pcd = pcd.to_legacy()

# Adicionar cor ao maior cluster
cluster_color = np.array([[1, 0, 0]])  # Cor vermelha
point_colors = np.tile(cluster_color, (largest_cluster_points.shape[0], 1))
legacy_pcd.colors = o3d.utility.Vector3dVector(point_colors)

# Adicionar a bounding box
bounding_box = legacy_pcd.get_axis_aligned_bounding_box()
bounding_box.color = (0, 1, 0)  # Cor verde para a bounding box

# Exibir a nuvem de pontos com a bounding box
o3d.visualization.draw_geometries([legacy_pcd, bounding_box])

# Imprimir as dimensões da bounding box
min_bound = bounding_box.min_bound  # Ponto de mínimo (x, y, z)
max_bound = bounding_box.max_bound  # Ponto de máximo (x, y, z)
print(f"Dimensões da bounding box (em unidades do modelo):")
print(f"X: {max_bound[0] - min_bound[0]}")
print(f"Y: {max_bound[1] - min_bound[1]}")
print(f"Z: {max_bound[2] - min_bound[2]}")
