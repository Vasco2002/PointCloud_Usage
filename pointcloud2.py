import argparse
import open3d as o3d
import open3d.core as o3c
import numpy as np
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('confidence_threshold')
args = parser.parse_args()

with open(args.filename) as f:
    confidence_threshold = float(args.confidence_threshold)
    points = []
    for line in f:
        x, y, z, confidence = map(float, line.split())
        if confidence >= confidence_threshold:
            points.append([x, y, z])

pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points, o3c.float32))
legacy_pcd = pcd.to_legacy()

points_np = np.asarray(legacy_pcd.points)
pca = PCA(n_components=3)
pca.fit(points_np)
axes = pca.components_
dimensions = 2 * np.sqrt(pca.explained_variance_)

print(f"Eixos principais:\n{axes}")
print(f"Dimensões (em unidades do modelo):\nX: {dimensions[0]}, Y: {dimensions[1]}, Z: {dimensions[2]}")

bounding_box = legacy_pcd.get_axis_aligned_bounding_box()
bounding_box.color = (1, 0, 0)
o3d.visualization.draw_geometries([legacy_pcd, bounding_box])
print("Dimensões da bounding box ajustada:", bounding_box.get_extent())
