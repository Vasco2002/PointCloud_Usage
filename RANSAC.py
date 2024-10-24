import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA

# Carregar a nuvem de pontos a partir de um ficheiro .ply
pcd = o3d.io.read_point_cloud("Armadillo.ply")  # Atualiza o caminho e nome do ficheiro

# Visualizar a nuvem de pontos original
print("Visualizando a nuvem de pontos original...")
o3d.visualization.draw_geometries([pcd])

# Segmentação de plano usando RANSAC
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# Visualizar o plano identificado
print("Visualizando o plano identificado...")
o3d.visualization.draw_geometries([inlier_cloud])

# Visualizar a nuvem de pontos restantes
print("Visualizando a nuvem de pontos restantes (outliers)...")
o3d.visualization.draw_geometries([outlier_cloud])

# Converter a nuvem de pontos restantes para um array NumPy
points_outliers = np.asarray(outlier_cloud.points)

# Aplicar PCA para encontrar os principais eixos de variação (dimensões)
pca = PCA(n_components=3)
pca.fit(points_outliers)

# Os vetores próprios (axes) são os eixos principais (X, Y, Z)
axes = pca.components_

# As variâncias explicadas indicam as dimensões principais ao longo de cada eixo
dimensions = 2 * np.sqrt(pca.explained_variance_)

# Mostrar os resultados
print(f"Eixos principais:\n{axes}")
print(f"Dimensões do objeto (em unidades do modelo):\nX: {dimensions[0]}, Y: {dimensions[1]}, Z: {dimensions[2]}")

# Visualizar a nuvem de pontos restantes com bounding box ajustada
bounding_box = outlier_cloud.get_axis_aligned_bounding_box()
bounding_box.color = (1, 0, 0)  # Definir cor da bounding box como vermelha
o3d.visualization.draw_geometries([outlier_cloud, bounding_box])

# Exibir as dimensões visualmente com a bounding box
print("Dimensões da caixa delimitadora ajustada (Bounding Box):", bounding_box.get_extent())
