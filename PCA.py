import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carregar a nuvem de pontos a partir de um ficheiro .ply
pcd = o3d.io.read_point_cloud("Armadillo.ply")

# Converter a nuvem de pontos para um array NumPy
points = np.asarray(pcd.points)

# Visualizar a nuvem de pontos original
o3d.visualization.draw_geometries([pcd])

# Aplicar PCA para encontrar os principais eixos de variação (dimensões)
pca = PCA(n_components=3)
pca.fit(points)

# Os vetores próprios (axes) são os eixos principais (X, Y, Z)
axes = pca.components_

# As variâncias explicadas indicam as dimensões principais ao longo de cada eixo
dimensions = 2 * np.sqrt(pca.explained_variance_)

# Mostrar os resultados
print(f"Eixos principais:\n{axes}")
print(f"Dimensões do objeto (em unidades do modelo):\nX: {dimensions[0]}, Y: {dimensions[1]}, Z: {dimensions[2]}")

# Visualizar a nuvem de pontos com bounding box ajustada
bounding_box = pcd.get_axis_aligned_bounding_box()
bounding_box.color = (1, 0, 0)  # Definir cor da bounding box como vermelha
o3d.visualization.draw_geometries([pcd, bounding_box])

# Exibir as dimensões visualmente com a bounding box
print("Dimensões da caixa delimitadora ajustada (Bounding Box):", bounding_box.get_extent())
