import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

# Carregar a nuvem de pontos a partir de um ficheiro .ply
pcd = o3d.io.read_point_cloud("Armadillo.ply")  # Verifique o caminho

# Verificar se a nuvem de pontos contém dados
if len(pcd.points) == 0:
    print("A nuvem de pontos está vazia. Verifique o arquivo.")
else:
    print(f"Número de pontos na nuvem: {len(pcd.points)}")

# Visualizar a nuvem de pontos original
print("Visualizando a nuvem de pontos original...")
o3d.visualization.draw_geometries([pcd])

# Converter a nuvem de pontos para um array NumPy
points = np.asarray(pcd.points)

# Aplicar DBSCAN para segmentar a nuvem de pontos
# Tente ajustar os parâmetros aqui
db = DBSCAN(eps=0.05, min_samples=3).fit(points)  # Ajuste os parâmetros conforme necessário
labels = db.labels_

# Criar nuvens de pontos para cada cluster
unique_labels = set(labels)
if len(unique_labels) == 1 and -1 in unique_labels:
    print("Nenhum cluster encontrado, todos os pontos são considerados ruído.")
else:
    clusters = []
    for label in unique_labels:
        if label != -1:  # Ignorar os ruídos
            cluster_points = points[labels == label]
            clusters.append(cluster_points)

    # Visualizar cada cluster
    for i, cluster in enumerate(clusters):
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
        print(f"Visualizando o cluster {i + 1}...")
        o3d.visualization.draw_geometries([cluster_pcd])

# Aguarde o fechamento da janela de visualização
input("Pressione Enter para continuar...")
