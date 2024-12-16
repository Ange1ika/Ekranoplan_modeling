import torch
import pyvista as pv
import numpy as np
from pytorch3d.io import load_objs_as_meshes

# Устройство для рендеринга (GPU, если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Путь к файлу OBJ для модели экраноплана
obj_file = "/home/angelika/Downloads/km-caspian-sea-monster/source/km.zip/km.obj"  # Путь к файлу .obj

# Загружаем объект с текстурами

mesh = load_objs_as_meshes([obj_file], device=device)

# Получаем данные вершин и треугольников
verts = mesh.verts_packed().cpu().numpy()
faces = mesh.faces_packed().cpu().numpy()

# PyVista ожидает, что данные треугольников будут представлены в одномерном массиве
faces_pv = np.hstack([[3] + list(face) for face in faces])

# Создаем объект PolyData для 3D визуализации
poly_data = pv.PolyData(verts, faces_pv)

# Создаем интерактивное окно с PyVista
plotter = pv.Plotter()
plotter.add_mesh(poly_data, color="lightblue", opacity=0.8, show_edges=True)
plotter.show_grid()  # Показываем сетку для наглядности
plotter.view_xy()  # Задаем начальное положение камеры
plotter.show(title="3D Visualization of Ekranoplan")
