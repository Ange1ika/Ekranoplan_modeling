import os
import torch
import plotly.graph_objects as go
from pytorch3d.io import load_objs_as_meshes

# Устройство для рендеринга (GPU, если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Путь к файлу OBJ для модели
obj_file = "/home/angelika/Downloads/km-caspian-sea-monster/source/km.zip/km.obj"
#mtl_file = "/home/angelika/Downloads/km-caspian-sea-monster/source/km.zip/km.mtl"  # Путь к файлу .mtl
# Загружаем объект
mesh = load_objs_as_meshes([obj_file], device=device)

# Получаем данные вершин и треугольников
verts = mesh.verts_packed().cpu().numpy()
faces = mesh.faces_packed().cpu().numpy()

# Создаем 3D фигуру с использованием Plotly
fig = go.Figure(data=[
    go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',  # Цвет модели
        opacity=0.5
    )
])

# Настройки камеры и отображения
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ),
    title="3D Visualization of the Model",
    margin=dict(l=0, r=0, b=0, t=0)
)

fig.show()
