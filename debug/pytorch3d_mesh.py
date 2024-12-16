import os
import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    OpenGLPerspectiveCameras,
)

# Устройство для рендеринга (GPU, если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Путь к файлу OBJ для модели кролика
obj_file = "/home/angelika/Downloads/km-caspian-sea-monster/source/km.zip/km.obj"  # Путь к файлу .obj
mtl_file = "/home/angelika/Downloads/km-caspian-sea-monster/source/km.zip/km.mtl"  # Путь к файлу .mtl

# Загружаем объект с текстурами
mesh = load_objs_as_meshes([obj_file], device=device)

# Настройка камеры
# Изменяем позицию камеры, чтобы отодвинуть её дальше от модели
R, T = torch.eye(3)[None, ...].to(device), torch.tensor([[0, 0, 100.0]]).to(device)  # Отдаляем на 10 единиц по оси Z
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

# Настройка света
lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

# Настройки растрирования
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1
)

# Создаем рендерер с Phong shading
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# Выполняем рендеринг
image = renderer(mesh)
image = image[0, ..., :3].cpu().numpy()  # Извлекаем RGB

# Отображаем изображение
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()
