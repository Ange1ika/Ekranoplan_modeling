import bpy
import numpy as np
import matplotlib.pyplot as plt

# Загрузка модели экраноплана
def load_ekranoplan():
    file_path = "km.obj" 
    bpy.ops.import_scene.obj(filepath=file_path)
    obj = bpy.context.selected_objects[0]
    obj.name = "Ekranoplan"  # Название объекта для дальнейшего обращения
    obj.location = (0, 0, 0)  # Начальная позиция
    obj.rotation_euler = (0, 0, 0)  # Начальная ориентация


def create_environment():
    # Создание моря
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, -1))
    sea = bpy.context.object
    sea.name = "Sea"
    sea_material = bpy.data.materials.new(name="SeaMaterial")
    sea_material.use_nodes = True
    sea.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.0, 0.2, 0.8, 1)  # Синий цвет
    sea.data.materials.append(sea_material)

    # Создание неба
    bpy.ops.object.light_add(type='SUN', location=(10, 10, 10))

# Функции динамики
def rotate_vertices(vertices, angles):
    a1, a2, a3 = np.radians(angles)
    Rx = np.array([[1, 0, 0], [0, np.cos(a1), -np.sin(a1)], [0, np.sin(a1), np.cos(a1)]])
    Ry = np.array([[np.cos(a2), 0, np.sin(a2)], [0, 1, 0], [-np.sin(a2), 0, np.cos(a2)]])
    Rz = np.array([[np.cos(a3), -np.sin(a3), 0], [np.sin(a3), np.cos(a3), 0], [0, 0, 1]])
    rotation_matrix = Rz @ Ry @ Rx
    return vertices @ rotation_matrix.T


def dynamics(t, state, d, Fthrust, H_func):
    u, v, w, p, q, r, phi, psi, theta, x, y_pos, z = state


    theta = np.clip(theta, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    phi = np.clip(phi, -np.pi/2 + 0.01, np.pi/2 - 0.01)

    vel2 = u ** 2 + v ** 2 + w ** 2
    m = 11
    m2 = 5

    H = H_func(z)

    epsilon = 0.1
    k = 1.0
    Cl = (0.4 + 50 * theta) * (1 + k / (H + epsilon))
    Cd = (0.07 + 7 * theta) * (1 + k / (H + epsilon))

    g = 9.8
    Ixx = 8.93
    Iyy = 11.24
    Izz = 18.9

    D = Cd * vel2
    L = Cl * vel2

    Xg = m * g * np.sin(theta)
    Yg = m * g * np.sin(phi) * np.cos(theta)
    Zg = m * g * np.cos(phi) * np.cos(theta)

    X = Fthrust * np.cos(theta) - D * np.cos(theta) - Xg
    Y = Yg
    Z = -L * np.cos(theta) - Fthrust * np.sin(theta) + Zg

    du = X / m + r * v - q * w
    dv = Y / m - r * u + p * w
    dw = Z / m - p * v + q * u

    My = 0 #0.71 * (Cl + 5 * d) * vel2 - m2 * g * 1.5
    Mx = 0
    Mz = 0

    dp = (q * r * (Iyy - Izz)) / Ixx + Mx / Ixx
    dq = (p * r * (Izz - Ixx)) / Iyy + My / Iyy
    dr = (p * q * (Ixx - Iyy)) / Izz + Mz / Izz

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    tan_theta = np.tan(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    cos_theta = np.clip(cos_theta, 0.01, None)

    dphi = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
    dpsi = q * sin_phi / cos_theta + r * cos_phi / cos_theta
    dtheta = q * cos_phi - r * sin_phi

    dx = (u * cos_theta * np.cos(psi) +
          v * (sin_phi * sin_theta * np.cos(psi) - cos_phi * np.sin(psi)) +
          w * (cos_phi * sin_theta * np.cos(psi) + sin_phi * np.sin(psi)))
    dy = (u * cos_theta * np.sin(psi) +
          v * (sin_phi * sin_theta * np.sin(psi) + cos_phi * np.cos(psi)) +
          w * (cos_phi * sin_theta * np.sin(psi) - sin_phi * np.cos(psi)))
    dz = -u * sin_theta + v * sin_phi * cos_theta + w * cos_phi * cos_theta

    return [du, dv, dw, dp, dq, dr, dphi, dpsi, dtheta, dx, dy, dz]


# Таймер обновления
def update_simulation():
    global state, t, d, Fthrust
    dt = 0.01
    state = dynamics(t, state, d, Fthrust, lambda z: z)
    update_model(state)
    t += dt
    return 0.01  # Интервал обновления

# Обновление модели экраноплана
def update_model(state):
    x, y, z, phi, theta, psi = state[9], state[10], state[11], state[6], state[8], state[7]
    obj = bpy.data.objects['Ekranoplan']
    obj.location = (x, y, z)
    obj.rotation_euler = (phi, theta, psi)

# Пользовательский интерфейс
class EkranoplanPanel(bpy.types.Panel):
    bl_label = "Ekranoplan Control"
    bl_idname = "VIEW3D_PT_ekranoplan"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Ekranoplan'

    def draw(self, context):
        layout = self.layout
        layout.operator("ekranoplan.start_simulation")
        layout.operator("ekranoplan.stop_simulation")

# Операторы
class StartSimulationOperator(bpy.types.Operator):
    bl_idname = "ekranoplan.start_simulation"
    bl_label = "Start Simulation"
    def execute(self, context):
        bpy.app.timers.register(update_simulation)
        return {'FINISHED'}

class StopSimulationOperator(bpy.types.Operator):
    bl_idname = "ekranoplan.stop_simulation"
    bl_label = "Stop Simulation"
    def execute(self, context):
        bpy.app.timers.unregister(update_simulation)
        return {'FINISHED'}

# Регистрация
def register():
    bpy.utils.register_class(EkranoplanPanel)
    bpy.utils.register_class(StartSimulationOperator)
    bpy.utils.register_class(StopSimulationOperator)

def unregister():
    bpy.utils.unregister_class(EkranoplanPanel)
    bpy.utils.unregister_class(StartSimulationOperator)
    bpy.utils.unregister_class(StopSimulationOperator)

def create_environment():
    # Создание моря
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, -1))
    sea = bpy.context.object
    sea.name = "Sea"
    sea_material = bpy.data.materials.new(name="SeaMaterial")
    sea_material.use_nodes = True
    sea.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.0, 0.2, 0.8, 1)  # Синий цвет
    sea.data.materials.append(sea_material)

    # Создание неба
    bpy.ops.object.light_add(type='SUN', location=(10, 10, 10))


if __name__ == "__main__":
    load_ekranoplan()          # Загрузка модели экраноплана
    add_hdri_environment()     # Добавление HDRI или создайте вручную с create_environment()
    register()   