import sys
import numpy as np
from scipy.integrate import solve_ivp
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sympy.integrals.risch import NonElementaryIntegral
from mpl_toolkits.mplot3d import Axes3D



# Функция для загрузки OBJ-файла
def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Вершины
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith('f '):  # Грани
                parts = line.split()
                face = [int(parts[1].split('/')[0]) - 1,
                        int(parts[2].split('/')[0]) - 1,
                        int(parts[3].split('/')[0]) - 1]
                faces.append(face)
    return np.array(vertices), np.array(faces)

# Функция для масштабирования вершин
def scale_vertices(vertices, scale):
    return vertices * scale

# Функция для поворота вершин на углы (a1, a2, a3)
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


class EkranoplanApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Симулятор полёта экраноплана')
        self.setGeometry(100, 100, 1200, 800)
        self.mode = 'auto'

        # Начальные значения для управления мышью
        self.is_mouse_pressed = False
        self.mouse_start_pos = None
        self.azimuth = 0  # Начальный угол по азимуту
        self.elevation = 30  # Начальный угол по высоте

        # Параметры
        self.d = 0
        self.Fthrust = 50
        self.H = 10
        S = 0.05    # Масштаб
        filename = "km.obj"

        self.vertices, self.faces = load_obj(filename)
        self.scaled_vertices = scale_vertices(self.vertices, S)
        self.face_collection = None

        self.create_widgets()
        self.show()


        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_simulation)

        self.simulation_running = False

    def rotate_aircfart(self, x, y, z, a1=90, a2=0, a3=180):
        offset = np.array([x, y, z])
        # a1, a2, a3 = 90, 0, 90  # Углы поворота в градусах
        rotated_vertices = rotate_vertices(self.scaled_vertices, (a1, a2, a3))
        translated_vertices = rotated_vertices + offset
        polygons = []
        for face in self.faces:  # Берем каждую 10-ю грань для уменьшения нагрузки, можно изменить на faces для всех граней
            face_vertices = translated_vertices[face]
            polygons.append(face_vertices)
        self.face_collection = Poly3DCollection(polygons, facecolor='lightblue', edgecolor='k', alpha=0.5)

    def create_widgets(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)

        control_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_panel, 1)

        self.auto_button = QtWidgets.QRadioButton('Автоматический режим')
        self.manual_button = QtWidgets.QRadioButton('Ручное управление')
        self.auto_button.setChecked(True)
        control_panel.addWidget(self.auto_button)
        control_panel.addWidget(self.manual_button)

        self.d_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.d_slider.setMinimum(-10)
        self.d_slider.setMaximum(10)
        self.d_slider.setValue(0)
        control_panel.addWidget(QtWidgets.QLabel('Угол отклонения рулей (d)'))
        control_panel.addWidget(self.d_slider)

        self.Fthrust_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.Fthrust_slider.setMinimum(0)
        self.Fthrust_slider.setMaximum(100)
        self.Fthrust_slider.setValue(50)
        control_panel.addWidget(QtWidgets.QLabel('Тяга (Fthrust)'))
        control_panel.addWidget(self.Fthrust_slider)

        control_panel.addWidget(QtWidgets.QLabel('Начальная скорость (u0)'))
        self.u0_input = QtWidgets.QLineEdit('10')
        control_panel.addWidget(self.u0_input)

        control_panel.addWidget(QtWidgets.QLabel('Начальная высота (z0)'))
        self.z0_input = QtWidgets.QLineEdit('0')
        control_panel.addWidget(self.z0_input)

        self.start_button = QtWidgets.QPushButton('Начать симуляцию')
        control_panel.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button = QtWidgets.QPushButton('Остановить симуляцию')
        control_panel.addWidget(self.stop_button)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)

        control_panel.addWidget(QtWidgets.QLabel('Сообщения:'))
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        control_panel.addWidget(self.log_output)

        # Кнопки для смены вида
        view_buttons_layout = QtWidgets.QHBoxLayout()
        self.x_view_button = QtWidgets.QPushButton('Y')
        self.y_view_button = QtWidgets.QPushButton('X')
        self.z_view_button = QtWidgets.QPushButton('Z')
        view_buttons_layout.addWidget(self.x_view_button)
        view_buttons_layout.addWidget(self.y_view_button)
        view_buttons_layout.addWidget(self.z_view_button)
        control_panel.addLayout(view_buttons_layout)

        # Привязка событий
        self.x_view_button.clicked.connect(lambda: self.change_view('x'))
        self.y_view_button.clicked.connect(lambda: self.change_view('y'))
        self.z_view_button.clicked.connect(lambda: self.change_view('z'))

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, 3)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.param_figure, self.param_axes = plt.subplots(2, 2, figsize=(5, 4))
        self.param_canvas = FigureCanvas(self.param_figure)
        main_layout.addWidget(self.param_canvas, 2)

    def change_view(self, axis):
        """Изменяет вид камеры на оси X, Y или Z."""
        if axis == 'x':
            self.azimuth = 0
            self.elevation = 0
        elif axis == 'y':
            self.azimuth = 90
            self.elevation = 0
        elif axis == 'z':
            self.azimuth = 0
            self.elevation = 90
        self.plot_trajectory()

    def on_mouse_press(self, event):
        if event.button == 1:  # Левая кнопка мыши
            self.is_mouse_pressed = True
            self.mouse_start_pos = (event.x, event.y)

    def on_mouse_release(self, event):
        if event.button == 1:  # Левая кнопка мыши
            self.is_mouse_pressed = False
            self.mouse_start_pos = None

    def on_mouse_move(self, event):
        if self.is_mouse_pressed:
            dx = event.x - self.mouse_start_pos[0]
            dy = event.y - self.mouse_start_pos[1]

            # Изменяем азимут и высоту
            self.azimuth += dx * 0.1  # Умножаем на коэффициент для плавности
            self.elevation = np.clip(self.elevation - dy * 0.1, -90, 90)

            # Обновляем начальную позицию мыши
            self.mouse_start_pos = (event.x, event.y)

            # Перерисовываем траекторию с новым углом обзора
            self.plot_trajectory()

    def start_simulation(self):
        self.d = self.d_slider.value()
        self.Fthrust = self.Fthrust_slider.value()
        self.H = 10

        try:
            u0 = float(self.u0_input.text())
            z0 = float(self.z0_input.text())
        except ValueError:
            self.log_output.append('Ошибка: неверные начальные условия.')
            return

        v0, w0 = 0, 0
        p0, q0, r0 = 0, 0, 0
        phi0, psi0, theta0 = 0, 0, 0
        x0, y0 = 0, 0

        self.initial_state = [u0, v0, w0, p0, q0, r0, phi0, psi0, theta0, x0, y0, z0]

        self.t_span = (0, 1e9)

        self.simulation_running = True

        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

        self.current_time = 0
        self.state = self.initial_state.copy()
        self.t_data = []
        self.state_data = []

        self.timer.start(50)

        if self.auto_button.isChecked():
            self.mode = 'auto'
            self.d_func = lambda t: self.d
            self.Fthrust_func = lambda t: self.Fthrust
        else:
            self.mode = 'manual'
            self.d_func = lambda t: self.d
            self.Fthrust_func = lambda t: self.Fthrust

        self.setFocus()

    def stop_simulation(self):
        self.simulation_running = False
        self.timer.stop()
        self.log_output.append('Симуляция остановлена пользователем.')
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

    def update_simulation(self):
        if not self.simulation_running:
            return

        if self.mode == 'manual':
            d = self.d
            Fthrust = self.Fthrust
        else:
            d = self.d_func(self.current_time)
            Fthrust = self.Fthrust_func(self.current_time)

        H_func = lambda z: self.H

        dt = 0.05
        t_span = (self.current_time, self.current_time + dt)

        try:
            sol = solve_ivp(lambda t, y: dynamics(t, y, d, Fthrust, H_func),
                            t_span, self.state, method='Radau', max_step=dt)
            if not sol.success:
                raise RuntimeError('Интегратор не смог выполнить шаг.')
        except Exception as e:
            self.log_output.append(f'Ошибка при интегрировании на шаге времени {self.current_time:.2f}: {e}')
            self.stop_simulation()
            return

        self.current_time += dt
        self.state = sol.y[:, -1]

        self.t_data.append(self.current_time)
        self.state_data.append(self.state.copy())

        self.plot_trajectory()
        self.plot_parameters()

   
   
    def plot_trajectory(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        # Установка углов азимута и высоты
        ax.view_init(elev=self.elevation, azim=self.azimuth)

        state_array = np.array(self.state_data)
        x = state_array[:, 9]
        y_pos = state_array[:, 10]
        z = state_array[:, 11]

        # Определяем динамические границы графика
        margin = 5  # Отступ для визуального удобства
        x_min, x_max = np.min(x) - margin, np.max(x) + margin
        y_min, y_max = np.min(y_pos) - margin, np.max(y_pos) + margin
        z_min, z_max = np.min(z) - margin, np.max(z) + margin

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Рисуем самолет
        self.rotate_aircfart(x[-1], y_pos[-1], z[-1])  # Рисуем модель
        ax.add_collection3d(self.face_collection)

        # Вода как прямоугольный параллелепипед
        z_water_max = -2  # Граница воды

        # Нижняя грань воды (z = z_min)
        X_water = np.array([[x_min, x_max], [x_min, x_max]])
        Y_water = np.array([[y_min, y_min], [y_max, y_max]])
        Z_water = np.array([[z_min, z_min], [z_min, z_min]])
        ax.plot_surface(X_water, Y_water, Z_water, color='blue', alpha=0.7, rstride=1, cstride=1)

        # Верхняя грань воды (z = z_water_max)
        X_water_top = np.array([[x_min, x_max], [x_min, x_max]])
        Y_water_top = np.array([[y_min, y_min], [y_max, y_max]])
        Z_water_top = np.array([[z_water_max, z_water_max], [z_water_max, z_water_max]])
        ax.plot_surface(X_water_top, Y_water_top, Z_water_top, color='blue', alpha=0.7, rstride=1, cstride=1)

        # Боковые грани воды по оси X
        # Левая боковая грань воды (x = x_min)
        X_water_side = np.array([[x_min, x_min], [x_min, x_min]])  # x = const
        Y_water_side = np.array([[y_min, y_min], [y_max, y_max]])  # y = между y_min и y_max
        Z_water_side = np.array([[z_min, z_water_max], [z_min, z_water_max]])  # z = между z_min и z_water_max
        ax.plot_surface(X_water_side, Y_water_side, Z_water_side, color='blue', alpha=0.7)

        # Правая боковая грань воды (x = x_max)
        X_water_side = np.array([[x_max, x_max], [x_max, x_max]])  # x = const
        Y_water_side = np.array([[y_min, y_min], [y_max, y_max]])  # y = между y_min и y_max
        Z_water_side = np.array([[z_min, z_water_max], [z_min, z_water_max]])  # z = между z_min и z_water_max
        ax.plot_surface(X_water_side, Y_water_side, Z_water_side, color='blue', alpha=0.7)

        # Вторая пара боковых граней воды по оси Y
        # Левая боковая грань воды (y = y_min)
        X_water_side = np.array([[x_min, x_max], [x_min, x_max]])  # x = между x_min и x_max
        Y_water_side = np.array([[y_min, y_min], [y_min, y_min]])  # y = const
        Z_water_side = np.array([[z_min, z_min], [z_water_max, z_water_max]])  # z = между z_min и z_water_max
        ax.plot_surface(X_water_side, Y_water_side, Z_water_side, color='blue', alpha=0.7)

        # Правая боковая грань воды (y = y_max)
        X_water_side = np.array([[x_min, x_max], [x_min, x_max]])  # x = между x_min и x_max
        Y_water_side = np.array([[y_max, y_max], [y_max, y_max]])  # y = const
        Z_water_side = np.array([[z_min, z_min], [z_water_max, z_water_max]])  # z = между z_min и z_water_max
        ax.plot_surface(X_water_side, Y_water_side, Z_water_side, color='blue', alpha=0.7)

        z_sky_max = z_water_max + 0.1
        # Небо как прямоугольный параллелепипед
        # Нижняя грань неба (z = z_water_max)
        X_sky = np.array([[x_min, x_max], [x_min, x_max]])
        Y_sky = np.array([[y_min, y_min], [y_max, y_max]])
        Z_sky = np.array([[z_sky_max, z_sky_max], [z_sky_max, z_sky_max]])
        ax.plot_surface(X_sky, Y_sky, Z_sky, color='skyblue', alpha=0.5, rstride=1, cstride=1)

        # Верхняя грань неба (z = z_max)
        X_sky_top = np.array([[x_min, x_max], [x_min, x_max]])
        Y_sky_top = np.array([[y_min, y_min], [y_max, y_max]])
        Z_sky_top = np.array([[z_max, z_max], [z_max, z_max]])
        ax.plot_surface(X_sky_top, Y_sky_top, Z_sky_top, color='skyblue', alpha=0.5, rstride=1, cstride=1)

        # Боковые грани неба по оси X
        # Левая боковая грань неба (x = x_min)
        X_sky_side = np.array([[x_min, x_min], [x_min, x_min]])  # x = const
        Y_sky_side = np.array([[y_min, y_min], [y_max, y_max]])  # y = между y_min и y_max
        Z_sky_side = np.array([[z_max, z_sky_max], [z_max, z_sky_max]])  # z = между z_water_max и z_max
        ax.plot_surface(X_sky_side, Y_sky_side, Z_sky_side, color='skyblue', alpha=0.5)

        # Правая боковая грань неба (x = x_max)
        X_sky_side = np.array([[x_max, x_max], [x_max, x_max]])  # x = const
        Y_sky_side = np.array([[y_min, y_min], [y_max, y_max]])  # y = между y_min и y_max
        Z_sky_side = np.array([[z_max, z_sky_max], [z_max, z_sky_max]])  # z = между z_water_max и z_max
        ax.plot_surface(X_sky_side, Y_sky_side, Z_sky_side, color='skyblue', alpha=0.5)

        # Боковые грани неба по оси Y
        # Левая боковая грань неба (y = y_min)
        X_sky_side = np.array([[x_min, x_max], [x_min, x_max]])  # x = между x_min и x_max
        Y_sky_side = np.array([[y_min, y_min], [y_min, y_min]])  # y = const
        Z_sky_side = np.array([[z_sky_max, z_sky_max], [z_max, z_max]])  # z = между z_water_max и z_max
        ax.plot_surface(X_sky_side, Y_sky_side, Z_sky_side, color='skyblue', alpha=0.5)

        # Правая боковая грань неба (y = y_max)
        X_sky_side = np.array([[x_min, x_max], [x_min, x_max]])  # x = между x_min и x_max
        Y_sky_side = np.array([[y_max, y_max], [y_max, y_max]])  # y = const
        Z_sky_side = np.array([[z_sky_max, z_sky_max], [z_max, z_max]])  # z = между z_water_max и z_max
        ax.plot_surface(X_sky_side, Y_sky_side, Z_sky_side, color='skyblue', alpha=0.5)
        
        num_clouds = 5  # Уменьшили количество облаков
        cloud_radius = 0.5  # Меньший радиус облаков
        cloud_x = np.random.uniform(x_min + 2, x_max - 2, num_clouds)
        cloud_y = np.random.uniform(y_min + 2, y_max - 2, num_clouds)
        cloud_z = np.random.uniform(z_water_max + 2, z_max - 2, num_clouds)  # Облака в пределах неба

        for i in range(num_clouds):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x_sphere = cloud_radius * np.outer(np.cos(u), np.sin(v)) + cloud_x[i]
            y_sphere = cloud_radius * np.outer(np.sin(u), np.sin(v)) + cloud_y[i]
            z_sphere = cloud_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + cloud_z[i]
            
            # Генерация случайных искажений для формы облаков
            x_sphere += np.random.normal(0, 0.5, x_sphere.shape)  # Небольшие случайные искажения по оси X
            y_sphere += np.random.normal(0, 0.5, y_sphere.shape)  # Небольшие случайные искажения по оси Y
            z_sphere += np.random.normal(0, 0.5, z_sphere.shape)  # Небольшие случайные искажения по оси Z

            ax.plot_surface(x_sphere, y_sphere, z_sphere, color='white', alpha=0.3, rstride=3, cstride=3)

        # Рисуем траекторию
        ax.plot(x, y_pos, z, label='Траектория полёта', color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.legend()
        self.canvas.draw()


    def plot_parameters(self):
        self.param_figure.clear()
        axes = self.param_figure.subplots(2, 2)

        t = self.t_data
        state_array = np.array(self.state_data)

        u = state_array[:, 0]
        z = state_array[:, 11]
        theta = state_array[:, 8]
        phi = state_array[:, 6]
        speed = np.sqrt(state_array[:, 0]**2 + state_array[:, 1]**2 + state_array[:, 2]**2)
        axes[0, 0].plot(t, speed)
        axes[0, 0].set_title('Скорость')

        axes[0, 1].plot(t, z)
        axes[0, 1].set_title('Высота')

        axes[1, 0].plot(t, theta)
        axes[1, 0].set_title('Угол тангажа (θ)')

        axes[1, 1].plot(t, phi)
        axes[1, 1].set_title('Угол крена (φ)')

        self.param_canvas.draw()

    def keyPressEvent(self, event):
        if self.simulation_running:
            key = event.key()
            if key == Qt.Key_Up:
                self.d += 0.1
                self.log_output.append(f'Увеличение угла отклонения рулей: d = {self.d:.2f}')
            elif key == Qt.Key_Down:
                self.d -= 0.1
                self.log_output.append(f'Уменьшение угла отклонения рулей: d = {self.d:.2f}')
            elif key == Qt.Key_Left:
                self.Fthrust -= 1
                if self.Fthrust < 0:
                    self.Fthrust = 0
                self.log_output.append(f'Уменьшение тяги: Fthrust = {self.Fthrust}')
            elif key == Qt.Key_Right:
                self.Fthrust += 1
                self.log_output.append(f'Увеличение тяги: Fthrust = {self.Fthrust}')
            else:
                super().keyPressEvent(event)
                return

            self.d_slider.setValue(int(self.d))
            self.Fthrust_slider.setValue(int(self.Fthrust))

            self.d_func = lambda t: self.d
            self.Fthrust_func = lambda t: self.Fthrust
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.simulation_running = False
        self.timer.stop()
        event.accept()

def create_3d_model():
    # Пример создания куба, можно заменить загрузкой объекта из файла
    geometry = BoxGeometry(1, 1, 1)
    material = MeshBasicMaterial(color='red')
    model = Mesh(geometry=geometry, material=material)
    return model

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = EkranoplanApp()
    sys.exit(app.exec_())