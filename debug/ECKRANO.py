import torch
import pyvista as pv
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from scipy.integrate import solve_ivp
from PyQt5 import QtWidgets
import sys

# Set up the rendering device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the OBJ file for the ekranoplan model
obj_file =  "/home/angelika/Downloads/km-caspian-sea-monster/source/km.zip/km.obj"  # Путь к файлу .obj

# Load the 3D model
mesh = load_objs_as_meshes([obj_file], device=device)

# Get vertex and triangle data
verts = mesh.verts_packed().cpu().numpy()
faces = mesh.faces_packed().cpu().numpy()

# Prepare faces data for PyVista
faces_pv = np.hstack([[3] + list(face) for face in faces])

# Create PolyData for 3D visualization
poly_data = pv.PolyData(verts, faces_pv)

# Define detailed dynamics function based on main_z.py
def dynamics(t, state, Fthrust, H_func, control_params):
    u, v, w, p, q, r, phi, psi, theta, x, y_pos, z = state

    # Ensure theta and phi are within valid ranges
    theta = np.clip(theta, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    phi = np.clip(phi, -np.pi/2 + 0.01, np.pi/2 - 0.01)

    vel2 = u ** 2 + v ** 2 + w ** 2
    m = 11
    k = 1.0
    epsilon = 0.1

    H = H_func(z)

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
    Y = Yg + control_params["yaw"]  # Adding yaw control
    Z = -L * np.cos(theta) - Zg + control_params["vertical"]  # Adding vertical control

    dx = u + control_params["horizontal"]  # Adding horizontal movement (left/right)
    dy = v
    dz = w

    du = X / m
    dv = Y / m
    dw = Z / m

    dp = (0.1 * r * q) / Ixx + control_params["roll"]  # Adding roll control
    dq = (0.1 * p * r) / Iyy
    dr = (0.1 * q * p) / Izz

    return [du, dv, dw, dp, dq, dr, phi, psi, theta, dx, dy, dz]

# Initialize state
state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # [u, v, w, p, q, r, phi, psi, theta, x, y_pos, z]

# Interactive button-based movement
class ControlWindow(QtWidgets.QWidget):
    def __init__(self, plotter):
        super().__init__()
        self.plotter = plotter
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        # Movement buttons
        buttons = {
            "Increase Thrust": lambda: self.control_movement(thrust=10),
            "Pitch Up": lambda: self.control_movement(vertical=1.0),
            "Move Left": lambda: self.control_movement(horizontal=-1.0),
            "Move Right": lambda: self.control_movement(horizontal=1.0),
            "Yaw Left": lambda: self.control_movement(yaw=-1.0),
            "Yaw Right": lambda: self.control_movement(yaw=1.0),
            "Roll Left": lambda: self.control_movement(roll=-1.0),
            "Roll Right": lambda: self.control_movement(roll=1.0),
            "Move Up": lambda: self.control_movement(vertical=1.0),
            "Move Down": lambda: self.control_movement(vertical=-1.0),
        }
        for text, func in buttons.items():
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(func)
            layout.addWidget(button)

        self.setLayout(layout)

    def control_movement(self, thrust=0, vertical=0, horizontal=0, yaw=0, roll=0):
        global state
        control_params = {"thrust": thrust, "vertical": vertical, "horizontal": horizontal, "yaw": yaw, "roll": roll}
        state = solve_ivp(dynamics, [0, 0.1], state, args=(thrust, lambda z: 1.0, control_params)).y[:,-1]
        self.update_model_position()

    def update_model_position(self):
        # Update the visualization based on the new state
        theta = state[8]
        self.plotter.view_vector((0, 0, 1), viewup=(0, np.sin(theta), np.cos(theta)))
        self.plotter.render()

# PyVista plotter setup
plotter = pv.Plotter()
plotter.add_mesh(poly_data, color="lightblue", opacity=0.8, show_edges=True)
plotter.show_grid()  # Show the grid for reference

# Create control window
app = QtWidgets.QApplication(sys.argv)
control_window = ControlWindow(plotter)
control_window.show()

# Start the visualization
plotter.show(title="3D Visualization of Ekranoplan with Enhanced Controls")
app.exec_()





