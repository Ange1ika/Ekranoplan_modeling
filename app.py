from flask import Flask, render_template, jsonify, request, send_from_directory

import numpy as np
import threading
import time
import json


app = Flask(__name__)

# Глобальные переменные для состояния и параметров симуляции


state = np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10], dtype=np.float64) # Начальное состояние
dt = 0.05  # Шаг времени
simulation_running = False

params = {
    "d": 0,
    "Fthrust": 100,
    "H": 10
}

def H_func(z):
    return max(0, z)  


@app.route('/get_state', methods=['GET'])
def get_state():
    global state
    position = state[9:12].tolist()  # Преобразование в список
    rotation = state[6:9].tolist() 
    
    # Преобразование углов Эйлера в кватернион
    rotation_quaternion = quaternion_from_euler(rotation[0], rotation[1], rotation[2])

    return jsonify({"position": position, "rotation": rotation_quaternion})
    
def quaternion_from_euler(phi, theta, psi):
    """Преобразует углы Эйлера (phi, theta, psi) в кватернион [x, y, z, w].
    Углы в радианах.
        phi: крен (вращение вокруг оси X)
        theta: тангаж (вращение вокруг оси Y)
        psi: рыскание (вращение вокруг оси Z) """

    cy = np.cos(psi * 0.5);
    sy = np.sin(psi * 0.5);
    cp = np.cos(theta * 0.5);
    sp = np.sin(theta * 0.5);
    cr = np.cos(phi * 0.5);
    sr = np.sin(phi * 0.5);

    w = cr * cp * cy + sr * sp * sy;
    x = sr * cp * cy - cr * sp * sy;
    y = cr * sp * cy + sr * cp * sy;
    z = cr * cp * sy - sr * sp * cy;

    return [x, y, z, w]  # Важно: порядок x, y, z, w!




def dynamics(t, state, d, Fthrust, H_func):
    u, v, w, p, q, r, phi, psi, theta, x, y_pos, z = state
    
    # Clamp theta and phi to avoid invalid values
    theta = np.clip(theta, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    phi = np.clip(phi, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    
    # Clamp u, v, w to prevent large values
    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    w = np.clip(w, -100, 100)
    
    
    # Compute velocity squared (use np.clip to prevent overflow)
    vel2 = np.clip(u ** 2 + v ** 2 + w ** 2, 0, 10000)  # Limiting the squared velocity

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
     
    X = np.clip(Fthrust * np.cos(theta) - D * np.cos(theta) - Xg, -1000, 1000)
    Y = np.clip(Yg, -1000, 1000)
    Z = np.clip(-L * np.cos(theta) - Fthrust * np.sin(theta) + Zg, -1000, 1000)

    du = X / m + r * v - q * w
    dv = Y / m - r * u + p * w
    dw = Z / m - p * v + q * u

    My = 0 #0.71 * (Cl + 5 * d) * vel2 - m2 * g * 1.5
    Mx = 0
    Mz = 0
    print(u)
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

    return np.array([du, dv, dw, dp, dq, dr, dphi, dpsi, dtheta, dx, dy, dz], dtype=np.float64)

@app.route('/')
def index():
    return render_template('index.html')  


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        directory='static', 
        path='favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

stop_event = threading.Event()

def simulation_loop():
    global state, simulation_running
    while not stop_event.is_set():
        if simulation_running:
            t = 0
            state_dot = dynamics(t, state, params["d"], params["Fthrust"], H_func)
            state += state_dot * dt
            state = np.array(state, dtype=np.float64)
        time.sleep(dt)



@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    global state, simulation_running, params
    data = request.json

    initial_state = data.get("initial_state", state)
    if len(initial_state) != 12:
        return jsonify({"error": "Invalid initial_state length"}), 400
    state = np.array(initial_state, dtype=np.float64)


    # Инициализация параметров и состояния
    #state = np.array(data.get("initial_state", state), dtype=np.float64)
    params["d"] = data.get("d", params["d"])
    params["Fthrust"] = data.get("Fthrust", params["Fthrust"])
    params["H"] = data.get("H", params["H"])

    simulation_running = True
    return jsonify({"status": "simulation_started"})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    global simulation_running
    simulation_running = False
    return jsonify({"status": "simulation_stopped"})

if __name__ == '__main__':
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    try:
        app.run(debug=True)
    finally:
        stop_event.set()
