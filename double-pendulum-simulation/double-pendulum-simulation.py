import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81 # m/s^2
m1 = 2.0 # mass rod 1 in kg
m2 = 1.0 # mass rod 2 in kg
l1 = 1.2 # length rod 1 in m
l2 = 0.6 # length rod 2 in m

# 4th order Runge-Kutta
def RK4(y, t, dt):
    k1 = dYdt(y) * dt
    k2 = dYdt(y + k1 / 2) * dt                              
    k3 = dYdt(y + k2 / 2) * dt
    k4 = dYdt(y + k3) * dt

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# dampenening coefficients
c1 = 0.3  # 1/s
c2 = 0.2   # 1/s

# system of differential equations 
def dYdt(y):
    theta1, theta2, omega1, omega2 = y
    dTheta1_dt = omega1
    dTheta2_dt = omega2

    dOmega1_dt = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * m2 * 
                    (omega2**2 * l2 + omega1**2 * l1 * np.cos(theta1 - theta2))) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    
    dOmega2_dt = (2 * np.sin(theta1 - theta2) * (omega1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + 
                    omega2**2 * l2 * m2 * np.cos(theta1 - theta2))) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    
    dOmega1_dt -= c1 * omega1
    dOmega2_dt -= c2 * omega2

    return np.array([dTheta1_dt, dTheta2_dt, dOmega1_dt, dOmega2_dt])

theta1_0 = np.pi  # rod 1 initial angle, radians
theta2_0 = np.pi / 6 # rod 2 initial angle, radians
omega1_0 = 0.0 # rod 1 initial angular velocity, rad/s
omega2_0 = 0.0 # rod 2 initial angular velocity, rad/s

y0 = np.array([theta1_0, theta2_0, omega1_0, omega2_0])

t_end = 60
dt = 0.01
n = int(t_end / dt)
t = np.linspace(0, t_end, n)

y = np.zeros((n, 4))
y[0] = y0

for i in range(1, n):
    y[i] = RK4(y[i - 1], t[i - 1], dt)

theta1, theta2, omega1, omega2 = y.T


# matplotlib animation code below

x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_aspect("equal", adjustable="box")

L = l1 + l2
pad = 0.6
ax.set_xlim(-L - pad, L + pad)
ax.set_ylim(-L - pad, L + pad)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
span = max(xmax - xmin, ymax - ymin)

xc = 0.5 * (xmin + xmax)
yc = 0.5 * (ymin + ymax)

ax.set_xlim(xc - span/2, xc + span/2)
ax.set_ylim(yc - span/2, yc + span/2)

line, = ax.plot([], [], lw=2)
masses, = ax.plot([], [], "o", markersize=6)

trail, = ax.plot([], [], lw=1, alpha=0.6)
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
trail_x = []
trail_y = []
trail_len = 300

k = 1 # animate every k-th timestep

def start():
    line.set_data([], [])
    masses.set_data([], [])
    trail.set_data([], [])
    time_text.set_text("")

    return line, masses, trail, time_text

def update(frame):
    i = frame * k

    if i >= len(t):
        i = len(t) - 1

    line.set_data([0.0, x1[i], x2[i]], [0.0, y1[i], y2[i]])
    masses.set_data([x1[i], x2[i]], [y1[i], y2[i]])

    trail_x.append(x2[i])
    trail_y.append(y2[i])

    if len(trail_x) > trail_len:
        del trail_x[0]
        del trail_y[0]

    trail.set_data(trail_x, trail_y)

    time_text.set_text("t = %.2f s" % t[i])
    return line, masses, trail, time_text

frames = (len(t) + k - 1) // k
interval_ms = 1000 * dt * k

ani = FuncAnimation(fig, update, frames=frames, init_func=start, interval=interval_ms, blit=True)

plt.show()

# angle plot
plt.plot(t, theta1, label="Theta1")
plt.plot(t, theta2, label="Theta2")
plt.xlabel("t")
plt.ylabel("angle")
plt.legend()
plt.show()
