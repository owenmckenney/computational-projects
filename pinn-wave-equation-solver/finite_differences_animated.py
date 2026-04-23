import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# physical parameters
Lx = 1.0
Ly = 1.0
c = 1.0
T = 1

# grid size
Nx = 51
Ny = 51

# initial condition
x0 = 0.2 * Lx
y0 = 0.2 * Ly
sigma = 0.08 * min(Lx, Ly)
amplitude = 1.0

# use a little less than the stability limit
safety_factor = 0.95


dx = Lx / float(Nx - 1)
dy = Ly / float(Ny - 1)

x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")


u0 = amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (sigma ** 2))
v0 = np.zeros_like(X)


# fixed boundaries
u0[0, :] = 0.0
u0[-1, :] = 0.0
u0[:, 0] = 0.0
u0[:, -1] = 0.0

v0[0, :] = 0.0
v0[-1, :] = 0.0
v0[:, 0] = 0.0
v0[:, -1] = 0.0


dt_limit = 1.0 / (c * np.sqrt((1.0 / dx ** 2) + (1.0 / dy ** 2)))
dt = safety_factor * dt_limit
Nt = int(np.ceil(T / dt))
dt = T / float(Nt)
t = np.linspace(0.0, T, Nt + 1)


all_frames = []
all_times = []


u_prev = u0.copy()
all_frames.append(u_prev.copy())
all_times.append(t[0])


# first step uses the initial velocity separately
u_curr = np.zeros_like(u_prev)

laplacian = (
    (u_prev[2:, 1:-1] - 2.0 * u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1]) / dx ** 2
    + (u_prev[1:-1, 2:] - 2.0 * u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2]) / dy ** 2
)

u_curr[1:-1, 1:-1] = (
    u_prev[1:-1, 1:-1]
    + dt * v0[1:-1, 1:-1]
    + 0.5 * c ** 2 * dt ** 2 * laplacian
)

all_frames.append(u_curr.copy())
all_times.append(t[1])


for n in range(1, Nt):
    u_next = np.zeros_like(u_curr)

    laplacian = (
        (u_curr[2:, 1:-1] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) / dx ** 2
        + (u_curr[1:-1, 2:] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) / dy ** 2
    )

    u_next[1:-1, 1:-1] = (
        2.0 * u_curr[1:-1, 1:-1]
        - u_prev[1:-1, 1:-1]
        + c ** 2 * dt ** 2 * laplacian
    )

    all_frames.append(u_next.copy())
    all_times.append(t[n + 1])

    u_prev = u_curr
    u_curr = u_next


all_frames = np.array(all_frames)
all_times = np.array(all_times)
u_final = u_curr.copy()
u_ref = np.max(np.abs(u0))

if u_ref == 0.0:
    u_ref = 1.0

all_frames_nd = all_frames / u_ref


print("Prepared a 2D wave equation solver using finite differences.")
print("Domain: Lx={0}, Ly={1}".format(Lx, Ly))
print("Grid: Nx={0}, Ny={1}, dx={2:.6f}, dy={3:.6f}".format(Nx, Ny, dx, dy))
print("Time: T={0}, Nt={1}, dt={2:.6f}, stability_limit={3:.6f}".format(T, Nt, dt, dt_limit))
print("Stored animation frame array shape:", all_frames.shape)
print("Plotting nondimensional field u / u_ref with u_ref = {0:.6f}".format(u_ref))


fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(
    all_frames_nd[0].T,
    extent=[0.0, Lx, 0.0, Ly],
    origin="lower",
    cmap="RdBu_r",
    aspect="equal",
    vmin=-1.0,
    vmax=1.0,
)

ax.set_xlabel("x")
ax.set_ylabel("y")
title = ax.set_title("t = %.3f" % all_times[0])

cbar = fig.colorbar(im, ax=ax, label="u / u_ref")
del cbar


def update(frame):
    im.set_data(all_frames_nd[frame].T)
    title.set_text("t = %.3f" % all_times[frame])
    return im, title


interval_ms = max(20, int(1000 * dt))
ani = FuncAnimation(fig, update, frames=len(all_frames_nd), interval=interval_ms, blit=True)

plt.tight_layout()
plt.show()
