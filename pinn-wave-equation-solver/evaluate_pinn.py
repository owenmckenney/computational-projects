import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_model import PINN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# These values should match the values used in train_pinn.py and finite_differences.py.
Lx = 1.0
Ly = 1.0
c = 1.0
T = 0.5


# These values should match the initial condition used in training.
x0 = 0.5 * Lx
y0 = 0.5 * Ly
sigma = 0.2 * min(Lx, Ly)
amplitude = -1.0

# These values should match the IC ranges used in train_pinn.py.
x0_min = 0.1
x0_max = 0.9
y0_min = 0.1
y0_max = 0.9
sigma_min = 0.05
sigma_max = 0.2


# Grid and snapshot settings should match finite_differences.py.
Nx = 101
Ny = 101
num_snapshots = 5
safety_factor = 0.95


def load_model(checkpoint_path):
    model = PINN(
        input_dim=6,
        output_dim=1,
        hidden_dim=64,
        num_hidden_layers=4,
        lower_bounds=(0.0, 0.0, 0.0, x0_min, y0_min, sigma_min),
        upper_bounds=(Lx, Ly, T, x0_max, y0_max, sigma_max),
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def evaluate_on_grid(model, X, Y, time_value):
    # Build input points with columns [x, y, t, x0, y0, sigma].
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = time_value * np.ones_like(x_flat)
    x0_flat = x0 * np.ones_like(x_flat)
    y0_flat = y0 * np.ones_like(x_flat)
    sigma_flat = sigma * np.ones_like(x_flat)
    xyt = np.hstack([x_flat, y_flat, t_flat, x0_flat, y0_flat, sigma_flat])

    xyt_tensor = torch.tensor(xyt, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(xyt_tensor)

    return u_pred.cpu().numpy().reshape(Nx, Ny)


def compute_reference_grid():
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    return x, y, X, Y


def run_finite_differences():
    x, y, X, Y = compute_reference_grid()

    dx = Lx / float(Nx - 1)
    dy = Ly / float(Ny - 1)

    u0 = amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (sigma ** 2))
    v0 = np.zeros_like(X)

    # Fixed boundaries.
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

    snapshot_indices = np.linspace(0, Nt, num_snapshots, dtype=int)
    snapshot_indices = np.unique(snapshot_indices)
    snapshot_set = set(snapshot_indices.tolist())

    snapshots = []
    saved_times = []

    u_prev = u0.copy()
    snapshots.append(u_prev.copy())
    saved_times.append(t[0])

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

    if 1 in snapshot_set:
        snapshots.append(u_curr.copy())
        saved_times.append(t[1])

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

        if (n + 1) in snapshot_set:
            snapshots.append(u_next.copy())
            saved_times.append(t[n + 1])

        u_prev = u_curr
        u_curr = u_next

    snapshots = np.array(snapshots)
    saved_times = np.array(saved_times)

    u_ref = np.max(np.abs(u0))
    if u_ref == 0.0:
        u_ref = 1.0

    return X, Y, snapshots, saved_times, u_ref


def main():
    checkpoint_path = os.path.join("checkpoints", "pinn_wave_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Could not find checkpoints/pinn_wave_model.pt. "
            "Run train_pinn.py first to create the trained model."
        )

    X, Y, fd_snapshots, saved_times, u_ref = run_finite_differences()

    model = load_model(checkpoint_path)

    pinn_snapshots = []
    for time_value in saved_times:
        pinn_snapshots.append(evaluate_on_grid(model, X, Y, time_value))

    fd_snapshots_nd = fd_snapshots / u_ref
    pinn_snapshots_nd = np.array(pinn_snapshots) / u_ref
    error_snapshots_nd = fd_snapshots_nd - pinn_snapshots_nd

    print("Prepared finite-difference and PINN wave equation snapshots.")
    print("Domain: Lx={0}, Ly={1}".format(Lx, Ly))
    print("Grid: Nx={0}, Ny={1}".format(Nx, Ny))
    print("Time: T={0}".format(T))
    print("Finite-difference snapshot array shape:", fd_snapshots.shape)
    print("PINN snapshot array shape:", pinn_snapshots_nd.shape)
    print("Error snapshot array shape:", error_snapshots_nd.shape)
    print("Plotting nondimensional field u / u_ref with u_ref = {0:.6f}".format(u_ref))

    fig, axes = plt.subplots(3, len(saved_times), figsize=(3 * len(saved_times) + 1, 9))

    if len(saved_times) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]], dtype=object)

    for col, time_value in enumerate(saved_times):
        ax_fd = axes[0, col]
        ax_pinn = axes[1, col]
        ax_error = axes[2, col]

        im_solution = ax_fd.imshow(
            fd_snapshots_nd[col].T,
            extent=[0.0, Lx, 0.0, Ly],
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-1.0,
            vmax=1.0,
        )
        ax_fd.set_title("t = %.3f" % time_value)
        ax_fd.set_xlabel("x")
        ax_fd.set_ylabel("y")

        ax_pinn.imshow(
            pinn_snapshots_nd[col].T,
            extent=[0.0, Lx, 0.0, Ly],
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-1.0,
            vmax=1.0,
        )
        ax_pinn.set_xlabel("x")
        ax_pinn.set_ylabel("y")

        im_error = ax_error.imshow(
            error_snapshots_nd[col].T,
            extent=[0.0, Lx, 0.0, Ly],
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-0.1,
            vmax=0.1,
        )
        ax_error.set_xlabel("x")
        ax_error.set_ylabel("y")

    axes[0, 0].text(
        -0.28,
        0.5,
        "Finite Differences",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
    )
    axes[1, 0].text(
        -0.28,
        0.5,
        "PINN",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
    )
    axes[2, 0].text(
        -0.28,
        0.5,
        "FD - PINN",
        transform=axes[2, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
    )

    solution_cbar_ax = fig.add_axes([0.90, 0.39, 0.02, 0.43])
    fig.colorbar(im_solution, cax=solution_cbar_ax, label="u / u_ref")

    error_cbar_ax = fig.add_axes([0.90, 0.11, 0.02, 0.17])
    fig.colorbar(im_error, cax=error_cbar_ax, label="FD - PINN")

    plt.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
    plt.show()


if __name__ == "__main__":
    main()
