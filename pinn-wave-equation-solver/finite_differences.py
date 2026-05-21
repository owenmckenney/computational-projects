import numpy as np
import matplotlib.pyplot as plt

# class to solve the wave equation using finite differences
class WaveEquationFD():
    def __init__(self, Lx=1.0, Ly=1.0, c=1.0, T=0.5, Nx=101, Ny=101, x0=None, y0=None, sigma=None, amplitude=-1.0, safety_factor=0.95, num_snapshots=5):

        # physical parameters
        self.Lx = Lx
        self.Ly = Ly
        self.c = c
        self.T = T

        # grid size
        self.Nx = Nx
        self.Ny = Ny

        # initial condition
        self.x0 = 0.5 * Lx if x0 is None else x0
        self.y0 = 0.5 * Ly if y0 is None else y0
        self.sigma = 0.1 * min(Lx, Ly) if sigma is None else sigma
        self.amplitude = amplitude

        # use a little less than the stability limit
        self.safety_factor = safety_factor

        # save a few solution snapshots
        self.num_snapshots = num_snapshots

        self.dx = Lx / float(Nx - 1)
        self.dy = Ly / float(Ny - 1)

        self.x = np.linspace(0.0, Lx, Nx, dtype=np.float32)
        self.y = np.linspace(0.0, Ly, Ny, dtype=np.float32)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self.u0 = amplitude * np.exp(-((self.X - self.x0) ** 2 + (self.Y - self.y0) ** 2) / (self.sigma ** 2))
        self.v0 = np.zeros_like(self.X)

        # fixed boundaries
        self.u0[0, :] = 0.0
        self.u0[-1, :] = 0.0
        self.u0[:, 0] = 0.0
        self.u0[:, -1] = 0.0

        self.v0[0, :] = 0.0
        self.v0[-1, :] = 0.0
        self.v0[:, 0] = 0.0
        self.v0[:, -1] = 0.0

        self.dt_limit = 1.0 / (c * np.sqrt((1.0 / self.dx ** 2) + (1.0 / self.dy ** 2)))
        self.dt = safety_factor * self.dt_limit
        self.Nt = int(np.ceil(T / self.dt))
        self.dt = T / float(self.Nt)
        self.t = np.linspace(0.0, T, self.Nt + 1, dtype=np.float32)

        self.snapshot_indices = np.linspace(0, self.Nt, self.num_snapshots, dtype=int)
        self.snapshot_indices = np.unique(self.snapshot_indices)
        self.snapshot_set = set(self.snapshot_indices.tolist())

    def solve(self):
        u_history = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float32)

        u_prev = self.u0.copy()
        u_history[0] = u_prev

        # first step uses the initial velocity separately
        u_curr = np.zeros_like(u_prev)

        laplacian = ((u_prev[2:, 1:-1] - 2.0 * u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1]) / self.dx ** 2
            + (u_prev[1:-1, 2:] - 2.0 * u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2]) / self.dy ** 2)

        u_curr[1:-1, 1:-1] = (u_prev[1:-1, 1:-1] + self.dt * self.v0[1:-1, 1:-1] + 0.5 * self.c ** 2 * self.dt ** 2 * laplacian)

        u_history[1] = u_curr

        for n in range(1, self.Nt):
            u_next = np.zeros_like(u_curr)

            laplacian = ((u_curr[2:, 1:-1] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) / self.dx ** 2
                + (u_curr[1:-1, 2:] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) / self.dy ** 2)

            u_next[1:-1, 1:-1] = (2.0 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + self.c ** 2 * self.dt ** 2 * laplacian)

            u_history[n + 1] = u_next

            u_prev = u_curr
            u_curr = u_next

        self.u = u_history
        return self.u

    def get_snapshots(self):
        if not hasattr(self, "u"):
            self.solve()

        snapshots = self.u[self.snapshot_indices]
        saved_times = self.t[self.snapshot_indices]
        return snapshots, saved_times

    def plot_snapshots(self):
        snapshots, saved_times = self.get_snapshots()

        u_ref = np.max(np.abs(self.u0))

        if u_ref == 0.0:
            u_ref = 1.0

        snapshots_nd = snapshots / u_ref

        fig, axes = plt.subplots(1, len(snapshots), figsize=(3 * len(snapshots) + 1, 3))

        if len(snapshots) == 1:
            axes = [axes]
        else:
            axes = list(axes)

        for ax, snapshot_nd, time_value in zip(axes, snapshots_nd, saved_times):
            im = ax.imshow(
                snapshot_nd.T,
                extent=[0.0, self.Lx, 0.0, self.Ly],
                origin="lower",
                cmap="RdBu_r",
                aspect="equal",
                vmin=-1.0,
                vmax=1.0,
            )
            ax.set_title("t = %.3f" % time_value)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
        fig.colorbar(im, cax=cbar_ax, label="u / u_ref")
        plt.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
        plt.show()


def main():
    solver = WaveEquationFD()
    u = solver.solve()

    print("Prepared a 2D wave equation solver using finite differences.")
    print("Domain: Lx={0}, Ly={1}".format(solver.Lx, solver.Ly))
    print("Grid: Nx={0}, Ny={1}, dx={2:.6f}, dy={3:.6f}".format(solver.Nx, solver.Ny, solver.dx, solver.dy))
    print("Time: T={0}, Nt={1}, dt={2:.6f}, stability_limit={3:.6f}".format(solver.T, solver.Nt, solver.dt, solver.dt_limit))
    print("Solution array shape:", u.shape)

    solver.plot_snapshots()


if __name__ == "__main__":
    main()
