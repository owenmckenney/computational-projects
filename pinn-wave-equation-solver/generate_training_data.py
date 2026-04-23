import h5py
import numpy as np
from finite_differences import WaveEquationFD

# dataset parameters
samples = 500
output_path = r"C:\Users\Owen-McKenney\OneDrive\Desktop\computational-projects\pinn-wave-equation-solver\training-datasets\td_100N_500s.h5"
random_seed = 1234

# latin hypercube sampling ranges
x0_min = 0.1
x0_max = 0.9
y0_min = 0.1
y0_max = 0.9
sigma_min = 0.05
sigma_max = 0.2

# physical parameters
Lx = 1.0
Ly = 1.0
c = 1.0
T = 0.5

# grid size
Nx = 101
Ny = 101

# initial condition
amplitude = -1.0

safety_factor = 0.95


def latin_hypercube_samples(num_samples, num_dimensions, seed):
    rng = np.random.default_rng(seed)
    lhs = np.zeros((num_samples, num_dimensions), dtype=np.float32)

    for dim in range(num_dimensions):
        permutation = rng.permutation(num_samples)
        lhs[:, dim] = (permutation + rng.random(num_samples)) / num_samples

    return lhs

def scale_lhs_column(lhs_column, min_value, max_value):
    return min_value + (max_value - min_value) * lhs_column

def main():
    lhs = latin_hypercube_samples(samples, 3, random_seed)

    x0_values = scale_lhs_column(lhs[:, 0], x0_min, x0_max)
    y0_values = scale_lhs_column(lhs[:, 1], y0_min, y0_max)
    sigma_values = scale_lhs_column(lhs[:, 2], sigma_min, sigma_max)

    with h5py.File(output_path, "w") as h5_file:
        u_dataset = None
        ic_params_dataset = h5_file.create_dataset("ic_params", (samples, 4), dtype=np.float32)

        for i in range(samples):
            solver = WaveEquationFD(
                Lx=Lx,
                Ly=Ly,
                c=c,
                T=T,
                Nx=Nx,
                Ny=Ny,
                x0=float(x0_values[i]),
                y0=float(y0_values[i]),
                sigma=float(sigma_values[i]),
                amplitude=amplitude,
                safety_factor=safety_factor,
            )

            u = solver.solve()

            if u_dataset is None:
                h5_file.create_dataset("x", data=solver.x, dtype=np.float32)
                h5_file.create_dataset("y", data=solver.y, dtype=np.float32)
                h5_file.create_dataset("t", data=solver.t, dtype=np.float32)
                u_dataset = h5_file.create_dataset(
                    "u",
                    (samples, solver.Nt + 1, solver.Nx, solver.Ny),
                    dtype=np.float32,
                    compression="gzip",
                )

            u_dataset[i] = u
            ic_params_dataset[i] = np.array(
                [solver.x0, solver.y0, solver.sigma, solver.amplitude],
                dtype=np.float32,
            )

            print(
                "Saved sample {0}/{1} | x0={2:.4f}, y0={3:.4f}, sigma={4:.4f}".format(
                    i + 1,
                    samples,
                    solver.x0,
                    solver.y0,
                    solver.sigma,
                )
            )

        h5_file.attrs["samples"] = samples
        h5_file.attrs["random_seed"] = random_seed
        h5_file.attrs["Lx"] = Lx
        h5_file.attrs["Ly"] = Ly
        h5_file.attrs["c"] = c
        h5_file.attrs["T"] = T
        h5_file.attrs["Nx"] = Nx
        h5_file.attrs["Ny"] = Ny
        h5_file.attrs["amplitude"] = amplitude
        h5_file.attrs["safety_factor"] = safety_factor
        h5_file.attrs["x0_min"] = x0_min
        h5_file.attrs["x0_max"] = x0_max
        h5_file.attrs["y0_min"] = y0_min
        h5_file.attrs["y0_max"] = y0_max
        h5_file.attrs["sigma_min"] = sigma_min
        h5_file.attrs["sigma_max"] = sigma_max

    print("Saved dataset to {0}".format(output_path))


if __name__ == "__main__":
    main()
