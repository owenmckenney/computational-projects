import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# this class loads in the h5 dataset
class WaveH5Dataset(Dataset):
    def __init__(self, path, points_per_run):
        self.path = path
        self.points_per_run = points_per_run

        with h5py.File(self.path, "r") as h5_file:
            self.num_runs = h5_file["u"].shape[0]
            self.t_values = h5_file["t"][:]
            self.x_values = h5_file["x"][:]
            self.y_values = h5_file["y"][:]

        self.h5_file = None

    def _ensure_open(self):
        # Open the HDF5 handle lazily so dataset construction stays lightweight.
        if self.h5_file is None:
            self.h5_file = h5py.File(self.path, "r")

    def __len__(self):
        return self.num_runs

    # return one FD run with a random subset of supervised points
    def __getitem__(self, idx):
        self._ensure_open()

        u_run = self.h5_file["u"][idx] # (Nt, Nx, Ny)
        ic_params = self.h5_file["ic_params"][idx] # (x0, y0, sigma, amplitude)

        Nt, Nx, Ny = u_run.shape
        total_points = Nt * Nx * Ny

        # randomly sample points from one finite difference run 
        flat_idxs = np.random.choice(total_points, size=self.points_per_run, replace=False)

        # convert flattened indices into (t, x, y)
        t_idx, rem = np.divmod(flat_idxs, Nx * Ny)
        x_idx, y_idx = np.divmod(rem, Ny)

        t_sample = self.t_values[t_idx]
        x_sample = self.x_values[x_idx]
        y_sample = self.y_values[y_idx]
        u_sample = u_run[t_idx, x_idx, y_idx]

        coordinates = np.stack([x_sample, y_sample, t_sample], axis=1).astype(np.float32)
        targets = u_sample.reshape(-1, 1).astype(np.float32)

        ic_condition = ic_params[:3].astype(np.float32)

        return {"coords": torch.tensor(coordinates, dtype=torch.float32), "u": torch.tensor(targets, dtype=torch.float32), "ic_params": torch.tensor(ic_condition, dtype=torch.float32)}
