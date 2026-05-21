import os
import torch
from torch.utils.data import DataLoader

from pinn_model import PINN
from utils import WaveH5Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = r"C:\Users\Owen-McKenney\OneDrive\Desktop\computational-projects\pinn-wave-equation-solver\training-datasets\td_100N_500s.h5"

# physical parameters
Lx = 1.0
Ly = 1.0
c = 1.0
T = 0.5

# initial condition
amplitude = -1.0

# training parameters
num_epochs = 100
learning_rate = 1e-3
batch_size_runs = 8

# number of sampled points in each training run
num_supervised_points = 128
num_interior_points = 128
num_initial_points = 32
num_boundary_points = 32

# loss weights
w_data = 20.0
w_pde = 10.0
w_initial_displacement = 10.0
w_initial_velocity = 1.0
w_boundary = 10.0

# IC parameter ranges
x0_min = 0.1
x0_max = 0.9
y0_min = 0.1
y0_max = 0.9
sigma_min = 0.1
sigma_max = 0.1


# initial wave shape at t = 0 for a batch of Gaussian parameters
def initial_displacement(x, y, ic_params):
    x0 = ic_params[:, 0:1]
    y0 = ic_params[:, 1:2]
    sigma = ic_params[:, 2:3]
    return amplitude * torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (sigma ** 2))


# initial velocity at t = 0. wave starts from rest
def initial_velocity(x):
    return torch.zeros_like(x)


# sample points inside the domain for the PDE loss
def sample_interior_points(ic_params, num_points_per_run):
    batch_size = ic_params.shape[0]

    x = Lx * torch.rand(batch_size, num_points_per_run, 1, device=device)
    y = Ly * torch.rand(batch_size, num_points_per_run, 1, device=device)
    t = T * torch.rand(batch_size, num_points_per_run, 1, device=device)
    cond = ic_params[:, None, :].expand(-1, num_points_per_run, -1)

    return torch.cat([x, y, t, cond], dim=2).reshape(-1, 6)


# sample points at t = 0 for the initial condition loss
def sample_initial_points(ic_params, num_points_per_run):
    batch_size = ic_params.shape[0]

    x = Lx * torch.rand(batch_size, num_points_per_run, 1, device=device)
    y = Ly * torch.rand(batch_size, num_points_per_run, 1, device=device)
    t = torch.zeros(batch_size, num_points_per_run, 1, device=device)
    cond = ic_params[:, None, :].expand(-1, num_points_per_run, -1)

    return torch.cat([x, y, t, cond], dim=2).reshape(-1, 6)


# sample points along the four fixed boundaries
def sample_boundary_points(ic_params, num_points_per_run):
    batch_size = ic_params.shape[0]
    points_per_side = num_points_per_run // 4

    cond = ic_params[:, None, :].expand(-1, points_per_side, -1)

    x_left = torch.zeros(batch_size, points_per_side, 1, device=device)
    y_left = Ly * torch.rand(batch_size, points_per_side, 1, device=device)
    t_left = T * torch.rand(batch_size, points_per_side, 1, device=device)
    left = torch.cat([x_left, y_left, t_left, cond], dim=2)

    x_right = Lx * torch.ones(batch_size, points_per_side, 1, device=device)
    y_right = Ly * torch.rand(batch_size, points_per_side, 1, device=device)
    t_right = T * torch.rand(batch_size, points_per_side, 1, device=device)
    right = torch.cat([x_right, y_right, t_right, cond], dim=2)

    x_bottom = Lx * torch.rand(batch_size, points_per_side, 1, device=device)
    y_bottom = torch.zeros(batch_size, points_per_side, 1, device=device)
    t_bottom = T * torch.rand(batch_size, points_per_side, 1, device=device)
    bottom = torch.cat([x_bottom, y_bottom, t_bottom, cond], dim=2)

    x_top = Lx * torch.rand(batch_size, points_per_side, 1, device=device)
    y_top = Ly * torch.ones(batch_size, points_per_side, 1, device=device)
    t_top = T * torch.rand(batch_size, points_per_side, 1, device=device)
    top = torch.cat([x_top, y_top, t_top, cond], dim=2)

    return torch.cat([left, right, bottom, top], dim=1).reshape(-1, 6)


# compute the wave equation residual
def wave_equation_residual(model, inputs):
    inputs = inputs.clone().detach().requires_grad_(True)
    u = model(inputs)

    # first derivatives with respect to x, y, t
    grad_u = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]

    # second derivatives with respect to x, y, t
    u_xx = torch.autograd.grad(u_x, inputs, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, inputs, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    u_tt = torch.autograd.grad(u_t, inputs, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 2:3]

    return u_tt - c ** 2 * (u_xx + u_yy)


# compute the total hybrid loss for one batch
def compute_loss(model, batch):
    coords = batch["coords"].to(device)
    u_true = batch["u"].to(device)
    ic_params = batch["ic_params"].to(device)

    batch_size, num_supervised, _ = coords.shape

    # add gaussian parameters to each supervised point
    cond_supervised = ic_params[:, None, :].expand(-1, num_supervised, -1)
    supervised_inputs = torch.cat([coords, cond_supervised], dim=2).reshape(-1, 6)
    u_true = u_true.reshape(-1, 1)

    u_pred = model(supervised_inputs)
    loss_data = torch.mean((u_pred - u_true) ** 2)

    # PDE loss
    interior_points = sample_interior_points(ic_params, num_interior_points)
    residual = wave_equation_residual(model, interior_points)
    loss_pde = torch.mean(residual ** 2)

    # initial displacement loss
    initial_points = sample_initial_points(ic_params, num_initial_points)
    initial_points = initial_points.clone().detach().requires_grad_(True)

    u_initial_pred = model(initial_points)
    x_initial = initial_points[:, 0:1]
    y_initial = initial_points[:, 1:2]
    ic_initial = initial_points[:, 3:6]

    u_initial_true = initial_displacement(x_initial, y_initial, ic_initial)
    loss_initial_displacement = torch.mean((u_initial_pred - u_initial_true) ** 2)

    # initial velocity loss
    grad_u_initial = torch.autograd.grad(u_initial_pred, initial_points, grad_outputs=torch.ones_like(u_initial_pred), create_graph=True)[0]

    u_t_initial_pred = grad_u_initial[:, 2:3]
    u_t_initial_true = initial_velocity(x_initial)
    loss_initial_velocity = torch.mean((u_t_initial_pred - u_t_initial_true) ** 2)

    # boundary condition loss
    boundary_points = sample_boundary_points(ic_params, num_boundary_points)
    u_boundary_pred = model(boundary_points)
    loss_boundary = torch.mean(u_boundary_pred ** 2)

    total_loss = (
        (w_data * loss_data)
        + (w_pde * loss_pde)
        + (w_initial_displacement * loss_initial_displacement)
        + (w_initial_velocity * loss_initial_velocity)
        + (w_boundary * loss_boundary)
    )

    return (total_loss, loss_data, loss_pde, loss_initial_displacement, loss_initial_velocity, loss_boundary)


# train the conditional PINN
def main():
    dataset = WaveH5Dataset(data_path, num_supervised_points)
    train_loader = DataLoader(dataset, batch_size=batch_size_runs, shuffle=True, num_workers=0)

    # construct model
    model = PINN(
        input_dim=6,
        output_dim=1,
        hidden_dim=64,
        num_hidden_layers=4,
        lower_bounds=(0.0, 0.0, 0.0, x0_min, y0_min, sigma_min),
        upper_bounds=(Lx, Ly, T, x0_max, y0_max, sigma_max),
    ).to(device)

    # using adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs + 1):
        for batch in train_loader:
            loss, loss_data, loss_pde, loss_ic_u, loss_ic_v, loss_bc = compute_loss(model, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} batch done | loss = {loss.item():.6e}")

        # print progress every 100 epochs
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Total: {loss.item():.6e} | "
                f"Data: {loss_data.item():.6e} | "
                f"PDE: {loss_pde.item():.6e} | "
                f"IC u: {loss_ic_u.item():.6e} | "
                f"IC u_t: {loss_ic_v.item():.6e} | "
                f"BC: {loss_bc.item():.6e}"
            )

    # save model weights in a checkpoint folder
    os.makedirs("checkpoints", exist_ok=True)

    save_path = os.path.join("checkpoints", "pinn_wave_model_colab1.pt")
    torch.save(model.state_dict(), save_path)

    print("Training complete.")
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
