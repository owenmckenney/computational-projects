import torch
import torch.nn as nn

# tanh is good because the 2nd derivates can be 0 which will mean ReLU will fail
# issue is not network size/depth but rather lack of supervised data for PINN to work with 

# phyics-informed neural network for the 2D wave equation
# model takes in (x, y, t) and predicts wave displacement u
class PINN(nn.Module):
    def __init__(self, input_dim=6, output_dim=1, hidden_dim=64, num_hidden_layers=4, lower_bounds=(-1.0, -1.0, -1.0), upper_bounds=(1.0, 1.0, 1.0)):
        
        super().__init__()

        # store lower and upper bbounds of input domain
        self.register_buffer("lower_bounds", torch.tensor(lower_bounds, dtype=torch.float32))
        self.register_buffer("upper_bounds", torch.tensor(upper_bounds, dtype=torch.float32))

        layers = []

        # first layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # hidden layers
        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # final layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # combine all layers into one nn    
        self.network = nn.Sequential(*layers)

    # scales inputs from the physical domain to [-1, 1]
    def normalize_inputs(self, xyt):
        return 2 * (xyt - self.lower_bounds) / (self.upper_bounds - self.lower_bounds) - 1.0
    
    
    # runs the network on input points. outputs u
    def forward(self, xyt):
        xyt_normalized = self.normalize_inputs(xyt)
        return self.network(xyt_normalized)
        
