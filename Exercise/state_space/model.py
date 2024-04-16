import torch
import torch.nn as nn


# taken from pytorch-ident, in https://github.com/forgi86/pytorch-ident/blob/master/torchid/ss/dt/models.py

class StateSpaceSimulator(nn.Module):
    r""" Discrete-time state-space simulator.

    Args:
        f_xu (nn.Module): The neural state-space model.
        batch_first (bool): If True, first dimension is batch.

    Inputs: x_0, u
        * **x_0**: tensor of shape :math:`(N, n_{x})` containing the
          initial hidden state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **input**: tensor of shape :math:`(L, N, n_{u})` when ``batch_first=False`` or
          :math:`(N, L, n_{x})` when ``batch_first=True`` containing the input sequence

    Outputs: x
        * **x**: tensor of shape :math:`(L, N, n_{x})` corresponding to
          the simulated state sequence.

    Examples::

        >>> ss_model = NeuralStateSpaceModel(n_x=3, n_u=2)
        >>> nn_solution = StateSpaceSimulator(ss_model)
        >>> x0 = torch.randn(64, 3)
        >>> u = torch.randn(100, 64, 2)
        >>> x = nn_solution(x0, u)
        >>> print(x.size())
        torch.Size([100, 64, 3])
     """

    def __init__(self, f_xu, batch_first=True):
        super().__init__()
        self.f_xu = f_xu
        self.batch_first = batch_first


    def forward(self, x_0, u):
        x_step = x_0
        dim_time = 1 if self.batch_first else 0
        x = []
        for u_step in u.split(1, dim=dim_time):  # split along the time axis
            u_step = u_step.squeeze(dim_time)
            x += [x_step]
            dx = self.f_xu(x_step, u_step)
            x_step = x_step + dx

        x = torch.stack(x, dim_time)
        return x
    


class NeuralStateUpdate(nn.Module):

    def __init__(self, n_x, n_u, n_feat=32):
        super(NeuralStateUpdate, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_x+n_u, n_feat),
            nn.Tanh(),
            nn.Linear(n_feat, n_x),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-2)
                nn.init.constant_(m.bias, val=0)
                    
    def forward(self, x, u):
        z = torch.cat((x, u), dim=-1)
        dx = self.net(z)
        return dx
    

class NeuralOutput(nn.Module):

    def __init__(self, n_x, n_y, n_feat=32):
        super(NeuralOutput, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_x, n_feat),
            nn.Tanh(),
            nn.Linear(n_feat, n_y),
        )
                    
    def forward(self, x):
        y = self.net(x)
        return y
        

class CascadedTanksNeuralStateSpaceModel(nn.Module):
    r"""A state-space model to represent the cascaded two-tank system.


    Args:
        n_feat: (int, optional): Number of input features in the hidden layer. Default: 0
        scale_dx: (str): Scaling factor for the neural network output. Default: 1.0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True

    """

    def __init__(self, n_feat=64, init_small=True):
        super(CascadedTanksNeuralStateSpaceModel, self).__init__()
        self.n_feat = n_feat

        # Neural network for the first state equation = NN(x_1, u)
        self.net1 = nn.Sequential(
            nn.Linear(2, n_feat),
            nn.Tanh(),
            nn.Linear(n_feat, 1),
        )

        # Neural network for the first state equation = NN(x_1, x2)
        self.net2 = nn.Sequential(
            nn.Linear(2, n_feat),
            nn.Tanh(),
            nn.Linear(n_feat, 1),
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net1.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net2.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):

        # the first state derivative is NN(x1, u)
        in_1 = torch.cat((in_x[..., [0]], in_u), -1)  # concatenate 1st state component with input
        dx_1 = self.net1(in_1)

        # the second state derivative is NN(x1, x2)
        in_2 = in_x
        dx_2 = self.net2(in_2)

        # the state derivative is built by concatenation of dx_1 and dx_2, possibly scaled for numerical convenience
        dx = torch.cat((dx_1, dx_2), -1)
        return dx


class CascadedTanksOverflowNeuralStateSpaceModel(nn.Module):

    def __init__(self, n_feat=64, init_small=True):
        super(CascadedTanksOverflowNeuralStateSpaceModel, self).__init__()
        self.n_feat = n_feat

        # Neural network for the first state equation = NN(x_1, u)
        self.net1 = nn.Sequential(
            nn.Linear(2, n_feat),
            nn.ReLU(),
            #nn.Linear(n_feat, n_feat),
            #nn.ReLU(),
            nn.Linear(n_feat, 1),
        )

        # Neural network for the first state equation = NN(x_1, x2, u) # we assume that with overflow the input may influence the 2nd tank instantaneously
        self.net2 = nn.Sequential(
            nn.Linear(3, n_feat),
            nn.ReLU(),
            #nn.Linear(n_feat, n_feat),
            #nn.ReLU(),
            nn.Linear(n_feat, 1),
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net1.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-2)
                    nn.init.constant_(m.bias, val=0)

            for m in self.net2.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-2)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):

        # the first state derivative is net1(x1, u)
        in_1 = torch.cat((in_x[..., [0]], in_u), -1)  # concatenate 1st state component with input
        dx_1 = self.net1(in_1)

        # the second state derivative is net2(x1, x2, u)
        in_2 = torch.cat((in_x, in_u), -1) # concatenate states with input to define the
        dx_2 = self.net2(in_2)

        # the state derivative is built by concatenation of dx_1 and dx_2, possibly scaled for numerical convenience
        dx = torch.cat((dx_1, dx_2), -1)
        dx = dx
        return dx