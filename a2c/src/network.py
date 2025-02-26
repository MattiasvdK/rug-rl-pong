import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class SimulatedAnnealing(nn.Module):
    def __init__(self, t_start: float, cooling_steps: int):
        super(SimulatedAnnealing, self).__init__()

        self.schedule = np.linspace(t_start, 1, cooling_steps)
        self.step_count = 0
        self.cooling_steps = cooling_steps
        self.softmax = nn.Softmax(dim=1)
        self.function = self._cooled
        self.done = True

    def forward(self, x):
        return self.function(x)

    def step(self):
        if self.done:
            return
        self.step_count += 1

        if self.step_count == self.cooling_steps:
            self.function = self._cooled
            self.done = True
            print(f'--SIMULATED ANNEALING FINISHED--')

    def _annealing(self, x):
        x = x / self.schedule[self.step_count]
        return self.softmax(x)

    def _cooled(self, x):
        x = x / 1
        return self.softmax(x)


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        # Add any other arguments you need here
        # e.g. number of hidden layers, number of neurons in each layer, etc.
        in_size: Tuple[int, int, int] = (4, 21, 21),
        out_features: int = 3,
        cnn_layers: List[int] = [7],
        pooling: bool = True,
        fcl_layers: List[int] = [512],
        device: torch.device = torch.device('cuda'),
    ) -> None:
        """!
        Initialize a neural network. This network can be used to approximate
        some functions, maybe the reward function? Keep in mind that this class
        should only be used to define the neural network, not to train it using
        reinforcement learning algorithms.

        @param in_size (Tuple[int, int, int]): Input dimension of the network
        @param out_features (int): Output dimension of the network
        @cnn_layers (List[int]): Number of convolutional kernel_sizes
        @pooling (bool): Whether to use pooling
        @fcl_layers (List[int]): Number of neurons in the fcl layers
        @device (torch.device): Device used to run the network
        """
        super(NeuralNetwork, self).__init__()

        self.aggregation = None

        channels = in_size[0]
        dim = in_size[1]
        out_channels = 16

        cnn_modules = []
        for kernel_size in cnn_layers:
            cnn_modules.append(nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=channels,
                out_channels=out_channels
            ))
            cnn_modules.append(nn.ReLU())

            channels = out_channels
            out_channels = out_channels * 2
            dim = dim - (kernel_size - 1)

            if pooling:
                cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                dim = dim // 2

        n_features = channels * dim**2
        cnn_modules.append(nn.Flatten())
        self.cnn_features = n_features

        self.cnn = nn.Sequential(*cnn_modules)

        fcl_modules = []

        for n_neurons in fcl_layers:
            fcl_modules.append(nn.Linear(n_features, n_neurons))
            fcl_modules.append(nn.ReLU())
            n_features = n_neurons

        # Make sure the code still runs if the default device is unchanged
        if device == torch.device('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.fcl = nn.Sequential(*fcl_modules)

        self.actions = nn.Sequential(
            nn.Linear(n_features, out_features),
            nn.Softmax(dim=1),
        )

        self.critic = nn.Sequential(
            nn.Linear(n_features, 1),
        )

        self.cnn = self.cnn.to(self.device)
        self.fcl = self.fcl.to(self.device)
        self.actions = self.actions.to(self.device)
        self.critic = self.critic.to(self.device)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """!
        Convert the input state to the relevant output using the neural network.

        @param state (torch.Tensor): Input state for the network

        @return Tuple[torch.Tensor, torch.Tensor]: Output of the network
        """
        state = state.to(self.device)

        # Account for the batch dimension in torch modules
        batch = state.ndim == 4

        # If there is no batch, we add the batch dimension
        if not batch:
            state = state.unsqueeze(0)

        actions = self.cnn(state)
        actions = self.fcl(actions)
        action_probs = self.actions(actions)

        value = self.critic(actions)

        action_probs = action_probs if batch else action_probs.squeeze()
        value = value if batch else value.squeeze()

        return action_probs, value
