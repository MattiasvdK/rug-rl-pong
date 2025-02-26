import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        # Add any other arguments you need here
        # e.g. number of hidden layers, number of neurons in each layer, etc.
        in_size: Tuple[int, int, int] = (4, 84, 84),
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

        fcl_modules.append(nn.Linear(n_features, out_features))

        # Make sure the code still runs if the default device is unchanged
        if device == torch.device('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.streams = nn.ModuleList([nn.Sequential(*fcl_modules)])

        self.cnn = self.cnn.to(self.device)
        self.streams = self.streams.to(self.device)

    def forward(
        self,
        state: np.ndarray,
    ) -> torch.Tensor:
        """!
        Convert the input state to the relevant output using the neural network.

        @param state (np.ndarray): Input state for the network

        @return torch.Tensor: Output of the network
        """
        action_values = torch.from_numpy(state).float().to(self.device)
        ones = torch.ones_like(action_values)

        # Scale to 0, 1
        action_values = torch.where(action_values > 0.3, ones, 0)

        # Account for the batch dimension in torch modules
        batch = state.ndim == 4

        # If there is no batch, we add the batch dimension
        if not batch:
            action_values = action_values.unsqueeze(0).requires_grad_(False)

        action_values = action_values.permute(0, 3, 1, 2)

        action_values = self.cnn(action_values)

        if self.aggregation is not None:
            action_values = self.aggregation(
                *[stream(action_values) for stream in self.streams]
            )
        else:
            action_values = self.streams[0](action_values)

        action_values = action_values.to('cpu')
        return action_values if batch else action_values.squeeze()

    def add_stream(self, fcl_layers: List[int], output_dim: int):
        """Helper function to add an fcl stream to the network.

        This helper function allows the dueling architecture to add
        an extra stream parallel to the original fcl layers.

        @oaram fcl_layers (List[int]): Number of neurons in the streams
            fcl layers
        @param output_dim (int): Output dimension of the stream to add
        """
        layers = []
        n_features = self.cnn_features
        for n_neurons in fcl_layers:
            layers.append(nn.Linear(n_features, n_neurons))
            layers.append(nn.ReLU())
            n_features = n_neurons

        layers.append(nn.Linear(n_features, output_dim))

        self.streams = nn.ModuleList([*self.streams, nn.Sequential(*layers)])
        self.streams = self.streams.to(self.device)

    def add_aggregation(self, aggregator: callable):
        """Adds an aggregation method for the streams of the network.

        A helper function to allow the dueling architecture to add an
        aggregation method to the network.

        @param aggregator (callable): Function to apply to combine the values
            of the streams of the network
        """
        self.aggregation = aggregator

