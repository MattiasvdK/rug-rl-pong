from agent import Agent
import torch
import numpy as np
from typing import List

from network import NeuralNetwork


class DQVAgent(Agent):
    """Implementation of the Deep Quality Value Learning Agent.

    This class implements the learning algorithm for Deep Quality
    Value Learning as described in 'Deep Quality Value Family of
    Deep Reinforcement Learning'.
    """
    def __init__(
            self,
            *args,
            svalue_cnn_layers: List[int] = [7, 7],
            svalue_fcl_layers: List[int] = [256, 128],
            svalue_pooling: bool = True,
            qvalue_cnn_layers: List[int] = [7, 7],
            qvalue_fcl_layers: List[int] = [256, 128],
            qvalue_pooling: bool = True,
            **kwargs,
    ):
        """Initialize the DQVAgent agent.

        @param args: Arguments for Agent baseclass
        @param svalue_cnn_layers (List[int]): List of conv kernel sizes
            for the state-value network
        @param svalue_fcl_layers (List[int]): List of neurons for fcl
            layers for the state-value network
        @param svalue_pooling (bool): Whether to use pooling in the
            state-value network
        @param qvalue_cnn_layers (List[int]): List of conv kernel sizes
            for the quality-value network
        @param qvalue_fcl_layers (List[int]): List of neurons for fcl
            for the quality-value network
        @param qvalue_pooling (bool): Whether to use pooling in the
            quality-value network
        @param kwargs: Keyword arguments for Agent baseclass
        """

        super(DQVAgent, self).__init__(*args, **kwargs)

        self.svalue_cnn_layers = svalue_cnn_layers
        self.svalue_fcl_layers = svalue_fcl_layers
        self.svalue_pooling = svalue_pooling
        self.svalue_optimizer = None

        self.qvalue_cnn_layers = qvalue_cnn_layers
        self.qvalue_fcl_layers = qvalue_fcl_layers
        self.qvalue_pooling = qvalue_pooling
        self.qvalue_optimizer = None

        self.qvalue_net = None
        self.svalue_net = None

        # For DQV s_net, for DQV Max q_net
        self.net_min = None

        self.reset()

    def choose_action(self, observation: np.ndarray) -> int:
        if self.step_count < self.initial_random or self.xpolicy():
            chosen_action = self.xpolicy.choose_action()
        else:
            expected_values = self.qvalue_net(observation)
            chosen_action = np.argmax(expected_values.detach().cpu().numpy())
        self.step_count += 1

        if self.step_count >= self.initial_random:
            self.xpolicy.step()

        return chosen_action

    def learn(self) -> None:
        """Train the networks using a batch from the replay buffer."""
        if self.step_count < self.initial_random:
            return

        if self.learn_count % self.reset_index:
            self._reset_theta()
        self.learn_count += 1

        batch = self._create_batch()

        self._train_network(batch, 'qvalue_net')
        self._train_network(batch, 'svalue_net')

    def _train_network(
            self,
            batch: dict,
            network_name: str,
    ) -> None:
        """Train one of the networks specifically.

        This function trains the network corresponding to the given
        name and is used to account for the differences in the learning
        setup, mostly the TD-error.

        @param batch (dict): Batch to train the network on
        @param network_name (str): Name of the network to train
        """

        if network_name == 'qvalue_net':
            network = self.qvalue_net
            optimizer = self.qvalue_optimizer
        else:
            network = self.svalue_net
            optimizer = self.svalue_optimizer

        network.train()

        output = network(batch['states']).cpu()
        rewards = self._create_rewards(batch, network_name)

        if network_name == 'qvalue_net':
            predictions = self._select_action_rewards(output, batch['actions'])
        else:
            predictions = output

        optimizer.zero_grad()
        loss = self.criterion(predictions, rewards)
        loss.backward()
        optimizer.step()

    def _create_rewards(
            self,
            batch: dict,
            network_name: str = None,
    ):
        """Creates the rewards for training.

        Creates the rewards for training the network.
        This function is mainly added to be used by
        the DQVMaxAgent class.

        @param batch (dict): Batch of data
        @param network_name (str): Name of the network for which to
            prepare the data.

        """
        return torch.tensor(batch['rewards'], dtype=torch.float)

    def _new_reward(
            self,
            new_states: np.ndarray,
    ) -> np.ndarray:
        with torch.no_grad():
            rewards = self.net_min(new_states).detach().cpu().numpy()

        return rewards.reshape(-1)

    def reset(self):
        Agent.reset(self)

        self.qvalue_net = self._q_net()
        self.svalue_net = self._s_net()

        self.net_min = self._min_net()

        self.qvalue_optimizer = self.optimizer_class(self.qvalue_net.parameters(),
                                                     lr=self.learning_rate)

        self.svalue_optimizer = self.optimizer_class(self.svalue_net.parameters(),
                                                     lr=self.learning_rate)

        self._reset_theta()

    def _reset_theta(self):
        """Resets the theta-minus parameters."""
        self.net_min.load_state_dict(self.svalue_net.state_dict())

    def _q_net(self) -> NeuralNetwork:
        """Helper function to create the Q-value network."""
        return NeuralNetwork(
            cnn_layers=self.qvalue_cnn_layers,
            fcl_layers=self.qvalue_fcl_layers,
            pooling=self.qvalue_pooling,
        )

    def _s_net(self) -> NeuralNetwork:
        """Helper function to create the S-value network."""
        return NeuralNetwork(
            cnn_layers=self.svalue_cnn_layers,
            fcl_layers=self.svalue_fcl_layers,
            pooling=self.svalue_pooling,
            out_features=1,
        )

    def _min_net(self):
        """Helper function to create correct Theta-minus network"""
        return self._s_net()

    @staticmethod
    def name() -> str:
        return 'dqv'





