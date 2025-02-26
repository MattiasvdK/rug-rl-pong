from agent import Agent
from typing import List

import torch
import numpy as np

from network import NeuralNetwork


class DQNAgent(Agent):
    """Deep Q-Learning Agent.

    This class implements the Deep Q-Learning Algorithm as Reinforcemtent
    Learning Agent. The class inherits from the abtract Agent class.
    """

    # Needs:    One CNN to train
    #           One copy after iterations to solve loss
    #           

    def __init__(
        self,
        *args,
        cnn_layers: List[int] = [7],
        fcl_layers: List[int] = [256],
        pooling: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the agent.

        @param args: Arguments for the base agent class
        @param cnn_layers (List[int]): CNN layer kernel sizes
        @param fcl_layers (List[int]): FCL layer number of neurons
        @param pooling (bool): Whether to use pooling after each CNN
        @param kwargs: Keyword arguments for the base agent class
        """
        super(DQNAgent, self).__init__(*args, **kwargs)
        self.model = None
        self.model_min = None
        self.optimizer = None

        self.cnn_layers = cnn_layers
        self.fcl_layers = fcl_layers
        self.pooling = pooling

        DQNAgent.reset(self)

    def choose_action(self, observation: np.ndarray) -> int:
        """Selects an action based on the State-Action value function.
        
        The State-Action value function is approximated by the neural
        network of the model.

        @param observation (np.ndarray): Vector describing the current state
        
        @return (int): the action to take
        """
        if self.step_count < self.initial_random or self.xpolicy():
            chosen_action = self.xpolicy.choose_action()
        else:
            expected_values = self.model(observation)
            chosen_action = np.argmax(expected_values.detach().cpu().numpy())
        self.step_count += 1

        if self.step_count >= self.initial_random:
            self.xpolicy.step()

        return chosen_action

    def learn(self) -> None:
        """The learning step of the agent.

        This function updates the prediction model based on a random
        sample from the memory buffer. It updates the parameters based
        on the Loss.
        """

        # No learning before the needed steps have been taken
        if self.step_count < self.initial_random:
            return

        # Reset the theta-minus parameters to theta if step = c
        if self.learn_count % self.reset_index == 0:
            self._reset_theta()
        self.learn_count += 1

        self.optimizer.zero_grad()

        # Obtain the batch with the needed values
        batch = self._create_batch()
        states = batch['states']
        actions = batch['actions']
        rewards = torch.tensor(batch['rewards'], dtype=torch.float)

        self.model.train()
        # Get the model output and select the chosen action output
        output = self.model(states).cpu()

        # Select the correct actions through masking
        predictions = self._select_action_rewards(output, actions)

        # Calculate the loss and perform gradient descent
        loss = self.criterion(predictions, rewards)
        loss.backward()
        self.optimizer.step()

    def _reset_theta(self):
        """Reset the parameters of theta-minus."""
        self.model_min.load_state_dict(self.model.state_dict())

    def _new_reward(
            self,
            new_states: np.ndarray,
    ) -> np.ndarray:
        """Obtains the expected reward through network theta-minus.

        @param new_states (np.ndarray): batch of new states

        @return (np.ndarray): the predicted maximum rewards
        """
        with torch.no_grad():
            rewards = self.model_min(new_states).cpu().detach().numpy()

        return np.max(rewards, axis=1)

    def reset(self) -> None:
        Agent.reset(self)

        self.model = NeuralNetwork(
            cnn_layers=self.cnn_layers,
            fcl_layers=self.fcl_layers,
            pooling=self.pooling,
        )

        self.model_min = NeuralNetwork(
            cnn_layers=self.cnn_layers,
            fcl_layers=self.fcl_layers,
            pooling=self.pooling,
        )

        self.optimizer = self.optimizer_class(self.model.parameters(),
                                              lr=self.learning_rate)

        self.model_min.load_state_dict(self.model.state_dict())
        self.step_count = 0
        self.learn_count = 0

    @staticmethod
    def name() -> str:
        return 'dqn'

