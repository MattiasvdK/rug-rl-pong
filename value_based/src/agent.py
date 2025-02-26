from abc import abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict
from xpolicy import ExplorationPolicy


class Agent:
    def __init__(
        self,
        memory_size: int,
        state_dimensions: Tuple[int, int, int],
        n_actions: int,
        learning_rate: float,
        reward_discount: float,
        batch_size: int,
        exploration_policy: ExplorationPolicy,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        initial_random: int,
        reset_index: int,
    ) -> None:
        """!
        Initializes the agent.
        Agent is an abstract class that should be inherited by any
        agent that wants to interact with the environment. The agent
        should be able to store transitions, choose actions based on
        observations, and learn from the transitions.

        @param memory_size (int): Size of the memory buffer
        @param state_dimensions (int): Number of dimensions of the state space
        @param n_actions (int): Number of actions the agent can take
        @param learning_rate (float): Learning rate of the agent
        @param reward_discount (float): Discount factor of the TD-Error
        @param batch_size (int): Batch size of the learning algorithm
        @param exploration_policy (ExplorationPolicy): Object defining the
                                                       exploration policy.
        @param optimizer (optim.Optimizer): optimizer class for training the
                                            network
        @param criterion (nn.Module): Loss function
        @param initial_random (int): Initial random number actions to populate
                                     the memory buffer
        @param reset_index (int): The amount of steps before resetting the
                                  theta-minus network.
        """

        self.batch_size = batch_size
        self.xpolicy = exploration_policy
        self.optimizer_class = optimizer
        self.criterion = criterion

        self.initial_random = initial_random
        self.reset_index = reset_index

        self.learn_count = 0
        self.step_count = 0

        self.n_actions = n_actions
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.reward_discount = reward_discount
        self.state_dimensions = state_dimensions

        self.memory_end = 0
        self.memory_idx = 0

        self.state_buffer = np.zeros(
            (self.memory_size, *self.state_dimensions),
            dtype=np.float32
        )
        self.new_state_buffer = np.zeros(
            (self.memory_size, *self.state_dimensions),
            dtype=np.float32
        )
        self.action_buffer = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(self.memory_size, dtype=bool)

        # Reset should be called at the end of the agent class

    def store_transition(
        self,
        state: np.ndarray,
        action: int, # Is this always an int? It can be
        reward: float,
        new_state: np.ndarray,
        done: bool,
    ) -> None:
        """!
        Stores the state transition for later memory replay.
        Make sure that the memory buffer does not exceed its maximum
        size.

        Hint: after reaching the limit of the memory buffer, maybe you
        should start overwriting the oldest transitions?

        @param state        (np.ndarray): Vector describing current state
        @param action       (int): Action taken
        @param reward       (float): Received reward
        @param new_state    (list): Newly observed state
        @param done         (bool): Whether the state was terminal or not
        """
        self.state_buffer[self.memory_idx] = state
        self.action_buffer[self.memory_idx] = action
        self.reward_buffer[self.memory_idx] = reward
        self.new_state_buffer[self.memory_idx] = new_state
        self.terminal_buffer[self.memory_idx] = done

        self.memory_idx = (self.memory_idx + 1) % self.memory_size
        self.memory_end += 1 if self.memory_end < self.memory_size else 0

    @abstractmethod
    def choose_action(
        self,
        observation: np.ndarray
    ) -> int: # Is this always an int?
        """!
        Abstract method that should be implemented by the child class,
        e.g. DQN or DDQN agents. This method should contain the full
        logic needed to choose an action based on the current state.
        Maybe you can store the neural network in the agent class and
        use it here to decide which action to take?

        @param observation (np.ndarray): Vector describing current state

        @return (int): Action to take
        """

        return 0

    @abstractmethod
    def learn(self) -> None:
        """!
        Update the parameters of the internal networks.
        This method should be implemented by the child class.
        """

        pass

    def _create_batch(self) -> Dict[str, np.ndarray]:
        """Function that samples a batch from the memory buffer.

        This function samples a batch from the memory buffer and
        adds the expected reward for the next state.

        @return (Dict[str, np.ndarray]): Dictionary containing the batch
        """

        # Obtain a random selection of steps
        batch_indexes = np.random.choice(
            self.memory_end,
            size=self.batch_size,
            replace=False,
        )

        # TODO memory idx loops around, fix for batch selection
        states = self.state_buffer[batch_indexes]
        actions = self.action_buffer[batch_indexes]
        rewards = self.reward_buffer[batch_indexes]
        new_states = self.new_state_buffer[batch_indexes]
        terminal = self.terminal_buffer[batch_indexes]

        rewards += (self.reward_discount
                    * np.where(~terminal,
                               self._new_reward(new_states), 0)
                    )

        rewards.resize((self.batch_size, 1))

        return {
            'states': states,
            'new_states': new_states,
            'terminal': terminal,
            'actions': actions,
            'rewards': rewards,
        }

    @abstractmethod
    def _new_reward(
            self,
            new_states: np.ndarray,
    ) -> np.ndarray:
        """!
        The function to calculate the expected reward of the new state.
        To be implemented by the child class.

        @param new_states (np.ndarray): array of environment states

        @return (np.ndarray): Expected maximum reward of theta-minus
        """
        return np.zeros((self.batch_size, 1))

    def reset(self) -> None:
        """Resets the agent to train again.

        This function resets all parameters to prepare for another
        training run. Resets the counters and the policy.

        The function should be overwritten by any child class.
        """
        self.xpolicy.reset()

        self.memory_idx = 0
        self.memory_end = 0

        self.step_count = 0
        self.learn_count = 0

    def _select_action_rewards(
            self,
            rewards: torch.Tensor,
            actions: np.ndarray,
    ) -> torch.Tensor:
        """Select rewards specific to chosen actions.

        Select the rewards based on action index. This selects the
        relevant rewards based on the action used to transition to
        the next state from all rewards predicted by the model.
        Used to find the model predictions to calculate the Loss.

        @param rewards (torch.Tensor): The predicted rewards
        @param actions (np.ndarray): The actions from the buffer

        @return (torch.Tensor): The predicted rewards of the actions
        """

        selected_rewards = torch.zeros(rewards.shape[0])
        for idx in range(rewards.shape[0]):
            selected_rewards[idx] = rewards[idx, actions[idx]]
        return selected_rewards.view((self.batch_size, 1))

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Returns the name of the agent for file naming.

        To be overwritten by any child class.

        @return (str): Name of the agent.
        """
        return 'agent'
