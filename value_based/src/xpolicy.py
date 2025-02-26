import numpy as np
from abc import abstractmethod


class ExplorationPolicy:
    """Base class for Exploration Policies.

    The ExplorationPolicy class defines the interface for all
    exploration policies. The Exploration Policy is also used
    for providing the random action selection.
    """
    def __init__(self, num_actions):
        self.timestep = 0
        self.num_actions = num_actions

    @abstractmethod
    def __call__(self) -> bool:
        """!
        Returns whether to explore the environment or not.
        If not the agent should take the action with the highest expected
        reward.

        @return (bool): Whether to explore the environment or not.
        """
        return np.random.uniform() > 0.5

    @abstractmethod
    def step(self) -> None:
        """!
        A step to adapt the parameter of the exploration policy.
        """
        pass

    def choose_action(self) -> int:
        """!
        Returns a randomly selected action.
        """
        return np.random.choice(self.num_actions)

    def reset(self) -> None:
        """Reset the exploration policy.

        This function allows the same object to be used multiple times.
        """
        self.timestep = 0

