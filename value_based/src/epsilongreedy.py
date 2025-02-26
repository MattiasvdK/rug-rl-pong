from xpolicy import ExplorationPolicy
import numpy as np


class EpsilonGreedy(ExplorationPolicy):
    """Implements the EpsilonGreedy exploration policy.

    This class implements the EpsilonGreedy exploration policy
    using a starting epsilon, minimum epsilon and the timesteps
    needed to go from the first to the latter.
    """

    def __init__(
            self,
            num_actions: int,
            eps_max: float,
            eps_min: float,
            max_timesteps: int
    ):
        """Initializes the EpsilonGreedy exploration policy.

        @param num_actions (int): Number of available actions
        @param eps_max (float): Starting value of epsilon
        @param eps_min (float): Final value of epsilon
        @param max_timesteps (float): Maximum number of timesteps
        """
        ExplorationPolicy.__init__(self, num_actions)
        self.eps = np.linspace(eps_max, eps_min, max_timesteps)
        self.max_timesteps = max_timesteps - 1

    def __call__(self) -> bool:
        """Returns whether a random action should be taken.

        This method returns a bool indicating whether a random
        action should be taken based on the policy's epsilon.

        @returns bool indicating whether a random action should be taken.
        """
        return np.random.uniform() < self.eps[self.timestep]

    def step(self):
        """Performs one timestep forward reducing epsilon."""
        self.timestep = min(self.timestep+1, self.max_timesteps)
