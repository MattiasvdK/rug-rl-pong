import numpy as np
import torch

from dqvagent import DQVAgent


class DQVMaxAgent(DQVAgent):
    """Implements the DQV-Max Algorithm.

    This class implements the DQV-Max variation of the DQV algorithm
    and thus extends that repsective class. The implementation is
    based on the paper 'Deep Quality Value Family of Deep Reinforcement
    Learning'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _reset_theta(self):
        self.net_min.load_state_dict(self.qvalue_net.state_dict())

    def _min_net(self):
        return self._q_net()

    def _create_rewards(
            self,
            batch: dict,
            network_name: str = None,
    ) -> torch.Tensor:
        """Creates the TD reward for the networks.

        This function creates the two rewards since both networks
        require a different target value based on the other network.

        @param batch (dict): The batch of experience replay
        @param network_name (str): The name of the network to create
            the rewards for

        @return (torch.Tensor): The targets for the specified network
        """
        rewards = super()._create_rewards(batch)
        terminal = torch.from_numpy(batch['terminal']).bool()

        with torch.no_grad():
            if network_name == 'qvalue_net':
                addition = self.svalue_net(batch['new_states'])
            else:
                addition = self.net_min(batch['new_states'])
                addition = self._select_action_rewards(addition, batch['actions'])

        addition = torch.where(~terminal, addition, 0)

        rewards = torch.add(rewards, addition)

        return rewards

    def _new_reward(
            self,
            new_states: np.ndarray,
    ) -> np.ndarray:
        """Implements new reward function returning 0.

        The reward obtained is zero to be able to adapt the rewards of
        the batch to the two different networks in the learn function.
        """
        return np.zeros(self.batch_size)

    @staticmethod
    def name() -> str:
        return 'dqvm'
