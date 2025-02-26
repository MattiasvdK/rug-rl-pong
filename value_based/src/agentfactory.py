from agent import Agent
from typing import List, Tuple, Dict
from sklearn.model_selection import ParameterGrid


class AgentFactory:

    def __init__(self, agent, param_dict: Dict) -> None:
        """Initialize the initializer.

        The initializer creates a parameter dictionary over the
        hyperparameters of the agent. It is an iterable that returns
        initialized agents.

        @param (Agent) agent: The agent to be initialized.
        @param (dict) param_dict: Hyperparameters to use for
                                  initialization.
        """

        self.agent = agent
        self.param_dict = param_dict
        self.n_params = len(param_dict)

        self.param_grid = list(ParameterGrid(param_dict))
        self.n_combinations = len(self.param_grid)

        print(f'Parameter grid size: {self.n_combinations}')

        self.idx = -1

    def parameter_names(self) -> List[str]:
        return list(self.param_dict.keys())

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Agent, Dict]:
        # TODO check if need raise StopIteration
        self.idx += 1
        if self.idx >= self.n_combinations:
            raise StopIteration
        params = self.param_grid[self.idx]
        return self.agent(**params), params

    def __len__(self) -> int:
        return self.n_combinations

    def model_name(self):
        return self.agent.name()


