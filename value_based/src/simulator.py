from environment import CatchEnv
from typing import Tuple
import numpy as np
from agentfactory import AgentFactory
from agent import Agent
from csvwriter import CSVWriter
from tqdm import tqdm

class Simulator:
    """Class that runs a multitude of simulations for a given agent.

    The Simulator class runs a multitude of simulations in order to
    obtain proper results as well as hyperparameter optimization.
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        n_simulations: int,
        n_episodes: int,
        data_dir: str = '../data/',
    ):
        """Initialize the Simulator class.

        @param agent_factory (AgentFactory): The agent factory that
            will generate the agents to train.
        @param n_simulations (int): The number of simulations to run
            each agent for
        @param n_episodes (int): The number of episodes of each
            training run
        @param data_dir (str): The data directory to store the results
        """
        self.agent_factory = agent_factory
        self.n_simulations = n_simulations
        self.n_episodes = n_episodes

        self.env = CatchEnv()
        self.observation, self.info = self.env.reset()

        parameter_names = agent_factory.parameter_names()

        self.writer = CSVWriter(
            data_dir=data_dir,
            model_name=agent_factory.model_name(),
            parameter_names=parameter_names,
        )

    def run(self):
        """Runs the simulator.

        This function runs the simulator. It trains each agent
        n_simulations times and writes the results to a CSV file.
        """
        for idx, info in enumerate(self.agent_factory):
            identifier = self._identify(idx)
            self.writer.write_parameters(info[1], identifier)
            self._simulate(info[0], identifier)

    def _simulate(
            self,
            agent: Agent,
            identifier: str = None,
    ) -> None:
        """Runs the simulations for one specific agent.

        This function is a helper method that runs the simulations
        for one agent.
        """
        self.observation, self.info = self.env.reset()
        results = []

        for _ in tqdm(range(self.n_simulations), 'Simulations'):
            rewards = np.zeros((self.n_episodes, 2))
            agent.reset()
            for episode in tqdm(range(self.n_episodes), 'Episodes'):
                reward, terminated = self._step(agent)
                rewards[episode][0] = reward
                rewards[episode][1] = terminated

            scores = self._obtain_scores(rewards)
            results.append(scores)

        self.writer.write_results(results, identifier)

    def _step(self, agent: Agent) -> Tuple[float, bool]:
        """Performs one step on the agent and environment.

        A helper function that performs the logic for one step.
        This code is reused from the main function provided for
        the assignment.

        @param agent (Agent): The current agent

        @return (float, bool): The obtained reward and whether
            the environment terminated
        """
        action = agent.choose_action(self.observation)
        prev_observation = self.observation
        self.observation, reward, terminated, truncated, self.info \
            = self.env.step(action)
        reward = float(reward)
        done = terminated or truncated

        agent.store_transition(prev_observation, action, reward,
                               self.observation, done)
        agent.learn()

        if done:
            self.observation, self.info = self.env.reset()

        return reward, terminated

    @staticmethod
    def _obtain_scores(results: np.ndarray) -> np.ndarray:
        """Obtains scores based on the terminated episodes.

        Reduces the obtained rewards to only include the terminated
        states.

        @param results (np.ndarray): Array of all rewards and
            termination status of the environment
        @params results (np.ndarray): Array of rewards in terminated
            episodes.
        """
        scores = results[results[:, 1] == 1]
        return scores[:, 0]

    @staticmethod
    def _identify(index: int) -> str:
        """Creates identification string based on index.

        @param index (int): The index of the agent
        @return (str) The identification string
        """
        ident = str(index)
        if len(ident) < 3:
            ident = '0' * (3 - len(ident)) + ident
        return ident
