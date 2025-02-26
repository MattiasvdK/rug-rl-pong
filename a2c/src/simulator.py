from environment import CatchEnv
from typing import Tuple, List
import numpy as np
from agentfactory import AgentFactory
from agent import Agent
from csvwriter import CSVWriter
from tqdm import tqdm

from gifwriter import GifWriter


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
        gif_dir: str = '../gifs/',
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
        self.agent_name = self.agent_factory.model_name()

        self.env = CatchEnv()
        self.observation, self.info = self.env.reset()

        parameter_names = agent_factory.parameter_names()

        self.writer = CSVWriter(
            data_dir=data_dir,
            model_name=agent_factory.model_name(),
            parameter_names=parameter_names,
        )
        self.gifwriter = GifWriter(data_dir=gif_dir)

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
            identifier: str,
    ) -> None:
        """Runs the simulations for one specific agent.

        This function is a helper method that runs the simulations
        for one agent.

        @param agent (Agent): the agent to be used in the simulation.
        @param identifier (str): the identifier of the agent
        """
        results = np.zeros((self.n_simulations, self.n_episodes))

        self.gifwriter.new_agent(f'{self.agent_name}_{identifier}')

        for sim in tqdm(range(self.n_simulations), 'Simulations'):
            agent.reset()
            self.gifwriter.new_simulation(sim)

            for episode in tqdm(range(self.n_episodes), 'Episodes'):
                save_gif = (episode + 1) % 1000 == 0 or episode == 0
                reward = self._game(agent, save_gif, episode)
                results[sim, episode] = reward

        self.writer.write_results(results, identifier)

    def _game(
        self,
        agent: Agent,
        save_gif: bool = False,
        game_index: int = 0,
    ) -> float:
        """Runs one entire game of Catch.

        Runs one game of Catch and returns the last reward obtained.
        The last reward indicates whether the ball was caught or not
        and is thus the only needed information for calculating
        success.

        @param agent (Agent): The agent to run the game with.

        @return (float): The last reward obtained.
        """
        states = []
        self.observation, self.info = self.env.reset()
        while True:
            if save_gif:
                states.append(self.observation)
            reward, terminated = self._step(agent)
            last_reward = reward
            if terminated:
                break

        if save_gif:
            self.gifwriter.write_trajectory(states, game_index)

        return last_reward

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

        agent.observe_result(reward, done, prev_observation)

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
