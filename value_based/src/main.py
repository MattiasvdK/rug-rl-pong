from simulator import Simulator
from agentfactory import AgentFactory
from hyperparameters import get_hyperparameters, get_agent
import os

import sys

N_SIMULATIONS = 10
N_EPISODES = 10000
AGENTS = ['dqn', 'ddqn', 'duel', 'dqv', 'dqvm']

# Find the proper path regardless of where the program is executed from
DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../data/'
)


def run(agent_name: str = 'dqn'):
    """Runs the full logic of the training setup.

    This function runs the full logic of the training setup.
    It initializes the AgentFactory and the simulator and runs
    the latter to train the provided agent.

    @param agent_name: The name of the agent to train
    """
    agent_factory = AgentFactory(
        agent=get_agent(agent_name),
        param_dict=get_hyperparameters(agent_name)
    )

    simulator = Simulator(
        agent_factory=agent_factory,
        n_simulations=N_SIMULATIONS,
        n_episodes=N_EPISODES,
        data_dir=DATA_PATH,
    )
    simulator.run()


def print_usage():
    print("Usage:\tpython main.py <agent_name>",
          "\t'dqn' - Deep Q Learning",
          "\t'ddqn' - Double Q Learning",
          "\t'duel' - Dueling Architecture",
          "\t'dqv' - Deep Q-V Learning",
          "\t'dqvm' - Deep Q-V-Max Learning",
          sep='\n')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print_usage()
        quit(1)

    if len(sys.argv) == 1:
        agent = 'dqn'
    else:
        agent = sys.argv[1]

    if agent not in AGENTS:
        print_usage()
        quit(1)

    # Check if there is data for the specified agent
    if os.path.exists(f'../data/{agent}_params.csv'):
        # If it is, ask if it can be overwritten
        print(f'Do you wish to overwrite data for agent {agent}?')
        answer = input('[y/N]\n')
        if answer != 'y':
            quit(0)
        else:
            for filename in os.listdir(DATA_PATH):
                if agent in filename:
                    os.remove(DATA_PATH + filename)

    run(agent)
