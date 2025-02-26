from reinforceagent import ReinforceAgent
from agent import Agent

HP_AGENT = {
    'n_actions': [3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'discount_factor': [0.95, 0.99],
    'n_trajectories': [32, 64],
}

HP_A2C = {
    'cnn_layers': [[5, 5, 5]],
    'pooling': [False],
    'fcl_layers': [[256, 128]],
    'negative_scale': [-1.0],
    'positive_scale': [1.0],
    'theta_delay': [21, 42]
}

HP_SPEC = {
    'a2c': HP_A2C
}

AGENTS = {
    'rnf': ReinforceAgent,
}


def get_agent(agent: str) -> Agent:
    """Returns the agent class corresponding to the given name.

    @param agent (str): The name of the agent

    @return Agent: The agent class corresponding to the given name
    """
    return AGENTS[agent]


def get_hyperparameters(agent: str) -> dict:
    """Returns the hyperparameters of the given agent.

    @param agent (str): The name of the agent

    @return dict: The hyperparameters of the given agent
    """
    return HP_AGENT | HP_SPEC[agent]