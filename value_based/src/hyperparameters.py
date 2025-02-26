from epsilongreedy import EpsilonGreedy
from torch.optim import RMSprop
from torch.nn import MSELoss, HuberLoss

from agent import Agent
from dqnagent import DQNAgent
from ddqnagent import DDQNAgent
from duelingagent import DuelingAgent
from dqvagent import DQVAgent
from dqvmaxagent import DQVMaxAgent

XPOLICY = [
    EpsilonGreedy(3, 1, 0.0, 7000)
]

HP_AGENT = {
    'memory_size': [10000],
    'state_dimensions': [(84, 84, 4)],
    'n_actions': [3],
    'reset_index': [500],
    'initial_random': [750],
    'learning_rate': [5e-4],
    'reward_discount': [0.99],
    'batch_size': [64],
    'exploration_policy': XPOLICY,
    'optimizer': [RMSprop],
    'criterion': [MSELoss()],
}

HP_DQN = {
    'cnn_layers': [[7, 7, 7]],
    'fcl_layers': [[128]],
    'pooling': [True],
}

HP_DUEL = HP_DQN | {
    'value_layers': [[128]],
}

HP_DQV = {
    'svalue_cnn_layers': [[7, 7, 7]],
    'svalue_fcl_layers': [[128]],
    'svalue_pooling': [True],
    'qvalue_cnn_layers': [[7, 7, 7]],
    'qvalue_fcl_layers': [[128]],
    'qvalue_pooling': [True],
}

HP_SPEC = {
    'dqn': HP_DQN,
    'ddqn': HP_DQN,     # Shares all parameters with DQN
    'duel': HP_DUEL,
    'dqv': HP_DQV,
    'dqvm': HP_DQV,     # Shares all parameters with non-max DQV
}

AGENTS = {
    'dqn': DQNAgent,
    'ddqn': DDQNAgent,
    'duel': DuelingAgent,
    'dqv': DQVAgent,
    'dqvm': DQVMaxAgent,
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

