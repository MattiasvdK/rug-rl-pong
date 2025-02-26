from ddqnagent import DDQNAgent
import torch


class DuelingAgent(DDQNAgent):
    """Class implementing the Dueling Architecture.

    This class implements the Dueling Architecture as described in
    'Dueling Architecture Learning for Deep Reinforcement Learning'
    and uses the Double Deep Q-Learning algorithm to train the agent.
    """

    def __init__(
            self,
            *args,
            value_layers=[256],
            **kwargs,
    ):
        """Initialize the Dueling Architecture Agent.

        @param args: Arguments for the DDQN parent class
        @param value_layers: Number of fcl layers for the value stream
        @param kwargs: Keyword arguments for the DDQN parent class
        """
        super(DuelingAgent, self).__init__(*args, **kwargs)
        self.value_layers = value_layers

        self.reset()

    def reset(self):
        DDQNAgent.reset(self)

        self.model.add_stream(self.value_layers, 1)
        self.model_min.add_stream(self.value_layers, 1)

        self.model.add_aggregation(DuelingAgent.av_aggregation)
        self.model_min.add_aggregation(DuelingAgent.av_aggregation)

        self.optimizer = self.optimizer_class(self.model.parameters(),
                                              lr=self.learning_rate)

        self.optimizer_min = self.optimizer_class(self.model_min.parameters(),
                                                  lr=self.learning_rate)

    @staticmethod
    def name():
        return 'duel'

    @staticmethod
    def av_aggregation(adv: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        """Averate aggregation function of Value and Advantage.

        This function implements the average aggregation function of the
        Value and Advantage of the states and actions as described in
        equation (9) of 'Dueling Architecture Learning for Deep Reinforcement'.

        @param adv (torch.Tensor): Advantage of the states and actions.
        @param val (torch.Tensor): Values of the states and actions.

        @return torch.Tensor: The aggregated Q-values.
        """
        q_s = adv.clone()
        q_s -= torch.sum(adv, dim=-1, keepdim=True)
        q_s += val
        return q_s

