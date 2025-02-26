from dqnagent import DQNAgent


class DDQNAgent(DQNAgent):
    """Implementation of the Double Q-Learning agent.

    This class implements the Double Deep Q-Learning algorithm.
    It is a child class of DQNAgent. It adds the training setup for
    the theta-minus model.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the DDQNAgent.

        The initialization requires the same arguments as the DQNAgent
        except for the reset_index as that will be set to 1 for the
        DDQNAgent.

        This class requires the same arguments as the DQNAgent.

        @param args: The arguments for the DQNAgent
        @param kwargs: The keyword arguments for the DQNAgent
        """
        super(DDQNAgent, self).__init__(*args, **kwargs)

        self.optimizer_min = self.optimizer_class(self.model_min.parameters(),
                                                  lr=self.learning_rate)

    def _reset_theta(self) -> None:
        """Swaps the roles of the two models.

        This function swaps the roles of the two models and reinializes
        the optimizer to train the appropriate model.
        """
        tmp_net = self.model_min
        self.model_min = self.model
        self.model = tmp_net

        # TODO This can probably be more efficient
        tmp_optim = self.optimizer_min
        self.optimizer_min = self.optimizer
        self.optimizer = tmp_optim

    def reset(self) -> None:
        DQNAgent.reset(self)

        self.optimizer_min = self.optimizer_class(self.model_min.parameters(),
                                                  lr=self.learning_rate)

    @staticmethod
    def name() -> str:
        return 'ddqn'
