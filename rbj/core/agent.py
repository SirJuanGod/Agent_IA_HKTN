class SJGAgent:

    def __init__(
        self,
        brain
    ):

        self.brain = brain

    def act(
        self,
        state,
        epsilon=0.0
    ):

        return self.brain.choose_action(
            state,
            epsilon
        )

    def learn(self):

        return self.brain.train_step(
            batch_size=64
        )