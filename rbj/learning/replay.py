import random


class ReplayBuffer:

    def __init__(self, capacity=10000):

        self.capacity = capacity
        self.buffer = []

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append(experience)

    def sample(self, batch_size):

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):

        return len(self.buffer)