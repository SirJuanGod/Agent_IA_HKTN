import random

import torch
import torch.nn as nn
import torch.optim as optim

from rbj.models.policy import SJGPolicy
from rbj.learning.replay import ReplayBuffer


class DQN:

    def __init__(
        self,
        state_size=6,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=10000
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Device:", self.device)

        self.policy = SJGPolicy(
            input_size=state_size,
            output_size=action_size
        ).to(self.device)

        self.target = SJGPolicy(
            input_size=state_size,
            output_size=action_size
        ).to(self.device)

        self.target.load_state_dict(
            self.policy.state_dict()
        )

        self.target.eval()

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

        self.memory = ReplayBuffer(
            capacity=buffer_size
        )

    def predict(self, state):

        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.policy(
                state_tensor
            )

        return q_values[0]

    def choose_action(
        self,
        state,
        epsilon
    ):

        # Exploración
        if random.random() < epsilon:

            return random.randrange(
                self.action_size
            )

        # Explotación
        q_values = self.predict(state)

        return torch.argmax(q_values).item()

    def train_step(self, batch_size):

        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s,a)
        current_q = self.policy(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Q(s',a')
        with torch.no_grad():

            next_q = self.target(
                next_states
            ).max(
                dim=1
            )[0]

            target_q = rewards + (
                self.gamma
                * next_q
                * (1 - dones)
            )

        loss = self.loss_function(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update_target(self):

        self.target.load_state_dict(
            self.policy.state_dict()
        )

    def save(self, path):

        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            path
        )

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )
        
        if "model_state_dict" in checkpoint:
            self.policy.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            # Fallback para checkpoints viejos
            self.policy.load_state_dict(checkpoint)

        self.update_target()