import torch
import random


class ReplayBuffer:

    def __init__(self, capacity=10000, device="cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        # Convertir a tensores si no lo son
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long, device=self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Inicialización perezosa de los buffers
        if self.states is None:
            self.states = torch.zeros((self.capacity,) + state.shape, dtype=torch.float32, device=self.device)
            self.actions = torch.zeros((self.capacity,) + action.shape, dtype=torch.long, device=self.device)
            self.rewards = torch.zeros((self.capacity,) + reward.shape, dtype=torch.float32, device=self.device)
            self.next_states = torch.zeros((self.capacity,) + next_state.shape, dtype=torch.float32, device=self.device)
            self.dones = torch.zeros((self.capacity,) + done.shape, dtype=torch.float32, device=self.device)

        # Guardar en el buffer
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size