import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Dict, Optional

from rbj.environment import EnvironmentAdapter
from rbj.world import WorldModel
from rbj.memory import MemoryModule
from rbj.reasoning import Reasoner, ReasoningDecision


class CognitiveLoop:

    def __init__(
        self,
        environment: EnvironmentAdapter,
        world_model: WorldModel,
        memory: MemoryModule,
        reasoner: Optional[Reasoner] = None,
    ):
        self.environment = environment
        self.world_model = world_model
        self.memory = memory

        self.reasoner = reasoner or Reasoner()

        self.step_count = 0

        self.last_observation = None
        self.last_action = None
        self.last_decision: Optional[ReasoningDecision] = None

    def reset(self):

        self.step_count = 0
        self.last_action = None
        self.last_decision = None

        observation = self.environment.reset()

        self._process_observation(observation)

        self.last_observation = observation

        return observation

    def observe(self):

        observation = self.environment.observe()

        self._process_observation(observation)

        self.last_observation = observation

        return observation

    def _process_observation(self, observation):

        environment_data = observation.get(
            "environment",
            {},
        )

        if environment_data:

            self.world_model.update_environment(
                environment_data
            )

        self.world_model.update_from_perception(
            observation
        )

        self.memory.set_state(
            "world",
            self.world_model.to_dict(),
        )

        self.memory.add_history({
            "type": "OBSERVATION",
            "step": self.step_count,
            "observation": observation,
        })

    def reason(
        self,
        command: dict,
    ) -> ReasoningDecision:

        world = self.world_model.to_dict()

        memory = self.memory.summary()

        decision = self.reasoner.reason(
            world=world,
            command=command,
            memory=memory,
        )

        self.last_decision = decision

        self.memory.add_history({
            "type": "REASONING",
            "step": self.step_count,
            "command": command,
            "decision": decision.to_dict(),
        })

        return decision

    def step(self, action):

        self.last_action = action

        observation_before = self.last_observation

        observation_after = self.environment.step(
            action
        )

        self._process_observation(
            observation_after
        )

        self.memory.remember_episode(
            state_before=observation_before,
            command=action,
            action=action,
            state_after=observation_after,
            reward=None,
            success=None,
            result="EXECUTED",
        )

        self.step_count += 1

        self.last_observation = observation_after

        return observation_after

    def is_done(self):
        return self.environment.is_done()

    def get_world_state(self):
        return self.world_model.get_state()

    def get_world(self):
        return self.world_model.to_dict()

    def get_memory_summary(self):
        return self.memory.summary()

    def get_last_decision(self):
        return self.last_decision

    def close(self):
        self.environment.close()