from agents.agent import Agent
from gym_env import PokerEnv
import random

class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

    def act(self, observation, reward, terminated, truncated, info):
        """
        Simple starter: Discards randomly on the flop, otherwise plays randomly.
        """
        # Example of using the logger
        if observation["street"] == 0 and info["hand_number"] % 50 == 0:
            self.logger.info(f"Hand number: {info['hand_number']} ASDFSDAFDSAF")

        valid_actions = observation["valid_actions"]
        
        # Initialize default response (must be 4 values)
        action_type = 0
        raise_amount = 0
        keep_1, keep_2 = 0, 0

        # 1. Handle Discard Phase (Mandatory for the new variant)
        if valid_actions[self.action_types.DISCARD.value]:
            # Variant: We are dealt 5 cards, we must pick 2 to keep.
            indices = random.sample(range(5), 2)
            return self.action_types.DISCARD.value, 0, indices[0], indices[1]

        # 2. Handle Betting Phase
        valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        action_type = random.choice(valid_indices)

        if action_type == self.action_types.RAISE.value:
            raise_amount = observation["min_raise"]

        # Return the 4-tuple required by the API schema
        return action_type, raise_amount, keep_1, keep_2

    def observe(self, observation, reward, terminated, truncated, info):
        # Optional: update internal state or log hand results
        if terminated:
            self.logger.info(f"Hand Over. Reward: {reward}")
