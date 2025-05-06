from template import Agent
import random


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, game_state):
        print(f"hand:  {game_state.agents[self.id].hand}")
        action = random.choice(actions)
        print(action)
        return action
