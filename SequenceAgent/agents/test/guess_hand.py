from template import Agent
import random


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.guesses = []
        self.op_last = 0

    def SelectAction(self, actions, game_state):
        self.guess_hand(game_state)
        print(f"guess: {self.guesses}")
        action = random.choice(actions)
        print(action)
        return action

    def guess_hand(self, game_state):
        op_trace = game_state.agents[1-self.id].agent_trace.action_reward[self.op_last:]
        self.op_last = len(game_state.agents[1-self.id].agent_trace.action_reward)
        p, d = self.extract(op_trace)
        for each in p:
            self.guesses.append(each)
        for each in d:
            if each in self.guesses:
                self.guesses.remove(each)

    def extract(self, trace:list):
        picked = []
        discarded = []
        for action, r in trace:
            if action["draft_card"] is not None:
                picked.append(action["draft_card"])
            if action["play_card"] is not None:
                discarded.append(action["play_card"])
        return picked, discarded
