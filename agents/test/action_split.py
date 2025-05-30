from copy import deepcopy

from template import Agent
from Sequence.sequence_model import *
from Sequence.sequence_utils import *
import random

# -------------------------------- CONSTANTS --------------------------------
# CARD_MAPPING = {
#         '2c': [(1, 4), (3, 6)], '2d': [(2, 2), (5, 9)], '2h': [(5, 4), (8, 7)], '2s': [(0, 1), (8, 6)],
#         '3c': [(1, 3), (3, 5)], '3d': [(2, 3), (6, 9)], '3h': [(5, 5), (8, 8)], '3s': [(0, 2), (8, 5)],
#         '4c': [(1, 2), (3, 4)], '4d': [(2, 4), (7, 9)], '4h': [(4, 5), (7, 8)], '4s': [(0, 3), (8, 4)],
#         '5c': [(1, 1), (3, 3)], '5d': [(2, 5), (8, 9)], '5h': [(4, 4), (6, 8)], '5s': [(0, 4), (8, 3)],
#         '6c': [(1, 0), (3, 2)], '6d': [(2, 6), (9, 8)], '6h': [(4, 3), (5, 8)], '6s': [(0, 5), (8, 2)],
#         '7c': [(2, 0), (4, 2)], '7d': [(2, 7), (9, 7)], '7h': [(4, 8), (5, 3)], '7s': [(0, 6), (8, 1)],
#         '8c': [(3, 0), (5, 2)], '8d': [(3, 7), (9, 6)], '8h': [(3, 8), (6, 3)], '8s': [(0, 7), (7, 1)],
#         '9c': [(4, 0), (6, 2)], '9d': [(4, 7), (9, 5)], '9h': [(2, 8), (6, 4)], '9s': [(0, 8), (6, 1)],
#         'tc': [(5, 0), (7, 2)], 'td': [(5, 7), (9, 4)], 'th': [(1, 8), (6, 5)], 'ts': [(1, 9), (5, 1)],
#         'ac': [(7, 5), (8, 0)], 'ad': [(7, 6), (9, 1)], 'ah': [(1, 5), (4, 6)], 'as': [(2, 1), (4, 9)],
#         'kc': [(7, 0), (7, 4)], 'kd': [(7, 7), (9, 2)], 'kh': [(1, 6), (5, 6)], 'ks': [(3, 1), (3, 9)],
#         'qc': [(6, 0), (7, 3)], 'qd': [(6, 7), (9, 3)], 'qh': [(1, 7), (6, 6)], 'qs': [(2, 9), (4, 1)]
#     }
#
# LOCATION_MAPPING = {
#     (0, 1): '2s', (0, 2): '3s', (0, 3): '4s', (0, 4): '5s',
#     (0, 5): '6s', (0, 6): '7s', (0, 7): '8s', (0, 8): '9s',
#     (1, 0): '6c', (1, 1): '5c', (1, 2): '4c', (1, 3): '3c', (1, 4): '2c',
#     (1, 5): 'ah', (1, 6): 'kh', (1, 7): 'qh', (1, 8): 'th', (1, 9): 'ts',
#     (2, 0): '7c', (2, 1): 'as', (2, 2): '2d', (2, 3): '3d', (2, 4): '4d',
#     (2, 5): '5d', (2, 6): '6d', (2, 7): '7d', (2, 8): '9h', (2, 9): 'qs',
#     (3, 0): '8c', (3, 1): 'ks', (3, 2): '6c', (3, 3): '5c', (3, 4): '4c',
#     (3, 5): '3c', (3, 6): '2c', (3, 7): '8d', (3, 8): '8h', (3, 9): 'ks',
#     (4, 0): '9c', (4, 1): 'qs', (4, 2): '7c', (4, 3): '6h', (4, 4): '5h',
#     (4, 5): '4h', (4, 6): 'ah', (4, 7): '9d', (4, 8): '7h', (4, 9): 'as',
#     (5, 0): 'tc', (5, 1): 'ts', (5, 2): '8c', (5, 3): '7h', (5, 4): '2h',
#     (5, 5): '3h', (5, 6): 'kh', (5, 7): 'td', (5, 8): '6h', (5, 9): '2d',
#     (6, 0): 'qc', (6, 1): '9s', (6, 2): '9c', (6, 3): '8h', (6, 4): '9h',
#     (6, 5): 'th', (6, 6): 'qh', (6, 7): 'qd', (6, 8): '5h', (6, 9): '3d',
#     (7, 0): 'kc', (7, 1): '8s', (7, 2): 'tc', (7, 3): 'qc', (7, 4): 'kc',
#     (7, 5): 'ac', (7, 6): 'ad', (7, 7): 'kd', (7, 8): '4h', (7, 9): '4d',
#     (8, 0): 'ac', (8, 1): '7s', (8, 2): '6s', (8, 3): '5s', (8, 4): '4s',
#     (8, 5): '3s', (8, 6): '2s', (8, 7): '2h', (8, 8): '3h', (8, 9): '5d',
#     (9, 1): 'ad', (9, 2): 'kd', (9, 3): 'qd', (9, 4): 'td',
#     (9, 5): '9d', (9, 6): '8d', (9, 7): '7d', (9, 8): '6d',
# }

ONE_EYED_JACKS = ["js", "jh"]
TWO_EYED_JACKS = ["jc", "jd"]
JACKS = ONE_EYED_JACKS + TWO_EYED_JACKS

# -------------------------------- UTILS --------------------------------
# Two eyed jacks can be placed anywhere EMPTY
def get_two_eyed_pos(chips):
    res = []
    for i in range(10):
        for j in range(10):
            if (i, j) in COORDS['jk']:
                continue
            elif chips[i][j] == EMPTY:
                res.append((i, j))
    return res

# One eyed jacks can remove one opponents chip
def get_one_eyed_pos(chips, oc):
    res = []
    for i in range(10):
        for j in range(10):
            if (i, j) in COORDS['jk']:
                continue
            elif chips[i][j] == oc:
                res.append((i, j))
    return res

# Normal cards can be placed into its position when EMPTY
def get_normal_pos(chips, card):
    res = []
    for (i, j) in COORDS[card]:
        if chips[i][j] == EMPTY:
            res.append((i, j))
    return res

def reconstruct_action(chips, normal, one_eyed_jacks, two_eyed_jacks, drafts, allow_trade, oc):
    need_trade = False
    dead = []
    non_trade = []
    trade = []
    for each in normal:
        positions = get_normal_pos(chips,each)
        if not positions:
            need_trade = True
            dead.append(each)
            for d in drafts:
                trade.append({"type":"trade","play_card":each,"draft_card":d,"coords":None})
        else:
            for position in positions:
                for d in drafts:
                    non_trade.append({"type":"place","play_card":each,"draft_card":d,"coords":position})
    for each in one_eyed_jacks:
        positions = get_one_eyed_pos(chips,oc)
        for position in positions:
            for d in drafts:
                non_trade.append({"type": "remove", "play_card": each, "draft_card": d, "coords": position})
    for each in two_eyed_jacks:
        positions = get_two_eyed_pos(chips)
        for position in positions:
            for d in drafts:
                non_trade.append({"type": "place", "play_card": each, "draft_card": d, "coords": position})
    if need_trade and allow_trade:
        trade.append({"type": "trade", "play_card": None, "draft_card": None, "coords": None})
        return trade
    else:
        return non_trade

def advanced_actions(chips, normal, one_eyed_jacks, two_eyed_jacks, drafts, allow_trade, oc):
    need_trade = False
    normal_actions = []
    one_eyed_jacks_actions = []
    two_eyed_jacks_actions = []
    dead = []
    normal_positions = []
    for each in normal:
        positions = get_normal_pos(chips, each)
        if not positions:
            need_trade = True
            dead.append(each)
        else:
            normal_positions.extend(positions)
            for position in positions:
                normal_actions.append((each, position, False))
    if need_trade and allow_trade:
        for d in drafts:
            if d in ONE_EYED_JACKS:
                positions = get_one_eyed_pos(chips, oc)
                for position in positions:
                    one_eyed_jacks_actions.append((d, position, True))
            elif d in TWO_EYED_JACKS:
                positions = get_two_eyed_pos(chips)
                for position in positions:
                    # never place a jack into a position where your normal card could
                    if position in normal_positions:
                        continue
                    two_eyed_jacks_actions.append((d, position, True))
            else:
                positions = get_normal_pos(chips, d)
                for position in positions:
                    normal_actions.append((d, position, True))
    for each in one_eyed_jacks:
        positions = get_one_eyed_pos(chips, oc)
        for position in positions:
            one_eyed_jacks_actions.append((each, position, False))
    for each in two_eyed_jacks:
        positions = get_two_eyed_pos(chips)
        for position in positions:
            # never place a jack into a position where your normal card could
            if position in normal_positions:
                continue
            two_eyed_jacks_actions.append((each, position, False))
    return normal_actions, one_eyed_jacks_actions, two_eyed_jacks_actions

# -------------------------------- CLASSES --------------------------------

class myAgent(Agent):

    def __init__(self, _id):
        super().__init__(_id)
        # ---------- basic game info ----------
        self.deck = [
            (r + s)
            for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a']
            for s in ['d', 'c', 'h', 's']
        ]
        self.deck = self.deck * 2
        self.chips = None
        self.last_draft = []
        self.my_score = 0
        self.op_score = 0
        # ---------- basic my info ----------
        self.clr = BLU if _id % 2 else RED
        self.sclr = BLU_SEQ if _id % 2 else RED_SEQ
        self.my_hand = []
        self.my_normal = []
        self.my_dead = []
        self.my_one_eyed_jacks = []
        self.my_two_eyed_jacks = []
        self.draft = []
        self.my_trade = False
        # ---------- basic opp info ----------
        self.oc = RED if _id % 2 else BLU
        self.os = RED_SEQ if _id % 2 else BLU_SEQ
        self.op_trade = True
        self.op_trace_last = 0
        self.op_hand = []
        self.op_normal = []
        self.op_dead = []
        self.op_one_eyed_jacks = []
        self.op_two_eyed_jacks = []

    def SelectAction(self, actions, game_state:SequenceState):
        self.extract_info(game_state)
        n,o,t = advanced_actions(
            self.chips,
            self.my_normal,
            self.my_one_eyed_jacks,
            self.my_two_eyed_jacks,
            self.draft,
            not self.my_trade,
            self.oc
        )
        try:
            for _,_,trade in n:
                if trade:
                    for eachline in self.chips:
                        print(eachline)
                    print(self.draft)
                    print(self.my_hand)
                    print(trade)
                    print(n)
                    print(o)
                    print(t)
                    break
        except Exception as e:
            print(e)
        action = random.choice(actions)
        # print(action)
        return action


    def extract_info(self, game_state:SequenceState):
        # At Start Remove cards in my hand from deck
        if self.op_trace_last == 0:
            for each in game_state.agents[self.id].hand:
                self.deck.remove(each)
        # Remove all unseen draft from deck
        for each in game_state.board.draft:
            if each not in self.last_draft:
                self.deck.remove(each)
        # Save current draft status
        self.last_draft = []
        for each in game_state.board.draft:
            self.last_draft.append(each)
        self.chips = deepcopy(game_state.board.chips)
        self.draft = deepcopy(game_state.board.draft)
        self.extract_my_info(game_state)
        self.extract_opp_info(game_state)

    def extract_my_info(self, game_state:SequenceState):
        myself = game_state.agents[self.id]
        self.my_trade = myself.trade
        self.my_hand = deepcopy(myself.hand)
        self.my_normal = []
        self.my_one_eyed_jacks = []
        self.my_two_eyed_jacks = []
        for each in self.my_hand:
            if each in ONE_EYED_JACKS:
                self.my_one_eyed_jacks.append(each)
            elif each in TWO_EYED_JACKS:
                self.my_two_eyed_jacks.append(each)
            else:
                self.my_normal.append(each)
        # print(f"normal: {self.normal}")
        # print(f"one_eyed_jacks: {self.one_eyed_jacks}")
        # print(f"two_eyed_jacks: {self.two_eyed_jacks}")
        # print(f"draft: {self.draft}")

    def extract_opp_info(self, game_state:SequenceState):
        # Get what card opponent picked and discard
        self.op_trade = False
        trace = game_state.agents[1 - self.id].agent_trace.action_reward[self.op_trace_last:]
        self.op_trace_last = len(game_state.agents[1 - self.id].agent_trace.action_reward)
        picked = []
        discarded = []
        for action, r in trace:
            if action["draft_card"] is not None:
                picked.append(action["draft_card"])
            if action["play_card"] is not None:
                discarded.append(action["play_card"])
        # Add all cards picked by opponent to op_hand
        for each in picked:
            self.op_hand.append(each)
        # Remove from op_hand if we know it is in opponent's hand
        # Otherwise remove it from deck
        for each in discarded:
            if each in self.op_hand:
                self.op_hand.remove(each)
            else:
                self.deck.remove(each)


if __name__ == "__main__":
    agent = myAgent(0)
    test_chips = [[EMPTY for _ in range(10)] for _ in range(10)]
    for x, y in COORDS["jk"]:
        test_chips[x][y] = JOKER
    for x, y in [(4,i) for i in range(7) if i !=4]:
        test_chips[x][y] = RED

    deck = deepcopy(agent.deck)
    random.shuffle(deck)
    my_hand = deck[:6]
    normal = []
    one_eyed_jacks = []
    two_eyed_jacks = []

    for each in my_hand:
        if each in ONE_EYED_JACKS:
            one_eyed_jacks.append(each)
        elif each in TWO_EYED_JACKS:
            two_eyed_jacks.append(each)
        else:
            normal.append(each)

    for each in normal:
        print(each, get_normal_pos(test_chips, each, agent.oc))
    if one_eyed_jacks:
        print("has one eyed jack")
        print(get_one_eyed_pos(test_chips, one_eyed_jacks, agent.clr))
    if two_eyed_jacks:
        print("has two eyed jack")
        print(get_two_eyed_pos(test_chips, two_eyed_jacks, agent.clr))



    print("-" * 10 + "Original Board" + "-" * 10)
    for each_line in test_chips:
        print(each_line)

