# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Ruifan Zhang
# Date:    05/07/2021
# Purpose: Implements "Sequence" game agent

# IMPORTS ------------------------------------------------------------------------------------------------------------#

# import time
import random
# import numpy as np
from copy import deepcopy
from Sequence.sequence_utils import *
# from collections import defaultdict
from Sequence.sequence_model import COORDS

# CONSTANTS ----------------------------------------------------------------------------------------------------------#
THINKTIME = 0.99
PLACE = 0
BLOCK = 1
REMOVE = 2
REPLACE = 3
NUM_DIRECTIONS = 4
NUM_VALUES = 4
VR = 0
HZ = 1
D1 = 2
D2 = 3

# CLASS DEF ----------------------------------------------------------------------------------------------------------#

class myState:
    def __init__(self, chips: [list], _id: int):
        self.chips              = deepcopy(chips)
        self.id                 = _id
        self.my_turn            = True
        self.colour             = BLU if _id % 2 else RED
        self.opp_colour         = RED if _id % 2 else BLU
        self.seq_colour         = BLU_SEQ if _id % 2 else RED_SEQ
        self.opp_seq_colour     = RED_SEQ if _id % 2 else BLU_SEQ
        self.completed_seqs     = 0
        self.opp_completed_seqs = 0
        self.value              = None
        self.hand               = None
        self.opp_hand           = None
        self.draft              = None
        self.to_win             = {i:0 for i in range(1,5)}
        self.heart              = 0

    def next_turn(self):
        self.my_turn = not self.my_turn

    def get_colour(self, plr:int=None) -> (str,str,str,str):
        if (plr is None and self.my_turn) or (plr == self.id) :
            clr, sclr = self.colour, self.seq_colour
            oc, os = self.opp_colour, self.opp_seq_colour
        # 不指定玩家+非自己回合 or 指定对手
        else:
            clr, sclr = self.opp_colour, self.opp_seq_colour
            oc, os = self.colour, self.seq_colour
        return clr, sclr, oc, os

    def place(self, coords: (int, int), plr:int=None):
        debug = False
        r, c = coords
        clr, sclr, oc, os = self.get_colour(plr)
        if debug:
            print(clr, sclr, oc, os)
        self.chips[r][c] = clr

    def remove(self, coords):
        debug = False
        r, c = coords
        # 危险操作，需要前置检查是否可以移除
        self.chips[r][c] = EMPTY

    def update(self, coords: [(int, int)], plr):
        # 将形成Sequence的chips更新
        clr, sclr, oc, os = self.get_colour(plr)
        for r, c in coords:
            self.chips[r][c] = sclr

    # checkSeq Function
    # from sequence_model.py and has been MODIFIED
    def checkSeq(self, last_coords: (int, int), plr:int=None) -> (dict, int):
        debug = True
        clr, sclr, oc, os = self.get_colour(plr)
        if debug:
            print(clr, sclr, oc, os)
        seq_type = TRADSEQ
        seq_coords = []
        seq_found = {'vr': 0, 'hz': 0, 'd1': 0, 'd2': 0, 'hb': 0}
        found = False
        nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x
        lr, lc = last_coords

        # All joker spaces become player chips for the purposes of sequence checking.
        # for r, c in COORDS['jk']:
        #     self.chips[r][c] = clr

        # First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
        heart_chips = [self.chips[y][x] for x, y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (
                oc in heart_chips or os in heart_chips):
            seq_type = HOTBSEQ
            seq_found['hb'] += 2
            seq_coords.append(coord_list)

        # Search vertical, horizontal, and both diagonals.
        vr = [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        hz = [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        d1 = [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        d2 = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
        for seq, seq_name in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([self.chips[r][c] for r, c in coord_list])
            # Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                seq_found[seq_name] += 2
                seq_coords.append(coord_list)
            # If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                start_idx = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        start_idx = i + 1
                        sequence_len = 0
                    if sequence_len >= 5:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx + 5])
                        break
            else:  # Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr * 5, clr * 4 + sclr, clr * 3 + sclr + clr, clr * 2 + sclr + clr * 2,
                                clr + sclr + clr * 3, sclr + clr * 4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx + 5] == pattern:
                            seq_found[seq_name] += 1
                            seq_coords.append(coord_list[start_idx:start_idx + 5])
                            found = True
                            break
                    if found:
                        break

        # for r, c in COORDS['jk']:
        #     self.chips[r][c] = JOKER  # Joker spaces reset after sequence checking.

        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ
        return ({'num_seq': num_seq, 'orientation': [k for k, v in seq_found.items() if v], 'coords': seq_coords},
                seq_type) if num_seq else (None, None)

    def get_board_value(self, plr:int=None) -> (int, int):
        debug = False
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        value = [[[0 for _ in range(10)] for _ in range(10)] for _ in range(NUM_VALUES*NUM_DIRECTIONS)]
        self.value = [[[0 for _ in range(10)] for _ in range(10)] for _ in range(NUM_VALUES)]
        clr, sclr, oc, os = self.get_colour(plr)
        for r, c in COORDS['jk']:
            self.chips[r][c] = clr

        for i in range(10):
            for j in range(10):
                for idx,(dx,dy) in enumerate(directions):
                    dstx, dsty = i + 5*dx, j + 5*dy
                    if dstx < 0 or dstx > 9 or dsty < 0 or dsty > 9:
                        continue
                    window_idx = [(i+k*dx,j+k*dy) for k in range(5)]
                    window = [self.chips[r][c] for (r,c) in window_idx]
                    pv = self.place_value(window, plr)
                    rv = self.remove_value(window, plr)
                    for r,c in window_idx:
                        value[idx][r][c] = max(pv,value[idx][r][c])
                        value[REMOVE*NUM_DIRECTIONS+idx][r][c] = max(rv,value[REMOVE*NUM_DIRECTIONS+idx][r][c])
                    if debug:
                        print(window_idx, window, pv, rv)

        for r, c in COORDS['jk']:
            self.chips[r][c] = oc
        for i in range(10):
            for j in range(10):
                for idx,(dx,dy) in enumerate(directions):
                    dstx, dsty = i + 4*dx, j + 4*dy
                    if dstx < 0 or dstx > 9 or dsty < 0 or dsty > 9:
                        continue
                    window_idx = [(i+k*dx,j+k*dy) for k in range(5)]
                    window = [self.chips[r][c] for (r,c) in window_idx]
                    bv = self.block_value(window, plr)
                    rp = self.replace_value(window, plr)
                    for r,c in window_idx:
                        value[BLOCK*NUM_DIRECTIONS+idx][r][c] = max(bv,value[BLOCK*NUM_DIRECTIONS+idx][r][c])
                        value[REPLACE*NUM_DIRECTIONS+idx][r][c] = max(rp,value[REPLACE*NUM_DIRECTIONS+idx][r][c])
                    if debug:
                        print(window_idx, window, bv)
        for r, c in COORDS['jk']:
            self.chips[r][c] = JOKER
        for i in range(10):
            for j in range(10):
                for idx in range(4):
                    self.value[PLACE][i][j] += value[idx][i][j]
                    self.value[REMOVE][i][j] += value[REMOVE*NUM_DIRECTIONS+idx][i][j]
                    self.value[BLOCK][i][j] += value[BLOCK*NUM_DIRECTIONS+idx][i][j]


    def place_value(self, window:list, plr:int=None):
        debug = False
        # 窗口长度必须为5
        assert len(window) == 5
        clr, sclr, oc, os = self.get_colour(plr)
        # 该窗口无法作为潜在得分窗口
        # 5格窗口内存在对手棋子，无放置必要
        # 5格窗口内存在2枚及以上自己的Sequence Chips，无放置必要
        if oc in window or os in window or window.count(sclr) > 1:
            return 0
        # 在窗口内只有自己棋子的情况，返回窗口内自己棋子的数量
        return window.count(clr) + window.count(sclr)

    def replace_value(self, window:list, plr:int=None):
        debug = False
        # 窗口长度必须为5
        assert len(window) == 5
        clr, sclr, oc, os = self.get_colour(plr)
        # 该窗口无法作为潜在移除得分窗口
        # 5格窗口内存在对手Sequence Chips，无移除必要
        # 5格窗口内存在2枚及以上自己的Sequence Chips，无移除必要
        # 5格窗口内存在对手活棋大于2，无移除必要
        if os in window or window.count(sclr) > 1 or window.count(oc) > 1:
            return 0
        # 在窗口内最多一个对手活棋的情况，返回窗口内自己棋子的数量
        return window.count(clr) + window.count(sclr)


    # def remove_value(self, window:list, plr:int=None):
    #     debug = False
    #     # 窗口长度必须为5
    #     assert len(window) == 5
    #     clr, sclr, oc, os = self.get_colour(plr)
    #     # 该窗口无法内移除对手棋子无意义
    #     # 5格窗口内不存在对手活棋
    #     # 5格窗口内存在对手Sequence Chip
    #     # 5格窗口内存在2枚及以上对手活棋
    #     # 5格窗口内存在2枚及以上自己的Sequence Chips
    #     if oc not in window or os in window or window.count(oc) > 1 or window.count(sclr) > 1:
    #         return 0
    #     # 在窗口内只1枚对手棋子，有自己棋子的情况，返回窗口内自己棋子的数量
    #     return window.count(clr) + window.count(sclr)

    def remove_value(self, window:list, plr:int=None):
        debug = False
        # 窗口长度必须为5
        assert len(window) == 5
        clr, sclr, oc, os = self.get_colour(plr)
        # 该窗口内无法阻断对手棋子无意义
        # 5格窗口内不存在对手活棋
        # 5格窗口内存在自己棋子
        # 5格窗口内有2个及以上对手Sequence Chip
        if clr in window or sclr in window or oc not in window or window.count(os) > 1:
            return 0
        # 在窗口内只有对手棋子，返回窗口内对手棋子的数量
        return window.count(oc) + window.count(os)

    def block_value(self, window:list, plr:int=None):
        debug = False
        # 窗口长度必须为5
        assert len(window) == 5
        clr, sclr, oc, os = self.get_colour(plr)
        # 该窗口内存在自己的棋子则不需要阻止
        if clr in window or sclr in window:
            return 0
        # 在窗口内只只存在对手的棋子，则棋子越多，阻止价值越大
        return window.count(oc) + window.count(os)


    # action 包含坐标coords和类型type即可
    def next_state(self, action:dict, plr:int=None) -> "myState":
        debug = True
        next_state = deepcopy(self)
        coords = action.get('coords')
        act_type = action.get('type')
        play_card = action.get('play_card')
        draft_card = action.get('draft_card')
        if plr is None:
            executor = self.id if self.my_turn else 1-self.id
        else:
            executor = plr
        if play_card is not None:
            if executor == self.id:
                next_state.hand.remove(play_card)
            else:
                next_state.opp_hand.remove(play_card)
        if draft_card is not None:
            draft_card.remove(draft_card)
        if act_type == 'place':
            next_state.place(coords, executor)
            res = next_state.checkSeq(coords, executor)
            if debug:
                print(res)
            if res[0] is not None:
                for each in res[0].get('coords'):
                    next_state.update(each, executor)
                next_state.next_turn()
        elif act_type == 'remove':
            next_state.remove(coords)
            next_state.next_turn()
        return next_state



class myAgent:
    def __init__(self, _id):
        self.id = _id
        self.op_last = 0
        self.guesses = []
        self.seq_score = 0
        self.op_score = 0

        self.card_mapping = {
            '2c': [(1, 4), (3, 6)], '2d': [(2, 2), (5, 9)], '2h': [(5, 4), (8, 7)], '2s': [(0, 1), (8, 6)],
            '3c': [(1, 3), (3, 5)], '3d': [(2, 3), (6, 9)], '3h': [(5, 5), (8, 8)], '3s': [(0, 2), (8, 5)],
            '4c': [(1, 2), (3, 4)], '4d': [(2, 4), (7, 9)], '4h': [(4, 5), (7, 8)], '4s': [(0, 3), (8, 4)],
            '5c': [(1, 1), (3, 3)], '5d': [(2, 5), (8, 9)], '5h': [(4, 4), (6, 8)], '5s': [(0, 4), (8, 3)],
            '6c': [(1, 0), (3, 2)], '6d': [(2, 6), (9, 8)], '6h': [(4, 3), (5, 8)], '6s': [(0, 5), (8, 2)],
            '7c': [(2, 0), (4, 2)], '7d': [(2, 7), (9, 7)], '7h': [(4, 8), (5, 3)], '7s': [(0, 6), (8, 1)],
            '8c': [(3, 0), (5, 2)], '8d': [(3, 7), (9, 6)], '8h': [(3, 8), (6, 3)], '8s': [(0, 7), (7, 1)],
            '9c': [(4, 0), (6, 2)], '9d': [(4, 7), (9, 5)], '9h': [(2, 8), (6, 4)], '9s': [(0, 8), (6, 1)],
            'ac': [(7, 5), (8, 0)], 'ad': [(7, 6), (9, 1)], 'ah': [(1, 5), (4, 6)], 'as': [(2, 1), (4, 9)],
            'kc': [(7, 0), (7, 4)], 'kd': [(7, 7), (9, 2)], 'kh': [(1, 6), (5, 6)], 'ks': [(3, 1), (3, 9)],
            'qc': [(6, 0), (7, 3)], 'qd': [(6, 7), (9, 3)], 'qh': [(1, 7), (6, 6)], 'qs': [(2, 9), (4, 1)],
            'tc': [(5, 0), (7, 2)], 'td': [(5, 7), (9, 4)], 'th': [(1, 8), (6, 5)], 'ts': [(1, 9), (5, 1)]
        }

    def SelectAction(self, actions, game_state):
        return random.choice(actions)

    def guess_hand(self, game_state):
        op_trace = game_state.agents[1 - self.id].agent_trace.action_reward[self.op_last:]
        self.op_last = len(game_state.agents[1 - self.id].agent_trace.action_reward)
        p, d = self.extract(op_trace)
        for each in p:
            self.guesses.append(each)
        for each in d:
            if each in self.guesses:
                self.guesses.remove(each)

    def extract(self, trace: list):
        picked = []
        discarded = []
        for action, r in trace:
            if action["draft_card"] is not None:
                picked.append(action["draft_card"])
            if action["play_card"] is not None:
                discarded.append(action["play_card"])
        return picked, discarded


if __name__ == "__main__":
    test_chips = [[EMPTY for _ in range(10)] for _ in range(10)]
    for x,y in COORDS["jk"]:
        test_chips[x][y] = JOKER

    for x, y in [(4,i) for i in range(7) if i !=4]:
        test_chips[x][y] = RED
    ms = myState(test_chips, 0)
    print("-" * 10 + "Original Board" + "-" * 10)
    for each_line in ms.chips:
        print(each_line)

    ms.get_board_value(1)

    print("-" * 10 + "Place Value" + "-" * 10)
    for each_line in ms.value[PLACE]:
        print(each_line)

    print("-" * 10 + "Block Value" + "-" * 10)
    for each_line in ms.value[BLOCK]:
        print(each_line)

    print("-" * 10 + "Remove Value" + "-" * 10)
    for each_line in ms.value[REMOVE]:
        print(each_line)

    print("-" * 10 + "Replace Value" + "-" * 10)
    for each_line in ms.value[REPLACE]:
        print(each_line)

    # print("-" * 10 + "Original Board" + "-" * 10)
    # for each_line in ms.chips:
    #     print(each_line)
    #
    # next = ms.next_state({"type":"place","coords":(4,4)})
    # print("-" * 10 + "Next State" + "-" * 10)
    # for each_line in next.chips:
    #     print(each_line)


    # ms.place((5, 5))
    # print("-" * 10 + "After Place" + "-" * 10)
    # for each_line in ms.chips:
    #     print(each_line)
    #
    # print(ms.checkSeq((5, 5)))
    # print("-" * 10 + "After Check" + "-" * 10)
    # for each_line in ms.chips:
    #     print(each_line)
