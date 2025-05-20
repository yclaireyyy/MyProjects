# -------------------------------- INFO --------------------------------
# Author:   Ruifan Zhang
# Purpose:  An Sequence AI Agent
# Method:   Three Step Analyse
# Details:
#   Greedy select best position and best card

# -------------------------------- IMPORTS --------------------------------
import numpy as np
import time
import traceback
from copy import deepcopy
from collections import deque
from Sequence.sequence_model import *
from Sequence.sequence_utils import *
from agents.rz.search1 import THINKTIME

# -------------------------------- CONSTANTS --------------------------------
ONE_EYED_JACKS = ["js", "jh"]
TWO_EYED_JACKS = ["jc", "jd"]
JACKS = ONE_EYED_JACKS + TWO_EYED_JACKS
NORMAL_CARDS = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'q', 'k', 'a'] for s in
                ['d', 'c', 'h', 's']]

# (dx,dy), [(x_start,y_start,length)]
DIRECTIONS = [
    [(1, 0),
     [(0, 0, 10), (0, 1, 10), (0, 2, 10), (0, 3, 10), (0, 4, 10),
      (0, 5, 10), (0, 6, 10), (0, 7, 10), (0, 8, 10), (0, 9, 10)]],
    [(0, 1),
     [(0, 0, 10), (1, 0, 10), (2, 0, 10), (3, 0, 10), (4, 0, 10),
      (5, 0, 10), (6, 0, 10), (7, 0, 10), (8, 0, 10), (9, 0, 10)]],
    [(1, 1),
     [(5, 0, 5), (4, 0, 6), (3, 0, 7), (2, 0, 8), (1, 0, 9), (0, 0, 10),
      (0, 1, 9), (0, 2, 8), (0, 3, 7), (0, 4, 6), (0, 5, 5)]],
    [(1, -1),
     [(0, 4, 5), (0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 9), (0, 9, 10),
      (1, 9, 9), (2, 9, 8), (3, 9, 7), (4, 9, 6), (5, 9, 5)]],
]

HEART_POS = [(4, 4), (4, 5), (5, 4), (5, 5)]
USE_POSITION_WEIGHT = True
PLACE_REMOVE_SCALE = -0.2
PLACE_BIAS = 0.2
REMOVE_BIAS = 0.4
SMOOTH = 0.1
SCALE = 11
x = np.arange(10).reshape(-1, 1)
y = np.arange(10).reshape(1, -1)
z = (x - 4.5) ** 2 + (y - 4.5) ** 2
POSITION_WEIGHTS = np.exp(-SMOOTH * z)
POSITION_WEIGHTS *= SCALE

HEART_PRE_BIAS = 0.5
POSITION_WEIGHTS[HEART_POS] += HEART_PRE_BIAS
THINKTIME = 0.95

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


# Reconstruct actions
def advanced_actions(chips, normal, one_eyed_jacks, two_eyed_jacks, drafts, allow_trade, oc):
    need_trade = False
    used_normal = []
    used_o_jack = False
    used_t_jack = False
    normal_actions = []
    one_eyed_jacks_actions = []
    two_eyed_jacks_actions = []
    deads = []
    normal_positions = []
    for each in normal:
        if each in used_normal:
            continue
        else:
            used_normal.append(each)
        positions = get_normal_pos(chips, each)
        if not positions:
            need_trade = True
            deads.append(each)
        else:
            normal_positions.extend(positions)
            for position in positions:
                normal_actions.append((each, position, None))
    if need_trade and allow_trade:
        for d in drafts:
            if d in ONE_EYED_JACKS and not used_o_jack:
                used_o_jack = True
                positions = get_one_eyed_pos(chips, oc)
                for position in positions:
                    for dead in deads:
                        one_eyed_jacks_actions.append((d, position, dead))
            elif d in TWO_EYED_JACKS and not used_t_jack:
                used_t_jack = True
                positions = get_two_eyed_pos(chips)
                for position in positions:
                    for dead in deads:
                        two_eyed_jacks_actions.append((d, position, dead))
            elif d not in used_normal:
                used_normal.append(d)
                positions = get_normal_pos(chips, d)
                for position in positions:
                    for dead in deads:
                        normal_actions.append((d, position, dead))
    if one_eyed_jacks:
        o_jack = one_eyed_jacks[0]
        positions = get_one_eyed_pos(chips, oc)
        for position in positions:
            one_eyed_jacks_actions.append((o_jack, position, None))
    if two_eyed_jacks:
        t_jack = two_eyed_jacks[0]
        positions = get_two_eyed_pos(chips)
        for position in positions:
            # never place a jack into a position where your normal card could
            if position in normal_positions:
                continue
            two_eyed_jacks_actions.append((t_jack, position, None))
    return normal_actions, one_eyed_jacks_actions, two_eyed_jacks_actions


def exp_weight(values, ln):
    # print(values, ln)
    res = 0
    for v in values:
        if ln == 1 and v == 4:
            res = float("inf")
        res += 2.718 ** v
    return res


def heart_weight(my, op):
    # 硬编码所有 (my, op) → (place, remove)
    table = {
        (0, 0): (15, 0),
        (1, 1): (20, 10),
        (2, 2): (0, 30),

        (1, 0): (30, 0),
        (0, 1): (20, 10),

        (2, 0): (50, 0),
        (0, 2): (30, 20),
        (2, 1): (50, 20),
        (1, 2): (30, 50),

        (0, 3): (float('inf'), 100),
        (3, 0): (float('inf'), 0),
        (3, 1): (0, 200),
        (1, 3): (0, 100),
    }
    return table[(my, op)]


# -------------------------------- CLASSES --------------------------------
class myState:
    def __init__(self,
                 chips,
                 _id,
                 my_normal,
                 my_ojack,
                 my_tjack,
                 op_normal,
                 op_ojack,
                 op_tjack,
                 draft,
                 deck):
        self.chips = deepcopy(chips)
        self.id = _id
        self.my_turn = True
        self.allow_trade = True
        self.colour = BLU if _id % 2 else RED
        self.opp_colour = RED if _id % 2 else BLU
        self.seq_colour = BLU_SEQ if _id % 2 else RED_SEQ
        self.opp_seq_colour = RED_SEQ if _id % 2 else BLU_SEQ
        self.my_normal = list(my_normal)
        self.my_ojack = list(my_ojack)
        self.my_tjack = list(my_tjack)
        self.op_normal = list(op_normal)
        self.op_ojack = list(op_ojack)
        self.op_tjack = list(op_tjack)
        self.draft = deepcopy(draft)
        self.deck = deepcopy(deck)

    def next_turn(self):
        self.my_turn = not self.my_turn

    def disable_trade(self):
        self.allow_trade = False

    def get_colour(self, plr: int = None) -> (str, str, str, str):
        if (plr is None and self.my_turn) or (plr == self.id):
            clr, sclr = self.colour, self.seq_colour
            oc, os = self.opp_colour, self.opp_seq_colour
        # 不指定玩家+非自己回合 or 指定对手
        else:
            clr, sclr = self.opp_colour, self.opp_seq_colour
            oc, os = self.colour, self.seq_colour
        return clr, sclr, oc, os

    def place(self, coords: (int, int), plr: int = None):
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
    def checkSeq(self, last_coords: (int, int), plr: int = None) -> (dict, int):
        debug = False
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

    def trade(self, dead, new, plr:int=None):
        if plr is None:
            executor = self.id if self.my_turn else 1 - self.id
        else:
            executor = plr
        if executor == self.id:
            self.my_normal.remove(dead)
            if new in ONE_EYED_JACKS:
                self.my_ojack.append(new)
            elif new in TWO_EYED_JACKS:
                self.my_tjack.append(new)
            else:
                self.my_normal.append(new)
        else:
            self.op_normal.remove(dead)
            if new in ONE_EYED_JACKS:
                self.op_ojack.append(new)
            elif new in TWO_EYED_JACKS:
                self.op_tjack.append(new)
            else:
                self.op_normal.append(new)
        self.draft.remove(new)

    # action 包含坐标coords和类型type即可
    def apply_move(self, action: dict, plr: int = None):
        debug = False
        coords = action.get('coords')
        act_type = action.get('type')
        play_card = action.get('play_card')
        draft_card = action.get('draft_card')
        if plr is None:
            executor = self.id if self.my_turn else 1 - self.id
        else:
            executor = plr
        if play_card is not None:
            if executor == self.id:
                if play_card in ONE_EYED_JACKS:
                    self.my_ojack.remove(play_card)
                elif play_card in TWO_EYED_JACKS:
                    self.my_tjack.remove(play_card)
                else:
                    self.my_normal.remove(play_card)
            else:
                if play_card in ONE_EYED_JACKS:
                    self.op_ojack.remove(play_card)
                elif play_card in TWO_EYED_JACKS:
                    self.op_tjack.remove(play_card)
                else:
                    self.op_normal.remove(play_card)
        if draft_card is not None:
            self.draft.remove(draft_card)
        if act_type == 'place':
            self.place(coords, executor)
            res = self.checkSeq(coords, executor)
            if debug:
                print(res)
            if res[0] is not None:
                for each in res[0].get('coords'):
                    self.update(each, executor)
        elif act_type == 'remove':
            self.remove(coords)
        self.next_turn()

# 用于评估 Sequence 棋盘中每个位置在不同方向下的行动价值：
# 包括放置、阻止、移除和取代价值，分别从红方（0）和蓝方（1）视角评估。
class BoardEvaluator:
    # 将方向向量 (dx, dy) 映射为索引值：
    # 0: (1, 0)  -> 横向
    # 1: (0, 1)  -> 纵向
    # 2: (1, 1)  -> 主对角线
    # 3: (1, -1) -> 副对角线
    @staticmethod
    def direction_index(dx, dy):
        if (dx, dy) == (1, 0): return 0
        if (dx, dy) == (0, 1): return 1
        if (dx, dy) == (1, 1): return 2
        if (dx, dy) == (1, -1): return 3
        return -1

    # 将分方向的原始动作值整合为放置类（空格）和移除类（one-eyed jack）两种整合价值。
    #
    # 参数:
    #     values: dict[side][action][dir][r][c]
    #     chips: 10x10 当前棋盘
    #     win_threshold: 连子数阈值
    #     win_value / block_value: 用于关键点打分
    #     weight_fn: 用于组合四方向四元组的函数，如 np.max, np.sum, sorted_sum
    #
    # 返回:
    #     {
    #         0: {'place': 10x10 array, 'remove': 10x10 array},
    #         1: {'place': ...,         'remove': ...}
    #     }
    @staticmethod
    def combine_value(chips):
        values = BoardEvaluator.evaluate_locations(chips)
        line_values, (red_heart, blue_heart) = BoardEvaluator.evaluate_board(chips)
        # print(line_values)
        seq = (line_values[0][-1], line_values[1][-1])
        weight_fn = exp_weight
        pos_weight = np.zeros((3, 10, 10))
        combined = {
            0: {'place': np.zeros((10, 10), dtype=np.float64),
                'remove': np.zeros((10, 10), dtype=np.float64)},
            1: {'place': np.zeros((10, 10), dtype=np.float64),
                'remove': np.zeros((10, 10), dtype=np.float64)}
        }
        for player in [0, 1]:
            for r in range(10):
                for c in range(10):
                    cell = chips[r][c]
                    # ----- 放置类：空格 -----
                    if cell == EMPTY:
                        pos_weight[0][r][c] = 1
                        place_4 = values[player]['place'][:, r, c]
                        block_4 = values[player]['block'][:, r, c]
                        place_val = weight_fn(place_4, seq[player])
                        block_val = weight_fn(block_4, seq[1 - player])
                        total = (1 + PLACE_BIAS) * place_val + (1 - PLACE_BIAS) * block_val
                        total *= (1 + PLACE_REMOVE_SCALE)
                        combined[player]['place'][r][c] = total

                    # ----- 移除类：对方活子 -----
                    elif ((player == 0 and cell == BLU) or (player == 1 and cell == RED)):
                        pos_weight[player + 1][r][c] = 1
                        remove_4 = values[player]['removal'][:, r, c]
                        override_4 = values[player]['override'][:, r, c]
                        remove_val = weight_fn(remove_4, seq[1 - player])
                        override_val = weight_fn(override_4, seq[player])
                        total = (1 + REMOVE_BIAS) * remove_val + (1 - REMOVE_BIAS) * override_val
                        total *= (1 - PLACE_REMOVE_SCALE)
                        combined[player]['remove'][r][c] = total

        place_heart_red, remove_heart_red = heart_weight(red_heart, blue_heart)
        place_heart_blue, remove_heart_blue = heart_weight(blue_heart, red_heart)
        # print(place_heart_red, place_heart_blue, remove_heart_red, remove_heart_blue)
        for x, y in HEART_POS:
            if chips[x][y] == EMPTY:
                # print(x, y, chips[r][c], combined[0]['place'][x][y])
                combined[0]['place'][x][y] = max(combined[0]['place'][x][y], place_heart_red)
                combined[1]['place'][x][y] = max(combined[1]['place'][x][y], place_heart_blue)
            elif chips[x][y] == BLU:
                combined[0]['remove'][x][y] = max(combined[0]['remove'][x][y], remove_heart_red)
            elif chips[x][y] == RED:
                combined[1]['remove'][x][y] = max(combined[1]['remove'][x][y], remove_heart_blue)
        if USE_POSITION_WEIGHT:
            combined[0]['place'] += POSITION_WEIGHTS * pos_weight[0]
            combined[1]['place'] += POSITION_WEIGHTS * pos_weight[0]
            combined[0]['remove'] += POSITION_WEIGHTS * pos_weight[1]
            combined[1]['remove'] += POSITION_WEIGHTS * pos_weight[2]
        return combined

    @staticmethod
    def evaluate_board(chips):
        values = {
            0: [0 for _ in range(6)],
            1: [0 for _ in range(6)]
        }
        for (dx, dy), starts in DIRECTIONS:
            for x_start, y_start, length in starts:
                r, b = BoardEvaluator.evaluate_line_max_streak(chips, x_start, y_start, length, dx, dy)
                values[0][r] += 1
                values[1][b] += 1
        heart_value = BoardEvaluator.evaluate_heart(chips)
        return values, heart_value

    # 滑动窗口动态统计一条线段上 RED / BLUE 的最大连子数（合法窗口内）。
    #
    # 合法窗口定义：
    #     - 窗口内对方棋子（活子 + 死子）为 0
    #     - 不再限制己方死子数量（因为每个 sequence 只会留下一个）
    #
    # 参数:
    #     chips: 10x10 棋盘（每格为字符）
    #     x_start, y_start: 起始坐标
    #     length: 线段长度
    #     dx, dy: 方向增量
    #
    # 返回:
    #     (max_red, max_blue): 双方在该线段中的最大连子数
    @staticmethod
    def evaluate_line_max_streak(chips, x_start, y_start, length, dx, dy):
        counts = {
            RED: 0, RED_SEQ: 0,
            BLU: 0, BLU_SEQ: 0,
            JOKER: 0, EMPTY: 0
        }

        pos_queue = deque()
        max_red = 0
        max_blue = 0

        # 初始化前5个窗口
        for i in range(5):
            x = x_start + i * dx
            y = y_start + i * dy
            c = chips[x][y]
            counts[c] += 1
            pos_queue.append((x, y))

        left = 0
        right = 5

        while True:
            # RED 评估窗口
            if counts[BLU] == 0 and counts[BLU_SEQ] == 0:
                red_streak = counts[RED] + counts[RED_SEQ] + counts[JOKER]
                max_red = max(max_red, red_streak)

            # BLUE 评估窗口
            if counts[RED] == 0 and counts[RED_SEQ] == 0:
                blue_streak = counts[BLU] + counts[BLU_SEQ] + counts[JOKER]
                max_blue = max(max_blue, blue_streak)

            if right >= length:
                break

            # 滑出窗口头
            old_x = x_start + left * dx
            old_y = y_start + left * dy
            old = chips[old_x][old_y]
            counts[old] -= 1
            pos_queue.popleft()
            left += 1

            # 滑入窗口尾
            new_x = x_start + right * dx
            new_y = y_start + right * dy
            new = chips[new_x][new_y]
            counts[new] += 1
            pos_queue.append((new_x, new_y))
            right += 1

        return max_red, max_blue

    # 遍历棋盘所有可评估的行、列、对角线。
    # 对每条线调用
    # evaluate_line_to_board
    # 并累积结果。
    #
    # 返回:
    #   values: dict[int -> dict[str -> np.ndarray]]
    #   values[0]为红方视角下的价值，values[1]为蓝方。
    #   每种价值为一个形状为(4, 10, 10)的数组，对应4个方向。
    @staticmethod
    def evaluate_locations(chips):
        values = {
            # RED
            0: {
                "place": np.zeros((4, 10, 10), dtype=np.int8),
                "block": np.zeros((4, 10, 10), dtype=np.int8),
                "removal": np.zeros((4, 10, 10), dtype=np.int8),
                "override": np.zeros((4, 10, 10), dtype=np.int8)
            },
            # BLU
            1: {
                "place": np.zeros((4, 10, 10), dtype=np.int8),
                "block": np.zeros((4, 10, 10), dtype=np.int8),
                "removal": np.zeros((4, 10, 10), dtype=np.int8),
                "override": np.zeros((4, 10, 10), dtype=np.int8)
            }
        }

        for (dx, dy), starts in DIRECTIONS:
            for x_start, y_start, length in starts:
                BoardEvaluator.evaluate_lines(chips, values, x_start, y_start, length, dx, dy)
        # heart_value = BoardEvaluator.evaluate_heart(chips)

        return values

    # 评估从 (x_start, y_start) 出发，方向为 (dx, dy)，长度为 length 的一条线段。
    # 使用滑动窗口法计算每个 5 格窗口中的 RED/BLUE 双方的放置、阻止、移除和取代价值，
    # 并将最大值写入 values 的相应位置。
    #
    # 参数:
    #   chips: 10x10 棋盘
    #   values: 双方价值表，结构为 values[side][action][direction][r][c]
    #   x_start, y_start: 起始坐标
    #   length: 线段长度（必须 ≥ 5）
    #   dx, dy: 方向向量
    @staticmethod
    def evaluate_lines(chips, values, x_start, y_start, length, dx, dy):
        if length < 5:
            return
        dir_idx = BoardEvaluator.direction_index(dx, dy)
        counts = {
            RED: 0, RED_SEQ: 0,
            BLU: 0, BLU_SEQ: 0,
            JOKER: 0, EMPTY: 0
        }
        pos_queue = deque()

        # 初始化窗口
        for i in range(5):
            x = x_start + i * dx
            y = y_start + i * dy
            c = chips[x][y]

            counts[c] += 1
            pos_queue.append((x, y))

        left = 0
        right = 5

        while True:
            # RED 方
            red_idx = 0
            # 该窗口无法作为潜在得分窗口
            # 5格窗口内存在对手棋子，无放置必要
            # 5格窗口内存在2枚及以上自己的Sequence Chips，无放置必要
            if counts[BLU] == 0 and counts[BLU_SEQ] == 0 and counts[RED_SEQ] < 2:
                red_place = counts[RED] + counts[RED_SEQ] + counts[JOKER]
            else:
                red_place = 0

            # 该窗口内存在自己的棋子则不需要阻止
            # 该窗口内存在2枚及以上对手的Sequence Chips则不需要阻止
            if counts[RED] == 0 and counts[RED_SEQ] == 0 and counts[BLU_SEQ] < 2:
                red_block = counts[BLU] + counts[BLU_SEQ] + counts[JOKER]
            else:
                red_block = 0

            # 该窗口内无法阻断对手棋子无意义
            # 5格窗口内不存在对手活棋
            # 5格窗口内存在自己棋子
            # 5格窗口内有2个及以上对手Sequence Chip
            if counts[RED] == 0 and counts[RED_SEQ] == 0 and counts[BLU_SEQ] < 2:
                red_removal = counts[BLU] + counts[BLU_SEQ] + counts[JOKER]
            else:
                red_removal = 0

            # 该窗口无法作为潜在移除得分窗口
            # 5格窗口内存在对手Sequence Chips，无移除必要
            # 5格窗口内存在2枚及以上自己的Sequence Chips，无移除必要
            # 5格窗口内存在对手活棋大于2，无移除必要
            if counts[BLU] < 2 and counts[BLU_SEQ] == 0 and counts[RED_SEQ] < 2:
                red_override = counts[RED] + counts[RED_SEQ] + counts[JOKER] + 1
            else:
                red_override = 0

            # BLU 方
            blu_idx = 1
            # 该窗口无法作为潜在得分窗口
            # 5格窗口内存在对手棋子，无放置必要
            # 5格窗口内存在2枚及以上自己的Sequence Chips，无放置必要
            if counts[RED] == 0 and counts[RED_SEQ] == 0 and counts[BLU_SEQ] < 2:
                blu_place = counts[BLU] + counts[BLU_SEQ] + counts[JOKER]
            else:
                blu_place = 0

            # 该窗口内存在自己的棋子则不需要阻止
            # 该窗口内存在2枚及以上对手的Sequence Chips则不需要阻止
            if counts[BLU] == 0 and counts[BLU_SEQ] == 0 and counts[RED_SEQ] < 2:
                blu_block = counts[RED] + counts[RED_SEQ] + counts[JOKER]
            else:
                blu_block = 0

            # 该窗口内无法阻断对手棋子无意义
            # 5格窗口内不存在对手活棋
            # 5格窗口内存在自己棋子
            # 5格窗口内有2个及以上对手Sequence Chip
            if counts[BLU] == 0 and counts[BLU_SEQ] == 0 and counts[RED_SEQ] < 2:
                blu_removal = counts[RED] + counts[RED_SEQ] + counts[JOKER] + 1
            else:
                blu_removal = 0

            # 该窗口无法作为潜在移除得分窗口
            # 5格窗口内存在对手Sequence Chips，无移除必要
            # 5格窗口内存在2枚及以上自己的Sequence Chips，无移除必要
            # 5格窗口内存在对手活棋大于2，无移除必要
            if counts[RED] <= 1 and counts[RED_SEQ] == 0 and counts[BLU_SEQ] < 2:
                blu_override = counts[BLU] + counts[BLU_SEQ] + counts[JOKER] + 1
            else:
                blu_override = 0

            # 更新 value 矩阵
            for x, y in pos_queue:
                chip = chips[x][y]
                if chip == EMPTY:
                    values[red_idx]['place'][dir_idx][x][y] = max(
                        values[red_idx]['place'][dir_idx][x][y],
                        red_place
                    )
                    values[red_idx]['block'][dir_idx][x][y] = max(
                        values[red_idx]['block'][dir_idx][x][y],
                        red_block
                    )
                    values[blu_idx]['place'][dir_idx][x][y] = max(
                        values[blu_idx]['place'][dir_idx][x][y],
                        blu_place
                    )
                    values[blu_idx]['block'][dir_idx][x][y] = max(
                        values[blu_idx]['block'][dir_idx][x][y],
                        blu_block
                    )
                elif chip == BLU:
                    values[red_idx]['removal'][dir_idx][x][y] = max(
                        values[red_idx]['removal'][dir_idx][x][y],
                        red_removal
                    )
                    values[red_idx]['override'][dir_idx][x][y] = max(
                        values[red_idx]['override'][dir_idx][x][y],
                        red_override
                    )
                elif chip == RED:
                    values[blu_idx]['removal'][dir_idx][x][y] = max(
                        values[blu_idx]['removal'][dir_idx][x][y],
                        blu_removal
                    )
                    values[blu_idx]['override'][dir_idx][x][y] = max(
                        values[blu_idx]['override'][dir_idx][x][y],
                        blu_override
                    )
            if right >= length:
                break

            # 滑动窗口
            old_x = x_start + left * dx
            old_y = y_start + left * dy
            old = chips[old_x][old_y]
            counts[old] -= 1
            pos_queue.popleft()
            left += 1

            new_x = x_start + right * dx
            new_y = y_start + right * dy
            new = chips[new_x][new_y]
            counts[new] += 1
            pos_queue.append((new_x, new_y))
            right += 1

    # 分析 HEART 区域当前局势，返回当前局面对胜负产生关键影响的控制值：
    # - 若 RED 完全无子，则 BLUE 控制数越多，RED 越需要阻止（返回蓝方控制数）
    # - 若 BLUE 完全无子，则 RED 控制数越多，RED 越接近胜利（返回红方控制数）
    # - 若双方都有子，互相阻断，则无法构成 HEART 胜利 → 返回 0
    @staticmethod
    def evaluate_heart(chips):
        counts = {RED: 0, RED_SEQ: 0, BLU: 0, BLU_SEQ: 0, EMPTY: 0}
        for r, c in HEART_POS:
            counts[chips[r][c]] += 1

        red_total = counts[RED] + counts[RED_SEQ]
        blu_total = counts[BLU] + counts[BLU_SEQ]

        return (red_total, blu_total)

    # 评估当前局面下，双方（player_id
    # 0 / 1）对所有
    # 52
    # 张牌在
    # place
    # 和
    # remove
    # 行动下的最大落子价值。
    #
    # 参数:
    # value: dict[player_id][action][r][c]  # eg: value[0]['place'][r][c]
    # COORD: dict[card] -> list[(r, c)]  # eg: COORD['3♠'] = [(r1,c1), ...]
    #
    #
    # 返回:
    # card_values: dict[player_id][action][card] -> int
    @staticmethod
    def evaluate_cards(values):
        card_values = {0: {}, 1: {}}

        for pid in [0, 1]:
            place_mat = values[pid]["place"]
            remove_mat = values[pid]["remove"]

            for card in NORMAL_CARDS:
                coords = COORDS.get(card, [])
                card_values[pid][card] = max(
                    (values[pid][x][y] for x, y in coords),
                    default=0
                )

            remove_max = np.max(remove_mat)
            place_max = np.max(place_mat)

            for card in ONE_EYED_JACKS:
                card_values[pid][card] = remove_max

            for card in TWO_EYED_JACKS:
                card_values[pid][card] = place_max

        return card_values


class Node:
    def __init__(self,
                 state: myState,
                 parent=None,
                 action=None,
                 ):
        self.state = state
        self.parent = parent
        self.action = action
        self.value = -float('inf')
        self.force_pick = None
        self.pick = None
        self.actions = []
        self.children = []
        self.pos_value = None

    def get_best(self):
        self.children.sort(key=lambda x: x.value, reverse=True)
        best_child = self.children[0]
        t,p,(r,c),d = best_child.action
        if d is not None:
            action = {
                "type":"trade",
                "play_card":p,
                "coords":(None,None),
                "draft_card":self.pick
            }
        else:
            action = {
                "type":t,
                "play_card":p,
                "coords":(r,c),
                "draft_card":self.pick
            }
        return action

    def get_value(self):
        self.pos_value = BoardEvaluator.combine_value(self.state.chips)

    def evaluate(self):
        self.get_value()
        chips = self.state.chips
        oc = self.state.opp_colour
        value_map = self.pos_value[self.state.id]
        values = []
        for each in self.state.my_normal:
            positions = get_normal_pos(chips, each)
            for r, c in positions:
                values.append(value_map["place"][r][c])
        if self.state.my_ojack:
            positions = get_one_eyed_pos(chips, oc)
            for r, c in positions:
                values.append(value_map["remove"][r][c])
        if self.state.my_tjack:
            positions = get_two_eyed_pos(chips)
            for r, c in positions:
                values.append(value_map["place"][r][c])
        values.sort(reverse=True)
        self.value = values[0]
        draft_t_jacks = []
        draft_o_jacks = []
        draft_normal = []
        pick_list = []
        # print("self.state.draft",self.state.draft)
        for d in self.state.draft:
            if d in TWO_EYED_JACKS:
                draft_t_jacks.append(d)
            elif d in ONE_EYED_JACKS:
                draft_o_jacks.append(d)
            else:
                draft_normal.append(d)
        if draft_t_jacks:
            self.pick = draft_t_jacks[0]
        else:
            if draft_o_jacks:
                positions = get_one_eyed_pos(chips, oc)
                for r, c in positions:
                    v = value_map["remove"][r][c]
                    pick_list.append((v,draft_o_jacks[0]))
            for each in draft_normal:
                positions = get_normal_pos(chips, each)
                for r, c in positions:
                    v = value_map["place"][r][c]
                    pick_list.append((v,each))
            pick_list.sort(key=lambda x: x[0], reverse=True)
            self.pick = pick_list[0][1]


    def get_actions(self):
        self.actions = []
        self.get_value()
        chips = self.state.chips
        draft = self.state.draft
        allow_trade = self.state.allow_trade
        if self.state.my_turn:
            normal = self.state.my_normal
            o_jacks = self.state.my_ojack
            t_jacks = self.state.my_tjack
            oc = self.state.opp_colour
            value = self.pos_value[self.state.id]
        else:
            normal = self.state.op_normal
            o_jacks = self.state.op_ojack
            t_jacks = self.state.op_tjack
            oc = self.state.colour
            value = self.pos_value[1-self.state.id]
        normal_actions, ojack_actions, tjack_actions = advanced_actions(
            chips,
            normal,
            o_jacks,
            t_jacks,
            draft,
            allow_trade,
            oc
        )
        # print(normal_actions)
        for (card,(r,c),dead) in normal_actions:
            v = float(value["place"][r,c])
            self.actions.append((v,"place",card,(r,c),dead))
        for (card,(r,c),dead) in ojack_actions:
            v = float(value["remove"][r,c])
            self.actions.append((v,"remove",card,(r,c),dead))
        for (card,(r,c),dead) in tjack_actions:
            v = float(value["place"][r,c])
            self.actions.append((v,"place",card,(r,c),dead))
        self.actions.sort(key=lambda a: a[0], reverse=True)


    def expand(self):
        self.get_actions()
        for v,type,p,(r,c),d in self.actions:
            new_state = deepcopy(self.state)
            if d is not None:
                new_state.trade(d,p)
            a = {"type":type,
                "play_card":p,
                "coords":(r,c)}
            new_state.apply_move(a)
            child = Node(
                new_state,
                self,
                (type,p,(r,c),d),
            )
            child.force_pick = d
            self.children.append(child)

    def back_propagation(self):
        node = self
        while node.parent is not None:
            if node.parent.state.my_turn:
                if node.parent.value < self.value:
                    node.parent.value = self.value
                    if self.force_pick:
                        node.parent.force_pick = self.force_pick
                        node.parent.pick = self.force_pick
                    else:
                        node.parent.pick = self.pick
            else:
                if node.parent.value < self.value:
                    node.parent.value = self.value
                    if self.force_pick:
                        node.parent.force_pick = self.force_pick
                    else:
                        node.parent.pick = self.pick
            node = node.parent


# class SearchTree:
#     def __init__(self, root):


class myAgent:
    def __init__(self, _id):
        self.id = _id
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
        self.op_trace_last = None
        self.op_hand = []
        self.op_normal = []
        self.op_dead = []
        self.op_one_eyed_jacks = []
        self.op_two_eyed_jacks = []

    def SelectAction(self, actions, game_state):
        try:
            a = self.SelectAction_db(actions, game_state)
        except Exception as e:
            print("⚠️ Exception occurred during SelectAction_db:")
            traceback.print_exc()  # 打印完整 traceback
            a = random.choice(actions)
        return a

    def SelectAction_db(self, actions, game_state: SequenceState):
        start_time = time.time()
        self.extract_info(game_state)
        root_state = myState(
            self.chips,
            self.id,
            self.my_normal,
            self.my_one_eyed_jacks,
            self.my_two_eyed_jacks,
            self.op_normal,
            self.op_one_eyed_jacks,
            self.op_two_eyed_jacks,
            self.draft,
            self.deck
        )
        node = Node(root_state)
        print("self.op_hand: ", self.op_hand)
        print("self.op_normal: ", self.op_normal)
        print("self.op_one_eyed_jacks: ", self.op_one_eyed_jacks)
        print("self.op_two_eyed_jacks: ", self.op_two_eyed_jacks)
        node.expand()
        for c in node.children:
            if time.time() - start_time > THINKTIME:
                break
            if self.op_hand:
                c.expand()
                for cc in c.children:
                    cc.evaluate()
                    cc.back_propagation()
            else:
                c.evaluate()
                c.back_propagation()
        a = node.get_best()
        if a not in actions:
            print(actions)
            a = random.choice(actions)
        print("action:", a)
        return a

    def extract_info(self, game_state: SequenceState):
        # At Start Remove cards in my hand from deck
        if self.op_trace_last is None:
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

    def extract_my_info(self, game_state: SequenceState):
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

    def extract_opp_info(self, game_state: SequenceState):
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
        self.op_normal = []
        self.op_one_eyed_jacks = []
        self.op_two_eyed_jacks = []
        for each in self.op_hand:
            if each in ONE_EYED_JACKS:
                self.op_one_eyed_jacks.append(each)
            elif each in TWO_EYED_JACKS:
                self.op_two_eyed_jacks.append(each)
            else:
                self.op_normal.append(each)

    def SelectActionOld(self, actions, game_state: SequenceState):
        myself = game_state.agents[self.id]
        hand_cards = myself.hand
        board = game_state.board
        clr = myself.colour
        sclr = myself.seq_colour
        oc = myself.opp_colour
        os = myself.opp_seq_colour
        normal = []
        o_jacks = []
        t_jacks = []
        d_normal = []
        d_o_jacks = []
        d_t_jacks = []
        all_positions = []
        new_actions = []
        chips = board.chips
        drafts = board.draft
        values = BoardEvaluator.combine_value(chips)
        my_value = values[self.id]
        for each in hand_cards:
            if each in ONE_EYED_JACKS:
                o_jacks.append(each)
            elif each in TWO_EYED_JACKS:
                t_jacks.append(each)
            else:
                normal.append(each)

        for each in drafts:
            if each in ONE_EYED_JACKS:
                d_o_jacks.append(each)
            elif each in TWO_EYED_JACKS:
                d_t_jacks.append(each)
            else:
                d_normal.append(each)

        for each in normal:
            positions = get_normal_pos(chips, each)
            for (r, c) in positions:
                if (r, c) not in all_positions:
                    all_positions.append((r, c))
                    new_actions.append((each, (r, c), "place", my_value["place"][r][c]))
        for each in t_jacks:
            positions = get_two_eyed_pos(chips)
            for (r, c) in positions:
                if (r, c) not in all_positions:
                    all_positions.append((r, c))
                    new_actions.append((each, (r, c), "place", my_value["place"][r][c]))
        for each in o_jacks:
            positions = get_one_eyed_pos(chips, oc)
            for (r, c) in positions:
                if (r, c) not in all_positions:
                    all_positions.append((r, c))
                    new_actions.append((each, (r, c), "remove", my_value["remove"][r][c]))
        if d_t_jacks:
            d = d_t_jacks[0]
        elif d_o_jacks:
            d = d_o_jacks[0]
        else:
            dlist = []
            for each in d_normal:
                positions = get_normal_pos(chips, each)
                for (r, c) in positions:
                    dlist.append((each, my_value["place"][r][c]))
            dlist.sort(key=lambda x: x[1], reverse=True)
            d = dlist[0][0]
        if actions[0].get("type") == "trade":
            action = deepcopy(actions[0])
            action["draft_card"] = d
        else:
            new_actions.sort(key=lambda a: a[3], reverse=True)
            action = {
                "type": new_actions[0][2],
                "play_card": new_actions[0][0],
                "draft_card": d,
                "coords": new_actions[0][1],
            }
        return action


if __name__ == "__main__":
    board = [[EMPTY for _ in range(10)]for _ in range(10)]
    for r,c in COORDS["jk"]:
        board[r][c] = JOKER
    board[3][0] = RED
    board[5][2] = RED

    deck = [
        (r + s)
        for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a']
        for s in ['d', 'c', 'h', 's']
    ]
    deck = deck * 2
    random.seed(2357)
    random.shuffle(deck)
    my_hand = deck[:6]
    op_hand = deck[6:12]
    draft = deck[12:17]
    deck = deck[17:]

    my_normal = []
    op_normal = []
    my_ojacks = []
    my_tjacks = []
    op_ojacks = []
    op_tjacks = []

    for each in my_hand:
        if each in ONE_EYED_JACKS:
            my_ojacks.append(each)
        elif each in TWO_EYED_JACKS:
            my_tjacks.append(each)
        else:
            my_normal.append(each)

    for each in op_hand:
        if each in ONE_EYED_JACKS:
            op_ojacks.append(each)
        elif each in TWO_EYED_JACKS:
            op_tjacks.append(each)
        else:
            op_normal.append(each)

    my_normal = ['8c', '8c', 'qs', '6d', '4h', '9d']
    draft = ['8c', '2h', 'qs', '7h', '6c']
    print(my_normal)
    print(my_ojacks)
    print(my_tjacks)
    print(op_normal)
    print(op_ojacks)
    print(op_tjacks)
    print(draft)
    state = myState(
        board,
        0,
        my_normal,
        my_ojacks,
        my_tjacks,
        op_normal,
        op_ojacks,
        op_tjacks,
        draft,
        deck
    )
    node = Node(state)
    node.expand()
    for c in node.children:
        c.expand()
        for cc in c.children:
            cc.evaluate()
            cc.back_propagation()
    print(node.value)