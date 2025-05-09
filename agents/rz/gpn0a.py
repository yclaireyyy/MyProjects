# -------------------------------- INFO --------------------------------
# Author:   Ruifan Zhang
# Purpose:  An Sequence AI Agent
# Method:   Three Step Analyse
# Details:
#   Select best position and best card

# -------------------------------- IMPORTS --------------------------------
import numpy as np
from copy import deepcopy
from collections import deque
from Sequence.sequence_model import *
from Sequence.sequence_utils import *

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
SCALE = 10
x = np.arange(10).reshape(-1, 1)
y = np.arange(10).reshape(1, -1)
z = (x - 4.5) ** 2 + (y - 4.5) ** 2
POSITION_WEIGHTS = np.exp(-SMOOTH * z)
POSITION_WEIGHTS *= SCALE

HEART_PRE_BIAS = 0
POSITION_WEIGHTS[HEART_POS] += HEART_PRE_BIAS

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
        (0, 1): (20, 0),

        (2, 0): (50, 0),
        (0, 2): (30, 0),

        (0, 3): (float('inf'), 100),
        (3, 0): (float('inf'), 0),

        (2, 1): (50, 20),
        (1, 2): (30, 50),
        (3, 1): (0, 200),
        (1, 3): (0, 100),
    }
    return table[(my, op)]


# -------------------------------- CLASSES --------------------------------
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
        pos_weight = np.zeros((3,10,10))
        combined = {
            0: {'place': np.zeros((10, 10), dtype=np.float32),
                'remove': np.zeros((10, 10), dtype=np.float32)},
            1: {'place': np.zeros((10, 10), dtype=np.float32),
                'remove': np.zeros((10, 10), dtype=np.float32)}
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
                        pos_weight[player+1][r][c] = 1
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
            combined[0]['place'] += POSITION_WEIGHTS*pos_weight[0]
            combined[1]['place'] += POSITION_WEIGHTS*pos_weight[0]
            combined[0]['remove'] += POSITION_WEIGHTS*pos_weight[1]
            combined[1]['remove'] += POSITION_WEIGHTS*pos_weight[2]
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
                # if r == 5 or b == 5:
                #     xxx = x_start
                #     yyy = y_start
                #     print(x_start,y_start,length,dx,dy,length,r,b)
                #     for ttt in range(length):
                #         print(chips[xxx+ttt*dx][yyy+ttt*dy], end=" ")
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


class myAgent:
    def __init__(self, _id):
        self.id = _id

    def SelectAction(self, actions, game_state:SequenceState):

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
            for (r,c) in positions:
                if (r,c) not in all_positions:
                    all_positions.append((r,c))
                    new_actions.append((each, (r,c), "place", my_value["place"][r][c]))
        for each in t_jacks:
            positions = get_two_eyed_pos(chips)
            for (r,c) in positions:
                if (r,c) not in all_positions:
                    all_positions.append((r,c))
                    new_actions.append((each, (r,c), "place", my_value["place"][r][c]))
        for each in o_jacks:
            positions = get_one_eyed_pos(chips, oc)
            for (r,c) in positions:
                if (r,c) not in all_positions:
                    all_positions.append((r,c))
                    new_actions.append((each, (r,c), "remove", my_value["remove"][r][c]))
        if d_t_jacks:
            d = d_t_jacks[0]
        elif d_o_jacks:
            d = d_o_jacks[0]
        else:
            dlist = []
            for each in d_normal:
                positions = get_normal_pos(chips, each)
                for (r,c) in positions:
                    dlist.append((each, my_value["place"][r][c]))
            dlist.sort(key=lambda x: x[1],reverse=True)
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


