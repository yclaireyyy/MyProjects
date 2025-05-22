from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import random
import time
import copy
import math
import itertools
import numpy as np
from collections import deque

# Constants
MAX_THINK_TIME = 0.95  # 最大思考时间（秒）
EXPLORATION_WEIGHT = 1.4  # UCB公式中的探索参数
HEART_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]  # 中心热点位置
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]  # 角落位置（自由点）
SIMULATION_LIMIT = 100  # MCTS模拟的最大次数

# Card constants
ONE_EYED_JACKS = ["js", "jh"]  # Used to remove opponent's chips
TWO_EYED_JACKS = ["jc", "jd"]  # Can be placed anywhere
JACKS = ONE_EYED_JACKS + TWO_EYED_JACKS

# Board constants
EMPTY = 0
RED = 'r'
BLU = 'b'
RED_SEQ = 'R'  # Red sequence
BLU_SEQ = 'B'  # Blue sequence
JOKER = 'jk'   # Corner spaces

# Direction vectors for sequences
DIRECTIONS = [
    [(1, 0),  # Horizontal
     [(0, 0, 10), (0, 1, 10), (0, 2, 10), (0, 3, 10), (0, 4, 10),
      (0, 5, 10), (0, 6, 10), (0, 7, 10), (0, 8, 10), (0, 9, 10)]],
    [(0, 1),  # Vertical
     [(0, 0, 10), (1, 0, 10), (2, 0, 10), (3, 0, 10), (4, 0, 10),
      (5, 0, 10), (6, 0, 10), (7, 0, 10), (8, 0, 10), (9, 0, 10)]],
    [(1, 1),  # Main diagonal
     [(5, 0, 5), (4, 0, 6), (3, 0, 7), (2, 0, 8), (1, 0, 9), (0, 0, 10),
      (0, 1, 9), (0, 2, 8), (0, 3, 7), (0, 4, 6), (0, 5, 5)]],
    [(1, -1),  # Anti-diagonal
     [(0, 4, 5), (0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 9), (0, 9, 10),
      (1, 9, 9), (2, 9, 8), (3, 9, 7), (4, 9, 6), (5, 9, 5)]],
]

# Position weights for strategic locations
USE_POSITION_WEIGHT = True
PLACE_REMOVE_SCALE = -0.2
OPPONENT_SCALE = 0.1
PLACE_BIAS = 0.2
REMOVE_BIAS = 0.4
SMOOTHING = 0.1
SCALE = 11

# Generate position weights based on distance from center
x = np.arange(10).reshape(-1, 1)
y = np.arange(10).reshape(1, -1)
z = (x - 4.5) ** 2 + (y - 4.5) ** 2
POSITION_WEIGHTS = np.exp(-SMOOTHING * z)
POSITION_WEIGHTS *= SCALE

# Add extra bias for heart positions
HEART_PRE_BIAS = 0.5
for x, y in HEART_COORDS:
    POSITION_WEIGHTS[x][y] += HEART_PRE_BIAS

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
        pos_weight = np.zeros((3, 10, 10))
        combined = {
            0: {'place': np.zeros((10, 10), dtype=np.float32),
                'remove': np.zeros((10, 10), dtype=np.float32)},
            1: {'place': np.zeros((10, 10), dtype=np.float32),
                'remove': np.zeros((10, 10), dtype=np.float32)}
        }

        def exp_weight(values, ln):
            """Exponential weighting of values."""
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
            return table.get((my, op), (10, 5))

        # 计算每个玩家的位置价值
        for player in [0, 1]:
            for r in range(10):
                for c in range(10):
                    cell = chips[r][c]
                    # ----- 放置类：空格 -----
                    if cell == EMPTY:
                        pos_weight[0][r][c] = 1
                        place_4 = values[player]['place'][:, r, c]
                        block_4 = values[player]['block'][:, r, c]
                        place_val = exp_weight(place_4, seq[player])
                        block_val = exp_weight(block_4, seq[1 - player])
                        total = (1 + PLACE_BIAS) * place_val + (1 - PLACE_BIAS) * block_val
                        total *= (1 + PLACE_REMOVE_SCALE)
                        combined[player]['place'][r][c] = total

                    # ----- 移除类：对方活子 -----
                    elif ((player == 0 and cell == BLU) or (player == 1 and cell == RED)):
                        pos_weight[player + 1][r][c] = 1
                        remove_4 = values[player]['removal'][:, r, c]
                        override_4 = values[player]['override'][:, r, c]
                        remove_val = exp_weight(remove_4, seq[1 - player])
                        override_val = exp_weight(override_4, seq[player])
                        total = (1 + REMOVE_BIAS) * remove_val + (1 - REMOVE_BIAS) * override_val
                        total *= (1 - PLACE_REMOVE_SCALE)
                        combined[player]['remove'][r][c] = total

        place_heart_red, remove_heart_red = heart_weight(red_heart, blue_heart)
        place_heart_blue, remove_heart_blue = heart_weight(blue_heart, red_heart)
        # print(place_heart_red, place_heart_blue, remove_heart_red, remove_heart_blue)
        for x, y in HEART_COORDS:
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

    # 分析 HEART 区域当前局势，返回当前局面对胜负产生关键影响的控制值：
    # - 若 RED 完全无子，则 BLUE 控制数越多，RED 越需要阻止（返回蓝方控制数）
    # - 若 BLUE 完全无子，则 RED 控制数越多，RED 越接近胜利（返回红方控制数）
    # - 若双方都有子，互相阻断，则无法构成 HEART 胜利 → 返回 0
    @staticmethod
    def evaluate_heart(chips):
        counts = {RED: 0, RED_SEQ: 0, BLU: 0, BLU_SEQ: 0, EMPTY: 0}
        for r, c in HEART_COORDS:
            counts[chips[r][c]] += 1

        red_total = counts[RED] + counts[RED_SEQ]
        blu_total = counts[BLU] + counts[BLU_SEQ]

        return (red_total, blu_total)

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

    @staticmethod
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

    @staticmethod
    # One eyed jacks can remove one opponent's chip
    def get_one_eyed_pos(chips, oc):
        res = []
        for i in range(10):
            for j in range(10):
                if (i, j) in COORDS['jk']:
                    continue
                elif chips[i][j] == oc:
                    res.append((i, j))
        return res

    @staticmethod
    # Normal cards can be placed into its position when EMPTY
    def get_normal_pos(chips, card):
        res = []
        for (i, j) in COORDS[card]:
            if chips[i][j] == EMPTY:
                res.append((i, j))
        return res

class Node:
    """
    The search tree node integrating MCTS and A*
    """
    def __init__(self, state, parent=None, action=None):
        # 状态表示
        try:
            self.state = state.clone()
        except:
            self.state = copy.deepcopy(state)
        # 节点关系
        self.parent = parent
        self.children = []
        self.action = action
        # MCTS统计数据
        self.visits = 0
        self.value = 0.0
        # 动作管理（延迟初始化）
        self.untried_actions = None

#
    def get_untried_actions(self):
        """获取未尝试的动作，使用启发式排序"""
        if self.untried_actions is None:
            # 初始化未尝试动作列表
            if hasattr(self.state, 'available_actions'):
                self.untried_actions = list(self.state.available_actions)
            else:
                self.untried_actions = []

            # 使用融合启发式排序（利用BoardEvaluator的评估）
            self.untried_actions.sort(key=lambda a: Node.hybrid_heuristic(self.state, a))
        return self.untried_actions

    def is_fully_expanded(self):
        """检查节点是否已完全展开"""
        return len(self.get_untried_actions()) == 0

    """
        Selection (MCTS stage 1)
    """
    def select_child(self):
        """使用UCB公式选择最有希望的子节点,整合BoardEvaluator评估"""
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            # UCB计算
            if child.visits == 0:
                score = float('inf')
            else:
                # 结合A*&BoardEvaluator的UCB计算
                exploitation = child.value / child.visits
                exploration = EXPLORATION_WEIGHT * math.sqrt(2 * math.log(self.visits) / child.visits)
                # 结合BoardEvaluator的启发式调整
                if child.action:
                    # 获取BoardEvaluator的评估作为调整因子
                    heuristic_factor = 1.0 / (1.0 + Node.hybrid_heuristic(self.state, child.action) / 100)
                else:
                    heuristic_factor = 1.0

                score = exploitation + exploration * heuristic_factor

            # 更新最佳节点
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    """
        Expansion (MCTS stage 2)
    """
    def expand(self, agent):
        """扩展一个新子节点，使用混合启发式选择最有前途的动作"""
        untried = self.get_untried_actions()

        if not untried:
            return None

        # 选择（并移除）列表中第一个动作（已通过混合启发式排序）
        action = untried.pop(0)
        # 创建新状态
        new_state = agent.fast_simulate(self.state, action)
        # 创建子节点
        child = Node(new_state, parent=self, action=action)
        self.children.append(child)

        return child

    def update(self, result):
        """更新节点统计信息"""
        self.visits += 1
        self.value += result

    """
            Simulation (MCTS stage 3)
    """
    @staticmethod
    def hybrid_heuristic(state, action):
        """A* + BoardEvaluator启发式函数 -评估动作的潜在价值（越低越好)"""
        if (action.get('type') not in ['place', 'remove']) or ('coords' not in action):
            return 100  # 非放置/移除动作或无坐标

        r, c = action['coords']
        if (r, c) in CORNERS:
            return 100  # 角落位置

        board = state.board.chips
        values = BoardEvaluator.combine_value(board)

        # 获取当前玩家ID
        if hasattr(state, 'my_color'):
            player_id = 0 if state.my_color == RED else 1
        else:
            player_id = state.current_player_id if hasattr(state, 'current_player_id') else 0

        # 根据动作类型获取评分
        if action.get('type') == 'place':
            # BoardEvaluator的放置评分
            score = values[player_id]['place'][r][c]
        else:  # 移除动作
            # BoardEvaluator的移除评分
            score = values[player_id]['remove'][r][c]

        # 为防止除零，加1后取倒数，并放大为0-100的评分范围
        # 高价值动作会得到低启发式分数
        return max(1, 1000 / (score + 1))

        # # 创建假设放置后的棋盘
        # board_copy = [row[:] for row in board]
        # board_copy[r][c] = color
        #
        # # 计算各种分数
        # score = 0
        #
        # # 中心偏好
        # distance = abs(r - 4.5) + abs(c - 4.5)
        # score += max(0, 5 - distance) * 2
        #
        # # 连续链评分
        # for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        #     count = 1  # 当前位置
        #     # 正向检查
        #     for i in range(1, 5):
        #         x, y = r + dx * i, c + dy * i
        #         if 0 <= x < 10 and 0 <= y < 10 and board_copy[x][y] == color:
        #             count += 1
        #         else:
        #             break
        #     # 反向检查
        #     for i in range(1, 5):
        #         x, y = r - dx * i, c - dy * i
        #         if 0 <= x < 10 and 0 <= y < 10 and board_copy[x][y] == color:
        #             count += 1
        #         else:
        #             break
        #
        #     # 根据连续长度评分
        #     if count >= 5:
        #         score += 200  # 形成序列
        #     elif count == 4:
        #         score += 100
        #     elif count == 3:
        #         score += 30
        #     elif count == 2:
        #         score += 10
        #
        # # 阻止对手评分
        # for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        #     enemy_chain = 0
        #
        #     # 检查移除此位置是否会破坏对手的连续链
        #     for i in range(1, 5):
        #         x, y = r + dx * i, c + dy * i
        #         if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
        #             enemy_chain += 1
        #         else:
        #             break
        #
        #     for i in range(1, 5):
        #         x, y = r - dx * i, c - dy * i
        #         if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
        #             enemy_chain += 1
        #         else:
        #             break
        #
        #     if enemy_chain >= 3:
        #         score += 50  # 高优先级阻断
        #
        # # 中心控制评分
        # heart_controlled = sum(1 for x, y in HEART_COORDS if board_copy[x][y] == color)
        # score += heart_controlled * 15
        #
        # # 转换为启发式分数（越低越好）
        # return 100 - score

    """
        Evaluation (MCTS stage 4)
    """
    @staticmethod
    def hybrid_evaluate(state, last_action=None):
        """使用BoardEvaluator对局面进行全面评估"""
        board = state.board.chips

        # 获取当前玩家ID
        if hasattr(state, 'my_color'):
            player_id = 0 if state.my_color == RED else 1
            opponent_id = 1 - player_id
        else:
            player_id = state.current_player_id if hasattr(state, 'current_player_id') else 0
            opponent_id = 1 - player_id

        # 使用BoardEvaluator的综合评估
        values = BoardEvaluator.combine_value(board)

        # 分别计算己方和对方的位置平均价值
        my_place_avg = np.mean(values[player_id]['place'])
        my_remove_avg = np.mean(values[player_id]['remove'])
        opp_place_avg = np.mean(values[opponent_id]['place'])
        opp_remove_avg = np.mean(values[opponent_id]['remove'])

        # 计算总体优势分数
        my_score = my_place_avg + my_remove_avg
        opp_score = opp_place_avg + opp_remove_avg

        # 心脏区域特殊加分
        heart_values, (red_heart, blue_heart) = BoardEvaluator.evaluate_board(board)
        if player_id == 0:  # 红方
            heart_diff = red_heart - blue_heart
        else:  # 蓝方
            heart_diff = blue_heart - red_heart

        heart_bonus = heart_diff * 5  # 心脏控制加分

        # 连续棋子价值计算
        line_values, _ = BoardEvaluator.evaluate_board(board)
        line_score = 0
        for i in range(1, 6):
            line_score += line_values[player_id][i] * (2 ** i)  # 指数加权
            line_score -= line_values[opponent_id][i] * (2 ** i)  # 对手减分

        # 综合得分，归一化到[-1, 1]区间
        total_advantage = my_score - opp_score + heart_bonus + line_score * 0.1
        normalized_score = max(-1, min(1, total_advantage / 200))

        return normalized_score


class myAgent(Agent):
    """
    智能体 myAgent: 融合融合BoardEvaluator与MCTS+A*的版本
    """

    def __init__(self, _id):
        """初始化Agent"""
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)  # 2人游戏
        self.counter = itertools.count()  # 用于A*搜索的唯一标识符

        # 玩家颜色初始化
        self.my_color = None
        self.opp_color = None

        # 搜索参数
        self.simulation_depth = 5  # 模拟深度
        self.candidate_limit = 10  # A*筛选的候选动作数

        # 时间控制
        self.start_time = 0

        # 局面复杂度分析
        self.game_phase = "early"  # early, mid, late
        self.move_count = 0

    def SelectAction(self, actions, game_state):
        """主决策函数 - 结合BoardEvaluator与MCTS+A*"""
        self.start_time = time.time()
        self.move_count += 1

        # 游戏阶段判断（基于棋盘填充程度）
        filled_count = 0
        for r in range(10):
            for c in range(10):
                if game_state.board.chips[r][c] not in [EMPTY, JOKER]:
                    filled_count += 1

        # 根据填充程度确定游戏阶段
        if filled_count < 20:
            self.game_phase = "early"
        elif filled_count < 40:
            self.game_phase = "mid"
        else:
            self.game_phase = "late"

        # 初始化颜色信息（如果尚未初始化）
        if self.my_color is None:
            self.my_color = game_state.agents[self.id].colour
            self.opp_color = game_state.agents[1 - self.id].colour

        # 特殊情况处理：卡牌交易/选择（针对五张展示牌变体）
        if any(a.get('type') == 'trade' for a in actions):
            trade_actions = [a for a in actions if a.get('type') == 'trade']
            return self._select_strategic_card(trade_actions, game_state)

        # 准备一个默认的随机动作作为后备
        valid_actions = [a for a in actions if 'coords' not in a or a['coords'] not in CORNERS]
        default_action = random.choice(valid_actions) if valid_actions else random.choice(actions)

        # 使用BoardEvaluator对所有动作进行评估
        board = game_state.board.chips
        values = BoardEvaluator.combine_value(board)

        # 给所有动作评分
        scored_actions = []
        for action in actions:
            # 不考虑角落位置
            if action.get('coords') in CORNERS:
                continue

            if action.get('type') == 'place' and 'coords' in action:
                r, c = action['coords']
                score = values[self.id]['place'][r][c]
                scored_actions.append((action, score))
            elif action.get('type') == 'remove' and 'coords' in action:
                r, c = action['coords']
                score = values[self.id]['remove'][r][c]
                scored_actions.append((action, score))
            else:
                # 其他类型动作（如trade）
                scored_actions.append((action, 0))

        # 按评分排序（降序）
        scored_actions.sort(key=lambda x: x[1], reverse=True)

        # 如果没有有效动作，使用默认动作
        if not scored_actions:
            return default_action

        # 检查分数差异
        if len(scored_actions) >= 2:
            best_score = scored_actions[0][1]
            second_score = scored_actions[1][1]
            score_diff = best_score - second_score

            # 如果最佳动作明显优于其他动作，直接返回
            if score_diff > 100 or best_score > 1000:
                return scored_actions[0][0]

        # 简单局面和早期阶段：只使用BoardEvaluator
        if self.game_phase == "early" or len(actions) < 5:
            return scored_actions[0][0]

        # 检查剩余时间
        remaining_time = MAX_THINK_TIME - (time.time() - self.start_time)
        if remaining_time < 0.4:  # 时间不足时直接使用BoardEvaluator
            return scored_actions[0][0]

        # 筛选候选动作（当动作太多时，只使用前N个）
        candidate_actions = [a for a, _ in scored_actions[:self.candidate_limit]]

        # 复杂局面：使用MCTS深度分析
        try:
            return self._hybrid_mcts_search(candidate_actions, game_state)
        except Exception as e:
            # 出错时返回BoardEvaluator的最佳动作
            return scored_actions[0][0]

    def _a_star_filter(self, actions, game_state):
        """使用A*算法筛选最有前途的动作"""
        # 排除角落位置
        valid_actions = [a for a in actions if 'coords' not in a or a['coords'] not in CORNERS]
        if not valid_actions:
            return actions[:1]  # 如果没有有效动作，返回第一个动作

        # 评估每个动作
        scored_actions = []
        for action in valid_actions:
            score = Node.hybrid_heuristic(game_state, action)
            scored_actions.append((action, score))

        # 按评分排序（升序，越小越好）
        scored_actions.sort(key=lambda x: x[1])

        # 返回前N个候选动作
        candidates = [a for a, _ in scored_actions[:self.candidate_limit]]
        return candidates

    def _hybrid_mcts_search(self, candidate_actions, game_state):
        """混合MCTS搜索，使用BoardEvaluator增强的评估和模拟"""
        # 准备MCTS状态
        mcts_state = self._prepare_state_for_mcts(game_state, candidate_actions)
        root = Node(mcts_state)

        # 直接为根节点创建子节点
        for action in candidate_actions:
            next_state = self.fast_simulate(mcts_state, action)
            child = Node(next_state, parent=root, action=action)
            root.children.append(child)

        # MCTS主循环
        iterations = 0
        while not self._is_timeout() and iterations < SIMULATION_LIMIT:
            iterations += 1

            # 1. 选择阶段
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.select_child()

            # 2. 扩展阶段
            if node.visits > 0 and not node.is_fully_expanded():
                child = node.expand(self)
                if child:
                    node = child

            # 3. 模拟阶段
            value = self._hybrid_simulate(node.state)

            # 4. 回溯阶段
            while node:
                node.update(value)
                node = node.parent

        # 检查是否有足够的迭代
        if iterations < 10:  # 如果模拟次数太少，直接使用第一个动作
            return candidate_actions[0]

        # 选择最佳动作（综合访问次数和平均价值）
        if not root.children:
            return candidate_actions[0]

        # 使用加权评分选择最佳子节点
        best_child = None
        best_score = float('-inf')

        for child in root.children:
            if child.visits == 0:
                continue

            # 结合访问次数和平均值的评分
            visit_score = child.visits / max(1, iterations) * 0.7  # 访问比例 (70% 权重)
            value_score = (child.value / child.visits + 1) / 2 * 0.3  # 归一化值 (30% 权重)
            score = visit_score + value_score

            if score > best_score:
                best_score = score
                best_child = child

        if best_child:
            return best_child.action
        else:
            return candidate_actions[0]

    def _hybrid_simulate(self, state):
        """混合模拟"""
        state_copy = self.custom_shallow_copy(state)
        current_depth = 0

        while current_depth < self.simulation_depth:
            current_depth += 1

            # 获取可用动作
            if hasattr(state_copy, 'available_actions'):
                actions = state_copy.available_actions
            else:
                try:
                    actions = self.rule.getLegalActions(state_copy, self.id)
                except:
                    actions = []

            if not actions:
                break

            # 使用BoardEvaluator评估所有可能的动作
            board = state_copy.board.chips
            values = BoardEvaluator.combine_value(board)

            # 获取当前玩家ID
            player_id = state_copy.current_player_id if hasattr(state_copy, 'current_player_id') else self.id

            # 评估每个动作
            scored_actions = []
            for action in actions:
                # 根据动作类型分别评分
                if action.get('type') == 'place' and 'coords' in action:
                    r, c = action['coords']
                    score = values[player_id]['place'][r][c]
                elif action.get('type') == 'remove' and 'coords' in action:
                    r, c = action['coords']
                    score = values[player_id]['remove'][r][c]
                else:
                    score = 0
                scored_actions.append((action, score))

            # 90%时间选择高价值动作，10%时间随机选择（保持探索性）
            if random.random() < 0.9 and scored_actions:
                # 将动作按价值排序
                scored_actions.sort(key=lambda x: x[1], reverse=True)

                # 避免总是选最优，从前三个中随机选择增加多样性
                top_n = min(3, len(scored_actions))
                idx = random.randint(0, top_n - 1) if top_n > 0 else 0
                action = scored_actions[idx][0] if idx < len(scored_actions) else random.choice(actions)
            else:
                # 10%随机选择
                action = random.choice(actions)

            # 应用动作
            state_copy = self.fast_simulate(state_copy, action)

            # 模拟卡牌选择（针对5张展示牌变体）
            self._simulate_card_selection(state_copy)

        # 使用混合评估函数评估最终状态
        return Node.hybrid_evaluate(state_copy)

    def _simulate_card_selection(self, state):
        """模拟从5张展示牌中选择一张"""
        # 检查是否有展示牌属性
        if not hasattr(state, 'display_cards') or not state.display_cards:
            return
        # 评估每张牌的价值
        best_card = None
        best_value = float('-inf')

        # 获取当前棋盘
        board = state.board.chips
        player_id = state.current_player_id if hasattr(state, 'current_player_id') else self.id

        for card in state.display_cards:
            value = self._evaluate_card(card, state)
            if value > best_value:
                best_value = value
                best_card = card
        # 确保找到了最佳牌
        if not best_card:
            return
        # 更新玩家手牌
        if hasattr(state, 'agents'):
            if hasattr(state, 'current_player_id'):
                player_id = state.current_player_id
                if 0 <= player_id < len(state.agents) and hasattr(state.agents[player_id], 'hand'):
                    state.agents[player_id].hand.append(best_card)
            else:
                # 使用自己的ID
                if 0 <= self.id < len(state.agents) and hasattr(state.agents[self.id], 'hand'):
                    state.agents[self.id].hand.append(best_card)

        # 从展示区移除所选卡牌（再次检查，以防展示牌在其他地方被修改）
        if best_card in state.display_cards:
            state.display_cards.remove(best_card)

            # 从牌堆补充一张牌（确保牌堆非空且有牌）
            if hasattr(state, 'deck') and state.deck:
                try:
                    new_card = state.deck[0]  # 先查看第一张牌，不修改deck
                    state.display_cards.append(state.deck.pop(0))
                except IndexError:  # 处理边缘情况：属性检查后牌堆变空
                    pass  # 不处理，仅展示区减少一张牌

    def _evaluate_card(self, card, state, board=None, player_id=None):
        """增强版卡牌评估，结合棋盘位置价值和卡牌特性"""
        # 如果没有提供棋盘和玩家ID，则获取它们
        if board is None:
            board = state.board.chips
        if player_id is None:
            player_id = self.id

        # 使用BoardEvaluator计算棋盘价值
        board_values = BoardEvaluator.combine_value(board)

        # 检查特殊牌：Jack牌
        card_str = str(card).lower()
        if card_str[0] == 'j':
            if card_str[1] in ['h', 's']:  # 单眼J牌
                # 获取移除价值最高的位置
                max_remove_value = np.max(board_values[player_id]['remove'])
                return max_remove_value + 50  # 额外加分
            elif card_str[1] in ['d', 'c']:  # 双眼J牌
                # 获取放置价值最高的位置
                max_place_value = np.max(board_values[player_id]['place'])
                return max_place_value + 30  # 额外加分

        # 普通牌：分析它能放置的位置价值
        max_value = 0
        if card in COORDS:
            positions = COORDS[card]
            # 检查卡牌对应位置的价值
            for r, c in positions:
                if board[r][c] == EMPTY:  # 位置为空才能放置
                    pos_value = board_values[player_id]['place'][r][c]
                    max_value = max(max_value, pos_value)

        return max_value

    def _select_strategic_card(self, trade_actions, game_state):
        """策略性地选择卡牌"""
        # 处理变体规则：从5张展示牌中选择
        if hasattr(game_state, 'display_cards') and game_state.display_cards:
            best_card = None
            best_value = float('-inf')

            # 获取当前棋盘状态
            board = game_state.board.chips

            for card in game_state.display_cards:
                value = self._evaluate_card(card, game_state, board)
                if value > best_value:
                    best_value = value
                    best_card = card

            # 寻找对应的动作
            if best_card:
                for action in trade_actions:
                    if action.get('draft_card') == best_card:
                        return action

        # 备选策略：优先选择Jack牌
        jack_actions = []
        for action in trade_actions:
            card = action.get('draft_card', '')
            card_str = str(card).lower()
            if card_str and card_str[0] == 'j':
                if card_str[1] in ['h', 's']:  # 单眼J
                    jack_actions.append((action, 10))  # 最高优先级
                elif card_str[1] in ['d', 'c']:  # 双眼J
                    jack_actions.append((action, 8))  # 次高优先级

        # 如果有Jack牌，按优先级选择
        if jack_actions:
            jack_actions.sort(key=lambda x: x[1], reverse=True)
            return jack_actions[0][0]

            # 最后备选：随机选择
        return random.choice(trade_actions)

    def _prepare_state_for_mcts(self, game_state, actions):
        """准备用于MCTS的游戏状态"""
        # 创建状态副本
        mcts_state = self.custom_shallow_copy(game_state)
        # 添加必要的属性
        mcts_state.my_color = self.my_color
        mcts_state.opp_color = self.opp_color
        mcts_state.current_player_id = self.id
        # 添加可用动作
        mcts_state.available_actions = actions

        return mcts_state

    def fast_simulate(self, state, action):
        """快速模拟执行动作"""
        new_state = state.copy() if hasattr(state, "copy") else self.custom_shallow_copy(state)

        # 处理放置动作
        if action['type'] == 'place' and 'coords' in action:
            r, c = action['coords']
            # 确定颜色
            color = self.my_color
            if hasattr(state, 'current_player_id'):
                color = state.agents[state.current_player_id].colour
            # 放置棋子
            new_state.board.chips[r][c] = color
            # 更新手牌（如果需要）
            if hasattr(new_state, 'agents') and hasattr(new_state.agents[self.id], 'hand'):
                if 'play_card' in action:
                    card = action['play_card']
                    try:
                        new_state.agents[self.id].hand.remove(card)
                    except:
                        pass
        # 处理移除动作
        elif action['type'] == 'remove' and 'coords' in action:
            r, c = action['coords']
            # 移除棋子
            new_state.board.chips[r][c] = EMPTY
            # 更新手牌
            if hasattr(new_state, 'agents') and hasattr(new_state.agents[self.id], 'hand'):
                if 'play_card' in action:
                    card = action['play_card']
                    try:
                        new_state.agents[self.id].hand.remove(card)
                    except:
                        pass
        return new_state

    def custom_shallow_copy(self, state):
        """创建游戏状态的深拷贝"""
        from copy import deepcopy
        return deepcopy(state)

    def _is_timeout(self):
        """检查是否超时"""
        return time.time() - self.start_time > MAX_THINK_TIME * 0.95