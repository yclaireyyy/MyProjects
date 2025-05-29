from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import random
import time
import copy
import math
import itertools
from collections import defaultdict, deque

# ===========================
# 1. 常量定义区
# ===========================
MAX_THINK_TIME = 0.95  # 最大思考时间（秒）
EXPLORATION_WEIGHT = 1.414  # UCB公式中的探索参数（优化：使用理论最优值）
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]  # 中心热点位置
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]  # 角落位置（自由点）
SIMULATION_LIMIT = 300  # MCTS模拟的最大次数（提升）

# 预计算的方向向量和位置权重
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
POSITION_WEIGHTS = {}  # 位置权重缓存
ACTION_CACHE = {}  # 动作评估缓存

# 初始化位置权重（优化：考虑角落的特殊性）
for i in range(10):
    for j in range(10):
        if (i, j) in CORNERS:
            POSITION_WEIGHTS[(i, j)] = 2.0  # 角落是自由点，权重更高
        elif (i, j) in HOTB_COORDS:
            POSITION_WEIGHTS[(i, j)] = 1.5
        elif i in [0, 9] or j in [0, 9]:
            POSITION_WEIGHTS[(i, j)] = 0.8
        else:
            POSITION_WEIGHTS[(i, j)] = 1.0


# ===========================
# 2. 新增：游戏阶段定义
# ===========================
class GamePhase:
    OPENING = "opening"  # 开局（棋子少于15个）
    MIDDLE = "middle"  # 中局（15-35个棋子）
    CRITICAL = "critical"  # 关键阶段（有即将完成的序列）
    ENDGAME = "endgame"  # 终局（超过35个棋子）


# ===========================
# 3. 新增：LRU缓存实现
# ===========================
class LRUCache:
    """简单的LRU缓存实现，用于限制缓存大小"""

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            # 移除最久未使用的项
            oldest = self.order.popleft()
            del self.cache[oldest]

        self.cache[key] = value
        self.order.append(key)

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.order.clear()


class BoardState:
    """轻量级棋盘状态，用于快速复制"""

    def __init__(self, chips):
        self.chips = chips
        self._hash = None

    def get_hash(self):
        """获取棋盘状态哈希值用于缓存"""
        if self._hash is None:
            self._hash = hash(tuple(tuple(row) for row in self.chips))
        return self._hash

    def copy(self):
        """快速复制棋盘"""
        return BoardState([row[:] for row in self.chips])


class CardEvaluator:
    def __init__(self, agent):
        self.agent = agent
        self._card_cache = LRUCache(capacity=5000)  # 使用LRU缓存
        self._hand_diversity_cache = {}

    def _evaluate_card(self, card, state, consider_opponent=True):
        """评估卡牌在当前状态下的价值（增强版）"""
        # 生成缓存键
        board_hash = state.board.chips if hasattr(state.board, 'get_hash') else hash(
            tuple(tuple(row) for row in state.board.chips))
        cache_key = (str(card), board_hash, consider_opponent)

        cached_result = self._card_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        board = state.board.chips

        # 优先级1：双眼J - 直接最高分
        if self._is_two_eyed_jack(card):
            result = 10000
        # 优先级2：单眼J - 次高分
        elif self._is_one_eyed_jack(card):
            result = 5000
        # 优先级3：普通卡牌 - 使用指数评分
        elif card in COORDS:
            my_score = self._exponential_card_evaluation(card, state)

            # 新增：考虑对手价值
            if consider_opponent and hasattr(state, 'agents') and len(state.agents) > 1:
                opp_score = self._evaluate_card_for_opponent(card, state)
                result = my_score - opp_score * 0.3  # 减去对手价值的30%
            else:
                result = my_score
        else:
            result = 0

        self._card_cache.put(cache_key, result)
        return result

    def _evaluate_card_for_opponent(self, card, state):
        """评估卡牌对对手的价值"""
        # 临时切换颜色评估
        original_color = self.agent.my_color
        self.agent.my_color = self.agent.opp_color
        opp_value = self._exponential_card_evaluation(card, state)
        self.agent.my_color = original_color
        return opp_value

    def _calculate_hand_diversity(self, hand):
        """计算手牌多样性得分"""
        if not hand:
            return 0

        # 统计各种牌的数量
        card_counts = defaultdict(int)
        for card in hand:
            if card in COORDS:
                card_counts[str(card)] += 1

        # 多样性得分：不同卡牌种类越多越好
        diversity = len(card_counts)
        # 惩罚重复卡牌
        penalty = sum(count - 1 for count in card_counts.values() if count > 1)

        return diversity - penalty * 0.5

    def _exponential_card_evaluation(self, card, state):
        """基于指数的普通卡牌评估（优化版）"""
        if card not in COORDS:
            return 0

        board = state.board.chips
        total_score = 0
        # 获取该卡牌对应的所有可能位置
        positions = COORDS[card] if isinstance(COORDS[card], list) else [COORDS[card]]
        valid_positions = 0

        for pos in positions:
            if isinstance(pos, list):  # BUG修复：处理嵌套列表
                pos = tuple(pos)
            r, c = pos
            # 检查位置是否可用（修复：正确处理角落）
            if not self._is_position_available(board, r, c):
                continue
            # 计算该位置的指数评分
            position_score = self._calculate_position_score(board, r, c)
            total_score += position_score
            valid_positions += 1

        # 如果有多个位置，取平均值
        return total_score / max(1, valid_positions) if valid_positions > 0 else 0

    def _calculate_position_score(self, board, r, c):
        """计算单个位置的指数评分（优化版）"""
        total_score = 0

        # 四个主要方向：水平， 垂直，主对角线，反对角线
        for dx, dy in DIRECTIONS:
            my_pieces = self._count_my_pieces_in_direction(board, r, c, dx, dy)
            direction_score = self._exponential_scoring(my_pieces)
            total_score += direction_score

        return total_score

    def _count_my_pieces_in_direction(self, board, r, c, dx, dy):
        """统计特定方向5个位置内的我方棋子数量（优化：正确处理角落）"""
        my_pieces = 0
        my_color = self.agent.my_color

        # 检查该方向前后各4个位置（共8个位置）
        for i in range(-4, 5):
            # 跳过中心位置（即将放置的位置）
            if i == 0:
                continue
            x, y = r + i * dx, c + i * dy
            # 边界检查
            if 0 <= x < 10 and 0 <= y < 10:
                # 角落是自由点，任何人都可以使用
                if (x, y) in CORNERS:
                    my_pieces += 1  # 角落算作我方棋子
                elif board[x][y] == my_color:
                    my_pieces += 1

        return my_pieces

    def _exponential_scoring(self, piece_count):
        """指数评分规则：1个=10分，2个=100分，3个=1000分"""
        if piece_count == 0:
            return 1  # 基础分
        elif piece_count == 1:
            return 10
        elif piece_count == 2:
            return 100
        elif piece_count == 3:
            return 1000
        elif piece_count >= 4:
            return 10000  # 4个或以上 - 接近获胜

        return 0

    def _is_two_eyed_jack(self, card):
        """检查是否为双眼J"""
        try:
            card_str = str(card).lower()
            return card_str in ['jc', 'jd']  # 双眼J
        except:
            return False

    def _is_one_eyed_jack(self, card):
        """检查是否为单眼J"""
        try:
            card_str = str(card).lower()
            return card_str in ['js', 'jh']  # 单眼J
        except:
            return False

    def _is_position_available(self, board, r, c):
        """检查位置是否可用（修复：正确处理角落）"""
        if not (0 <= r < 10 and 0 <= c < 10):
            return False

        # 修复：角落是自由点，总是可用的（对于序列完成）
        if (r, c) in CORNERS:
            return True

        # 普通位置需要是空的
        return board[r][c] == 0 or board[r][c] == '0'


class ActionEvaluator:
    _evaluation_cache = LRUCache(capacity=15000)  # 增加缓存容量
    _threat_cache = LRUCache(capacity=8000)

    @staticmethod
    def evaluate_action_quality(state, action):
        """评估动作的质量得分（越低越好）- 增强版"""
        if action.get('type') != 'place' or 'coords' not in action:
            return 100  # 非放置动作或无坐标

        r, c = action['coords']

        # 生成缓存键
        board_hash = hash(tuple(tuple(row) for row in state.board.chips))
        cache_key = (board_hash, r, c)

        cached_result = ActionEvaluator._evaluation_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        board = state.board.chips

        # 获取玩家颜色
        if hasattr(state, 'my_color'):
            color = state.my_color
            enemy = state.opp_color
        else:
            # 从行动中推断颜色
            agent_id = state.current_player_id if hasattr(state, 'current_player_id') else 0
            color = state.agents[agent_id].colour
            enemy = 'r' if color == 'b' else 'b'

        # 计算各种分数
        score = 0

        # 中心偏好（使用预计算的距离）
        distance = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 5 - distance) * 2

        # 角落特殊处理：角落是自由点，有特殊价值
        if (r, c) in CORNERS:
            score += 20  # 角落的基础价值

        # 连续链评分（优化版，正确处理角落）
        for dx, dy in DIRECTIONS:
            count = ActionEvaluator._count_consecutive_with_corners(board, r, c, dx, dy, color)
            # 根据连续长度评分
            if count >= 5:
                score += 3000  # 形成序列 - 大幅提高分数
            elif count == 4:
                score += 800  # 接近获胜
            elif count == 3:
                score += 150
            elif count == 2:
                score += 30

        # 增强防守评分（正确处理角落）
        critical_block = False
        for dx, dy in DIRECTIONS:
            enemy_chain = ActionEvaluator._count_enemy_threat_with_corners(board, r, c, dx, dy, enemy)
            if enemy_chain >= 4:
                score += 1500  # 极高优先级阻断
                critical_block = True
            elif enemy_chain >= 3:
                score += 400  # 高优先级阻断

            # 检测双向威胁
            if ActionEvaluator._check_double_threat(board, r, c, dx, dy, enemy):
                score += 500

        # 特殊情况：如果是关键防守，大幅提升得分
        if critical_block:
            score *= 2

        # 中心控制评分（使用位置权重）
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if (x, y) == (r, c))
        score += hotb_controlled * 20

        # 转换为质量分数（越低越好）
        result = 100 - score
        ActionEvaluator._evaluation_cache.put(cache_key, result)
        return result

    @staticmethod
    def _count_consecutive_with_corners(board, x, y, dx, dy, color):
        """计算连续棋子数，正确处理角落作为自由点"""
        count = 1  # 起始位置算一个

        # 正向检查
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < 10 and 0 <= ny < 10:
                if (nx, ny) in CORNERS or board[nx][ny] == color:
                    count += 1
                else:
                    break
            else:
                break

        # 反向检查
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if 0 <= nx < 10 and 0 <= ny < 10:
                if (nx, ny) in CORNERS or board[nx][ny] == color:
                    count += 1
                else:
                    break
            else:
                break

        return min(count, 5)  # 最多返回5（形成一个序列）

    @staticmethod
    def _count_enemy_threat_with_corners(board, r, c, dx, dy, enemy):
        """计算敌方威胁，正确处理角落作为自由点"""
        enemy_chain = 0
        # 正向检查
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if (x, y) in CORNERS or board[x][y] == enemy:
                    enemy_chain += 1
                else:
                    break
            else:
                break

        # 反向检查
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if (x, y) in CORNERS or board[x][y] == enemy:
                    enemy_chain += 1
                else:
                    break
            else:
                break

        return enemy_chain

    @staticmethod
    def _check_double_threat(board, r, c, dx, dy, enemy):
        """检查是否存在双向威胁"""
        # 检查两端是否都可以延伸
        threats = 0

        # 正向检查
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if (x, y) in CORNERS or board[x][y] == 0 or board[x][y] == '0':
                    threats += 1
                    break
                elif board[x][y] != enemy and (x, y) not in CORNERS:
                    break

        # 反向检查
        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if (x, y) in CORNERS or board[x][y] == 0 or board[x][y] == '0':
                    threats += 1
                    break
                elif board[x][y] != enemy and (x, y) not in CORNERS:
                    break

        return threats >= 2

    @staticmethod
    def is_winning_move(state, action, color):
        """检查是否为获胜动作"""
        if action.get('type') != 'place' or 'coords' not in action:
            return False

        r, c = action['coords']
        board = state.board.chips

        # 模拟放置
        board_copy = [row[:] for row in board]
        board_copy[r][c] = color

        # 检查是否形成序列（使用角落友好的检查）
        for dx, dy in DIRECTIONS:
            if ActionEvaluator._count_consecutive_with_corners(board_copy, r, c, dx, dy, color) >= 5:
                return True
        return False

    @staticmethod
    def blocks_opponent_win(state, action, enemy):
        """检查是否阻止对手获胜"""
        if action.get('type') != 'place' or 'coords' not in action:
            return False

        r, c = action['coords']
        board = state.board.chips

        # 检查该位置是否是对手的关键位置
        for dx, dy in DIRECTIONS:
            if ActionEvaluator._count_enemy_threat_with_corners(board, r, c, dx, dy, enemy) >= 4:
                return True
        return False

    # 保持向后兼容的方法
    @staticmethod
    def _count_consecutive_fast(board, x, y, dx, dy, color):
        """向后兼容：使用新的角落友好方法"""
        return ActionEvaluator._count_consecutive_with_corners(board, x, y, dx, dy, color)

    @staticmethod
    def _count_enemy_threat_fast(board, r, c, dx, dy, enemy):
        """向后兼容：使用新的角落友好方法"""
        return ActionEvaluator._count_enemy_threat_with_corners(board, r, c, dx, dy, enemy)

    @staticmethod
    def _count_consecutive(board, x, y, dx, dy, color):
        """向后兼容的方法"""
        return ActionEvaluator._count_consecutive_with_corners(board, x, y, dx, dy, color)

    @staticmethod
    def _count_enemy_threat(board, r, c, dx, dy, enemy):
        """向后兼容的方法"""
        return ActionEvaluator._count_enemy_threat_with_corners(board, r, c, dx, dy, enemy)


class StateEvaluator:
    _state_cache = LRUCache(capacity=8000)  # 增加缓存容量

    @staticmethod
    def evaluate(state, last_action=None):
        """评估游戏状态的价值（带缓存，正确处理角落）"""
        # 生成缓存键
        board_hash = hash(tuple(tuple(row) for row in state.board.chips))

        cached_result = StateEvaluator._state_cache.get(board_hash)
        if cached_result is not None:
            return cached_result

        board = state.board.chips

        # 获取玩家颜色
        if hasattr(state, 'my_color'):
            my_color = state.my_color
            opp_color = state.opp_color
        else:
            # 从状态中推断颜色
            agent_id = state.current_player_id if hasattr(state, 'current_player_id') else 0
            my_color = state.agents[agent_id].colour
            opp_color = 'r' if my_color == 'b' else 'b'

        # 1. 位置评分（使用预计算权重）
        position_score = 0
        for i in range(10):
            for j in range(10):
                cell = board[i][j]
                if cell == my_color:
                    position_score += POSITION_WEIGHTS[(i, j)]
                elif cell == opp_color:
                    position_score -= POSITION_WEIGHTS[(i, j)]

        # 2. 序列潜力评分（批量处理，正确处理角落）
        sequence_score = StateEvaluator._calculate_sequence_score_with_corners(board, my_color)

        # 3. 防御评分 - 阻止对手的序列（正确处理角落）
        defense_score = StateEvaluator._calculate_defense_score_with_corners(board, opp_color)

        # 4. 中心控制评分
        hotb_score = 0
        for x, y in HOTB_COORDS:
            cell = board[x][y]
            if cell == my_color:
                hotb_score += 8
            elif cell == opp_color:
                hotb_score -= 8

        # 5. 综合评分
        total_score = position_score + sequence_score + defense_score + hotb_score

        # 归一化到[-1, 1]区间
        result = max(-1, min(1, total_score / 300))
        StateEvaluator._state_cache.put(board_hash, result)
        return result

    @staticmethod
    def _calculate_sequence_score_with_corners(board, color):
        """优化的序列得分计算（正确处理角落）"""
        sequence_score = 0
        # 使用集合避免重复计算
        counted_sequences = set()

        for i in range(10):
            for j in range(10):
                # 检查我方棋子或角落
                if board[i][j] == color or (i, j) in CORNERS:
                    for dx, dy in DIRECTIONS:
                        # 生成序列的唯一标识
                        sequence_id = (i, j, dx, dy)
                        if sequence_id not in counted_sequences:
                            count = ActionEvaluator._count_consecutive_with_corners(board, i, j, dx, dy, color)
                            if count >= 2:  # 只计算有意义的序列
                                # 将整个序列标记为已计算
                                for k in range(count):
                                    counted_sequences.add((i + k * dx, j + k * dy, dx, dy))

                                if count >= 5:
                                    sequence_score += 150
                                elif count == 4:
                                    sequence_score += 40
                                elif count == 3:
                                    sequence_score += 10
                                elif count == 2:
                                    sequence_score += 2

        return sequence_score

    @staticmethod
    def _calculate_defense_score_with_corners(board, opp_color):
        """优化的防御得分计算（正确处理角落）"""
        defense_score = 0
        # 使用集合避免重复计算
        counted_threats = set()

        for i in range(10):
            for j in range(10):
                # 检查对手棋子或角落
                if board[i][j] == opp_color or (i, j) in CORNERS:
                    for dx, dy in DIRECTIONS:
                        threat_id = (i, j, dx, dy)
                        if threat_id not in counted_threats:
                            count = ActionEvaluator._count_consecutive_with_corners(board, i, j, dx, dy, opp_color)
                            if count >= 3:  # 只关注真正的威胁
                                # 标记整个威胁序列
                                for k in range(count):
                                    counted_threats.add((i + k * dx, j + k * dy, dx, dy))

                                if count >= 4:
                                    defense_score -= 80
                                elif count == 3:
                                    defense_score -= 20

        return defense_score

    # 向后兼容的方法
    @staticmethod
    def _calculate_sequence_score_fast(board, color):
        """向后兼容的方法"""
        return StateEvaluator._calculate_sequence_score_with_corners(board, color)

    @staticmethod
    def _calculate_defense_score_fast(board, opp_color):
        """向后兼容的方法"""
        return StateEvaluator._calculate_defense_score_with_corners(board, opp_color)

    @staticmethod
    def _calculate_sequence_score(board, color):
        """向后兼容的方法"""
        return StateEvaluator._calculate_sequence_score_with_corners(board, color)

    @staticmethod
    def _calculate_defense_score(board, opp_color):
        """向后兼容的方法"""
        return StateEvaluator._calculate_defense_score_with_corners(board, opp_color)


class ActionSimulator:
    def __init__(self, agent):
        self.agent = agent

    def simulate_action(self, state, action):
        """模拟执行动作"""
        new_state = self._copy_state(state)

        if action['type'] == 'place':
            self._simulate_place(new_state, action)
        elif action['type'] == 'remove':
            self._simulate_remove(new_state, action)

        return new_state

    def _simulate_place(self, state, action):
        """模拟放置动作"""
        if 'coords' not in action:
            return

        r, c = action['coords']

        # 角落不能放置棋子（角落是自由点，不是可放置位置）
        if (r, c) in CORNERS:
            return

        color = self._get_current_color(state)
        state.board.chips[r][c] = color

        # 更新手牌
        self._update_hand(state, action)

    def _simulate_remove(self, state, action):
        """模拟移除动作"""
        if 'coords' not in action:
            return

        r, c = action['coords']

        # 角落不能移除（角落是自由点）
        if (r, c) in CORNERS:
            return

        state.board.chips[r][c] = 0  # 移除棋子

        # 更新手牌
        self._update_hand(state, action)

    def _get_current_color(self, state):
        """获取当前玩家颜色"""
        if hasattr(state, 'current_player_id'):
            return state.agents[state.current_player_id].colour
        return self.agent.my_color

    def _update_hand(self, state, action):
        """更新手牌"""
        if 'play_card' not in action:
            return

        card = action['play_card']
        player_id = getattr(state, 'current_player_id', self.agent.id)

        try:
            if (hasattr(state, 'agents') and
                    0 <= player_id < len(state.agents) and
                    hasattr(state.agents[player_id], 'hand')):
                state.agents[player_id].hand.remove(card)
        except (ValueError, AttributeError):
            pass

    def _copy_state(self, state):
        """拷贝状态"""
        if hasattr(state, "copy"):
            return state.copy()
        else:
            return copy.deepcopy(state)


class Node:
    """
    MCTS搜索树节点，集成启发式评估（优化版）
    """

    def __init__(self, state, parent=None, action=None):
        # 状态表示（优化的复制）
        self.state = self._efficient_copy_state(state)
        # 节点关系
        self.parent = parent
        self.children = []
        self.action = action
        # MCTS统计数据
        self.visits = 0
        self.value = 0.0
        self.squared_value = 0.0  # 新增：用于计算方差的UCB
        # 动作管理（延迟初始化）
        self.untried_actions = None
        # 新增：是否为终止节点
        self.is_terminal = False
        # 新增：RAVE统计（快速平均值估计）
        self.rave_visits = defaultdict(int)
        self.rave_value = defaultdict(float)

    def _efficient_copy_state(self, state):
        """高效的状态复制"""
        try:
            return state.clone()
        except:
            # 只复制关键部分
            if hasattr(state, 'board') and hasattr(state.board, 'chips'):
                new_state = copy.copy(state)
                new_state.board = copy.copy(state.board)
                new_state.board.chips = [row[:] for row in state.board.chips]
                return new_state
            return copy.deepcopy(state)

    def get_untried_actions(self):
        """获取未尝试的动作，使用启发式排序"""
        if self.untried_actions is None:
            # 初始化未尝试动作列表
            if hasattr(self.state, 'available_actions'):
                self.untried_actions = list(self.state.available_actions)
            else:
                self.untried_actions = []
            # 使用启发式评估进行排序（越小越优先）
            self.untried_actions.sort(key=lambda a: ActionEvaluator.evaluate_action_quality(self.state, a))
        return self.untried_actions

    def is_fully_expanded(self):
        """检查节点是否已完全展开"""
        return len(self.get_untried_actions()) == 0

    def select_child(self, c_param=EXPLORATION_WEIGHT, use_variance=True):
        """使用改进的UCB公式选择最有希望的子节点"""
        best_score = float('-inf')
        best_child = None
        log_visits = math.log(self.visits) if self.visits > 0 else 0

        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                # 基础UCB
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(log_visits / child.visits)

                # 可选：添加方差项（UCB-V）
                if use_variance and child.visits > 1:
                    variance = (child.squared_value / child.visits) - (exploitation ** 2)
                    variance_term = math.sqrt(variance * log_visits / child.visits)
                    exploration += variance_term * 0.1  # 较小的方差权重

                # 启发式调整
                if child.action:
                    heuristic_score = ActionEvaluator.evaluate_action_quality(self.state, child.action)
                    heuristic_factor = 1.0 / (1.0 + max(0, heuristic_score) / 100)
                else:
                    heuristic_factor = 1.0

                score = exploitation + exploration * heuristic_factor

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, agent):
        """扩展一个新子节点，使用启发式选择最有前途的动作"""
        untried = self.get_untried_actions()

        if not untried:
            return None

        # 选择（并移除）列表中第一个动作（已通过启发式排序）
        action = untried.pop(0)
        # 创建新状态
        new_state = agent.fast_simulate(self.state, action)
        # 创建子节点
        child = Node(new_state, parent=self, action=action)
        self.children.append(child)

        return child

    def update(self, result):
        """更新节点统计信息（增强版）"""
        self.visits += 1
        self.value += result
        self.squared_value += result * result  # 用于方差计算

    def update_rave(self, action_key, result):
        """更新RAVE统计"""
        self.rave_visits[action_key] += 1
        self.rave_value[action_key] += result

    def get_best_child_by_visits(self):
        """根据访问次数选择最佳子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)

    def get_best_child_by_value(self):
        """根据平均价值选择最佳子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.value / max(1, c.visits))


class AdaptiveTimeManager:
    """自适应时间管理器（优化版）"""

    def __init__(self):
        self.start_time = 0
        self.time_budget = MAX_THINK_TIME
        self.move_count = 0
        self.phase = GamePhase.OPENING
        self.phase_history = []
        self.avg_decision_time = 0.5  # 历史平均决策时间

    def start_timing(self):
        self.start_time = time.time()
        self.move_count += 1

    def get_remaining_time(self):
        elapsed = time.time() - self.start_time
        return self.time_budget - elapsed

    def is_timeout(self, buffer=0.05):
        return self.get_remaining_time() < buffer

    def should_use_quick_mode(self):
        return self.get_remaining_time() < 0.2

    def get_time_budget(self, state, importance=0.5):
        """根据游戏阶段和重要性动态分配时间（优化版）"""
        # 判断游戏阶段
        piece_count = sum(1 for i in range(10) for j in range(10)
                          if state.board.chips[i][j] not in [0, '0'])

        # 更精细的阶段划分
        if piece_count < 10:
            self.phase = GamePhase.OPENING
            base_time = 0.3  # 开局快速
        elif piece_count < 20:
            self.phase = GamePhase.MIDDLE
            base_time = 0.5
        elif piece_count < 40:
            self.phase = GamePhase.MIDDLE
            base_time = 0.7
        else:
            self.phase = GamePhase.ENDGAME
            base_time = 0.8

        # 检查是否为关键状态
        if self._is_critical_state(state):
            self.phase = GamePhase.CRITICAL
            base_time = min(0.9, base_time * 1.5)  # 关键时刻分配更多时间

        # 根据重要性和历史表现调整
        time_multiplier = 1 + importance * 0.3

        # 如果之前决策很快，可以稍微延长当前决策时间
        if self.avg_decision_time < 0.3:
            time_multiplier *= 1.2

        return min(MAX_THINK_TIME, base_time * time_multiplier)

    def _is_critical_state(self, state):
        """检查是否为关键状态（优化版）"""
        board = state.board.chips
        # 检查是否有4连的情况（使用角落友好的检查）
        for i in range(10):
            for j in range(10):
                cell = board[i][j]
                if cell not in [0, '0'] or (i, j) in CORNERS:
                    # 对于角落，检查所有可能的颜色
                    colors_to_check = []
                    if (i, j) in CORNERS:
                        colors_to_check = ['r', 'b']  # 角落可以为任何颜色服务
                    elif cell not in [0, '0']:
                        colors_to_check = [cell]

                    for color in colors_to_check:
                        for dx, dy in DIRECTIONS:
                            if ActionEvaluator._count_consecutive_with_corners(board, i, j, dx, dy, color) >= 4:
                                return True
        return False

    def finish_decision(self, decision_time):
        """记录决策时间，更新历史统计"""
        # 滑动平均更新
        alpha = 0.1  # 学习率
        self.avg_decision_time = (1 - alpha) * self.avg_decision_time + alpha * decision_time


class myAgent(Agent):
    """智能体 myAgent - 增强版（修复角落bug并优化MCTS）"""

    def __init__(self, _id):
        """初始化Agent"""
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)  # 2人游戏
        self.counter = itertools.count()  # 用于搜索的唯一标识符

        self.card_evaluator = CardEvaluator(self)
        self.time_manager = AdaptiveTimeManager()  # 使用自适应时间管理

        # 玩家颜色初始化
        self.my_color = None
        self.opp_color = None

        # 搜索参数（优化后）
        self.simulation_depth = 5  # 稍微增加深度
        self.candidate_limit = 15  # 增加候选数

        # 时间控制
        self.start_time = 0

        # 新增：开局库和模式识别
        self._initialize_opening_book()
        self.move_history = []

        # 新增：性能统计
        self.performance_stats = {
            'mcts_iterations': [],
            'decision_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }

        # 新增：Transposition Table
        self.transposition_table = LRUCache(capacity=20000)

    def _initialize_opening_book(self):
        """初始化开局库（优化：包含更多变化）"""
        self.opening_book = {
            # 第一手优先抢占中心
            0: [(4, 4), (4, 5), (5, 4), (5, 5)],
            # 第二手根据对手选择（更智能的对应）
            1: {
                (4, 4): [(5, 5), (4, 5), (5, 4), (3, 3)],
                (4, 5): [(5, 4), (4, 4), (5, 5), (3, 4)],
                (5, 4): [(4, 5), (5, 5), (4, 4), (4, 3)],
                (5, 5): [(4, 4), (5, 4), (4, 5), (3, 3)]
            },
            # 第三手策略
            2: {
                'center_control': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 6), (5, 3), (5, 6), (6, 3), (6, 4), (6, 5)]
            }
        }

    def _initialize_colors(self, game_state):
        """初始化颜色信息"""
        if self.my_color is None:
            self.my_color = game_state.agents[self.id].colour
            self.opp_color = game_state.agents[1 - self.id].colour

    def _is_card_selection(self, actions):
        """判断是否为卡牌选择"""
        return any(a.get('type') == 'trade' for a in actions)

    def _select_strategic_card(self, actions, game_state):
        """增强的卡牌选择逻辑"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']

        if not hasattr(game_state, 'display_cards') or not game_state.display_cards:
            return random.choice(trade_actions) if trade_actions else None

        # 获取当前手牌
        current_hand = []
        if hasattr(game_state.agents[self.id], 'hand'):
            current_hand = game_state.agents[self.id].hand

        # 评估所有展示牌
        best_card = None
        best_score = float('-inf')

        for card in game_state.display_cards:
            # 基础价值
            immediate_score = self.card_evaluator._evaluate_card(card, game_state, consider_opponent=True)

            # 手牌多样性奖励
            diversity_bonus = self._calculate_diversity_bonus(card, current_hand)

            # 特殊牌额外加分
            if self.card_evaluator._is_two_eyed_jack(card):
                special_bonus = 5000
            elif self.card_evaluator._is_one_eyed_jack(card):
                special_bonus = 2500
            else:
                special_bonus = 0

            total_score = immediate_score + diversity_bonus + special_bonus

            if total_score > best_score:
                best_score = total_score
                best_card = card

        # 找到对应动作
        for action in trade_actions:
            if action.get('draft_card') == best_card:
                return action

        return random.choice(trade_actions) if trade_actions else None

    def _calculate_diversity_bonus(self, card, hand):
        """计算选择该卡牌对手牌多样性的贡献"""
        if not hand:
            return 100  # 第一张牌给予奖励

        # 如果手牌中已有相同卡牌，降低分数
        card_str = str(card)
        count = sum(1 for c in hand if str(c) == card_str)

        if count == 0:
            return 50  # 新卡牌类型
        elif count == 1:
            return 0  # 已有一张
        else:
            return -50 * count  # 惩罚重复

    def SelectAction(self, actions, game_state):
        """主决策函数 - 融合启发式筛选和MCTS（优化版）"""
        decision_start = time.time()
        self.time_manager.start_timing()
        self._initialize_colors(game_state)

        # BUG修复：检查actions是否为空
        if not actions:
            return None

        if self._is_card_selection(actions):
            result = self._select_strategic_card(actions, game_state)
            self.time_manager.finish_decision(time.time() - decision_start)
            return result

        # 快速获胜检测
        for action in actions:
            if ActionEvaluator.is_winning_move(game_state, action, self.my_color):
                return action

        # 阻止对手获胜检测
        for action in actions:
            if ActionEvaluator.blocks_opponent_win(game_state, action, self.opp_color):
                return action

        # 开局库查询
        move_num = len(self.move_history)
        if move_num < 3 and self._check_opening_book(actions, game_state, move_num):
            opening_move = self._get_opening_move(actions, game_state, move_num)
            if opening_move:
                self.time_manager.finish_decision(time.time() - decision_start)
                return opening_move

        # 启发式筛选候选动作（修复：正确处理角落）
        candidates = self._heuristic_filter(actions, game_state)

        # 动态时间分配
        importance = self._evaluate_move_importance(game_state, candidates)
        time_budget = self.time_manager.get_time_budget(game_state, importance)

        # 紧急模式：时间不足时的激进剪枝
        if self.time_manager.get_remaining_time() < 0.1:
            # 只考虑前3个最佳动作
            candidates = candidates[:3]
            self.simulation_depth = 2
        elif self.time_manager.get_remaining_time() < time_budget:
            decision_time = time.time() - decision_start
            self.time_manager.finish_decision(decision_time)
            return candidates[0] if candidates else random.choice(actions)

        # MCTS深度搜索
        try:
            result = self._mcts_search(candidates, game_state)
            # 记录性能统计
            decision_time = time.time() - decision_start
            self.performance_stats['decision_times'].append(decision_time)
            self.time_manager.finish_decision(decision_time)
            return result
        except Exception as e:
            # BUG修复：异常处理
            print(f"MCTS搜索异常: {e}")
            decision_time = time.time() - decision_start
            self.time_manager.finish_decision(decision_time)
            return candidates[0] if candidates else random.choice(actions)

    def _check_opening_book(self, actions, game_state, move_num):
        """检查是否可以使用开局库（扩展版）"""
        if move_num >= len(self.opening_book):
            return False

        # 第三手特殊处理
        if move_num == 2:
            recommendations = self.opening_book[2]['center_control']
        elif move_num == 0:
            recommendations = self.opening_book[0]
        else:
            last_opp_move = self._get_last_opponent_move(game_state)
            if last_opp_move and last_opp_move in self.opening_book[1]:
                recommendations = self.opening_book[1][last_opp_move]
            else:
                return False

        # 检查是否有可用的推荐位置
        for r, c in recommendations:
            for action in actions:
                if action.get('coords') == (r, c):
                    return True
        return False

    def _get_opening_move(self, actions, game_state, move_num):
        """从开局库获取动作（扩展版）"""
        if move_num == 2:
            recommendations = self.opening_book[2]['center_control']
        elif move_num == 0:
            recommendations = self.opening_book[0]
        else:
            last_opp_move = self._get_last_opponent_move(game_state)
            if not last_opp_move:
                return None
            recommendations = self.opening_book[1].get(last_opp_move, [])

        for r, c in recommendations:
            for action in actions:
                if action.get('coords') == (r, c):
                    return action

        return None

    def _get_last_opponent_move(self, game_state):
        """获取对手最后一步棋的位置"""
        # 这里需要根据实际游戏状态追踪
        # 简化处理：扫描棋盘找到对手的棋子
        board = game_state.board.chips
        for r, c in HOTB_COORDS:
            if board[r][c] == self.opp_color:
                return (r, c)
        return None

    def _evaluate_move_importance(self, game_state, candidates):
        """评估当前移动的重要性（优化版）"""
        if not candidates:
            return 0.5

        # 获取最佳候选的评分
        best_score = ActionEvaluator.evaluate_action_quality(game_state, candidates[0])

        # 检查是否存在紧急情况
        board = game_state.board.chips
        urgent_defense = False
        for i in range(10):
            for j in range(10):
                if board[i][j] == self.opp_color or (i, j) in CORNERS:
                    for dx, dy in DIRECTIONS:
                        if ActionEvaluator._count_consecutive_with_corners(board, i, j, dx, dy, self.opp_color) >= 4:
                            urgent_defense = True
                            break
                if urgent_defense:
                    break
            if urgent_defense:
                break

        if urgent_defense:
            return 1.0  # 紧急防守

        # 评分越低（质量越高），重要性越大
        if best_score < -800:  # 极高质量动作
            return 0.95
        elif best_score < -200:  # 高质量动作
            return 0.8
        elif best_score < 0:  # 中等质量
            return 0.6
        elif best_score < 50:  # 一般质量
            return 0.4
        else:  # 低质量
            return 0.2

    def _heuristic_filter(self, actions, game_state):
        """使用启发式评估筛选最有前途的动作（优化版，修复角落bug）"""
        # 修复：不排除角落位置，角落可能是合法的移除目标
        valid_actions = []
        for a in actions:
            if a.get('type') == 'remove':  # 移除动作不过滤（但角落不能移除）
                if 'coords' in a and a['coords'] not in CORNERS:
                    valid_actions.append(a)
            elif a.get('type') == 'place':  # 放置动作（角落不能放置）
                if 'coords' not in a or a['coords'] not in CORNERS:
                    valid_actions.append(a)
            else:  # 其他动作
                valid_actions.append(a)

        if not valid_actions:
            return actions[:1]  # 如果没有有效动作，返回第一个动作

        # 批量评估并排序
        scored_actions = [(a, ActionEvaluator.evaluate_action_quality(game_state, a)) for a in valid_actions]
        scored_actions.sort(key=lambda x: x[1])

        # 动态调整候选数量
        if self.time_manager.phase == GamePhase.CRITICAL:
            limit = min(25, len(scored_actions))  # 关键时刻考虑更多选项
        elif self.time_manager.get_remaining_time() < 0.3:
            limit = min(8, len(scored_actions))  # 时间紧张时减少候选
        else:
            limit = self.candidate_limit

        # 返回前N个候选动作
        return [a for a, _ in scored_actions[:limit]]

    def _mcts_search(self, candidate_actions, game_state):
        """使用MCTS分析候选动作（优化版）"""
        if not candidate_actions:
            return None

        # 检查Transposition Table
        board_hash = hash(tuple(tuple(row) for row in game_state.board.chips))
        cached_result = self.transposition_table.get(board_hash)
        if cached_result:
            self.performance_stats['cache_hits'] += 1
            # 验证缓存的动作是否仍然有效
            if cached_result in candidate_actions:
                return cached_result

        self.performance_stats['cache_misses'] += 1

        # 准备MCTS状态
        mcts_state = self._prepare_state_for_mcts(game_state, candidate_actions)
        root = Node(mcts_state)

        # 直接为根节点创建子节点
        for action in candidate_actions:
            next_state = self.fast_simulate(mcts_state, action)
            child = Node(next_state, parent=root, action=action)
            root.children.append(child)

        # 动态设置搜索深度和迭代次数
        if self.time_manager.phase == GamePhase.CRITICAL:
            max_iterations = min(SIMULATION_LIMIT * 2, 500)
            self.simulation_depth = 6
        elif self.time_manager.get_remaining_time() < 0.3:
            max_iterations = 80  # 时间紧张时快速决策
            self.simulation_depth = 3
        else:
            max_iterations = SIMULATION_LIMIT
            self.simulation_depth = 5

        # MCTS主循环
        iterations = 0
        time_check_interval = 10  # 每10次迭代检查一次时间

        # 新增：早期终止相关变量
        dominant_threshold = 0.75  # 如果某个选择占75%以上访问，可以提前终止
        convergence_check_interval = 50  # 每50次迭代检查收敛性

        while not self.time_manager.is_timeout() and iterations < max_iterations:
            iterations += 1

            # 1. 选择阶段（使用改进的UCB）
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.select_child(use_variance=True)
                if node is None:  # BUG修复：检查空节点
                    break

            if node is None:
                continue

            # 2. 扩展阶段
            if node.visits > 0 and not node.is_fully_expanded():
                child = node.expand(self)
                if child:
                    node = child

            # 3. 模拟阶段（使用增强的模拟策略）
            value = self._enhanced_simulation(node.state)

            # 4. 回溯阶段
            while node:
                node.update(value)
                node = node.parent

            # 定期检查
            if iterations % time_check_interval == 0:
                # 时间检查
                if self.time_manager.is_timeout():
                    break

            # 收敛性检查
            if iterations % convergence_check_interval == 0 and iterations > convergence_check_interval:
                if self._check_convergence(root, dominant_threshold):
                    break

        # 记录迭代次数
        self.performance_stats['mcts_iterations'].append(iterations)

        # 选择最佳动作
        if not root.children:
            return candidate_actions[0] if candidate_actions else None

        # 选择策略：在后期游戏使用更保守的选择
        if self.time_manager.phase in [GamePhase.CRITICAL, GamePhase.ENDGAME]:
            best_child = root.get_best_child_by_value()  # 选择平均价值最高的
        else:
            best_child = root.get_best_child_by_visits()  # 选择访问次数最多的

        if not best_child:
            return candidate_actions[0]

        # 缓存结果
        self.transposition_table.put(board_hash, best_child.action)

        # 记录移动历史
        if best_child.action and 'coords' in best_child.action:
            self.move_history.append(best_child.action['coords'])

        return best_child.action

    def _check_convergence(self, root, threshold):
        """检查MCTS是否收敛（某个选择明显占优）"""
        if not root.children:
            return False

        total_visits = sum(child.visits for child in root.children)
        if total_visits == 0:
            return False

        max_visits = max(child.visits for child in root.children)
        return (max_visits / total_visits) > threshold

    def _enhanced_simulation(self, state):
        """增强的MCTS模拟（更智能的策略）"""
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

            # 根据游戏阶段和深度调整策略
            if self.time_manager.phase == GamePhase.CRITICAL:
                heuristic_prob = 0.95  # 关键时刻更依赖启发式
            elif current_depth == 1:
                heuristic_prob = 0.9  # 第一步更重要
            elif current_depth <= 2:
                heuristic_prob = 0.85
            else:
                heuristic_prob = 0.7  # 深层次降低启发式依赖

            # 选择动作
            if random.random() < heuristic_prob:
                # 使用启发式选择动作（增强版）
                action = self._smart_action_selection(actions, state_copy)
            else:
                action = random.choice(actions)

            # 应用动作
            state_copy = self.fast_simulate(state_copy, action)

            # 检查是否达到终止状态（获胜）
            if self._check_terminal_state(state_copy):
                break

            # 模拟卡牌选择
            self._simulate_card_selection(state_copy)

        # 评估最终状态
        return StateEvaluator.evaluate(state_copy)

    def _smart_action_selection(self, actions, state):
        """智能动作选择（在模拟中使用）"""
        # 快速检查获胜动作
        for action in actions:
            if ActionEvaluator.is_winning_move(state, action, self.my_color):
                return action

        # 快速检查阻止对手获胜
        for action in actions:
            if ActionEvaluator.blocks_opponent_win(state, action, self.opp_color):
                return action

        # 启发式评估前几个动作
        sample_size = min(8, len(actions))
        if len(actions) <= sample_size:
            sampled_actions = actions
        else:
            # 随机采样但偏向前面的动作（假设已经过某种排序）
            sampled_actions = actions[:sample_size // 2] + random.sample(actions[sample_size // 2:], sample_size // 2)

        scored_actions = [(a, ActionEvaluator.evaluate_action_quality(state, a)) for a in sampled_actions]
        scored_actions.sort(key=lambda x: x[1])

        # 在前25%的动作中随机选择，增加多样性
        top_count = max(1, len(scored_actions) // 4)
        return random.choice(scored_actions[:top_count])[0]

    def _heuristic_guided_simulate(self, state):
        """向后兼容：使用新的增强模拟方法"""
        return self._enhanced_simulation(state)

    def _check_terminal_state(self, state):
        """检查是否达到终止状态（优化版，正确处理角落）"""
        board = state.board.chips
        # 检查是否有5连（使用角落友好的检查）
        for i in range(10):
            for j in range(10):
                cell = board[i][j]
                if cell not in [0, '0'] or (i, j) in CORNERS:
                    # 对角落，检查所有可能颜色
                    colors_to_check = []
                    if (i, j) in CORNERS:
                        colors_to_check = ['r', 'b']
                    elif cell not in [0, '0']:
                        colors_to_check = [cell]

                    for color in colors_to_check:
                        for dx, dy in DIRECTIONS:
                            if ActionEvaluator._count_consecutive_with_corners(board, i, j, dx, dy, color) >= 5:
                                return True
        return False

    def fast_simulate(self, state, action):
        """快速模拟执行动作（优化版，正确处理角落）"""
        new_state = self.custom_shallow_copy(state)

        # 处理放置动作
        if action['type'] == 'place' and 'coords' in action:
            r, c = action['coords']

            # 修复：角落不能放置棋子（角落是自由点，不是放置目标）
            if (r, c) in CORNERS:
                return new_state  # 角落不能放置，返回原状态

            # 检查位置是否已被占用
            if new_state.board.chips[r][c] not in [0, '0']:
                return new_state  # 位置已被占用

            # 确定正确的颜色
            if hasattr(state, 'current_player_id') and hasattr(state, 'agents'):
                color = state.agents[state.current_player_id].colour
            else:
                color = self.my_color
            new_state.board.chips[r][c] = color
            self._update_hand(new_state, action)

        # 处理移除动作
        elif action['type'] == 'remove' and 'coords' in action:
            r, c = action['coords']

            # 修复：角落不能移除（角落是自由点，不是棋子）
            if (r, c) in CORNERS:
                return new_state  # 角落不能移除

            # 检查是否有棋子可移除
            if new_state.board.chips[r][c] in [0, '0']:
                return new_state  # 没有棋子可移除

            new_state.board.chips[r][c] = 0
            self._update_hand(new_state, action)

        return new_state

    def _update_hand(self, state, action):
        """更新手牌"""
        if 'play_card' not in action:
            return
        card = action['play_card']
        try:
            player_id = getattr(state, 'current_player_id', self.id)
            if (hasattr(state, 'agents') and
                    0 <= player_id < len(state.agents) and
                    hasattr(state.agents[player_id], 'hand')):
                if card in state.agents[player_id].hand:  # BUG修复：检查卡牌是否存在
                    state.agents[player_id].hand.remove(card)
        except Exception:
            pass

    def _simulate_card_selection(self, state):
        """模拟从5张展示牌中选择一张 - 使用增强评估"""
        if not (hasattr(state, 'display_cards') and state.display_cards):
            return

        # 使用相同的评估逻辑
        best_card = None
        best_value = float('-inf')

        for card in state.display_cards:
            value = self.card_evaluator._evaluate_card(card, state)
            if value > best_value:
                best_value = value
                best_card = card

        if best_card:
            # 更新玩家手牌
            if hasattr(state, 'current_player_id'):
                player_id = state.current_player_id
                if 0 <= player_id < len(state.agents) and hasattr(state.agents[player_id], 'hand'):
                    state.agents[player_id].hand.append(best_card)

            # 从展示区移除所选卡牌
            if best_card in state.display_cards:  # BUG修复：检查卡牌是否存在
                state.display_cards.remove(best_card)

            # 补充一张牌（如果有牌堆）
            if hasattr(state, 'deck') and state.deck:
                state.display_cards.append(state.deck.pop(0))

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

    def custom_shallow_copy(self, state):
        """优化的状态复制（关键优化）"""
        if hasattr(state, 'board') and hasattr(state.board, 'chips'):
            # 高效的浅拷贝 + 关键部分深拷贝
            new_state = copy.copy(state)
            new_state.board = copy.copy(state.board)
            new_state.board.chips = [row[:] for row in state.board.chips]

            # 复制agents列表（如果存在）
            if hasattr(state, 'agents'):
                new_state.agents = list(state.agents)

            return new_state
        else:
            # 回退到深拷贝
            return copy.deepcopy(state)