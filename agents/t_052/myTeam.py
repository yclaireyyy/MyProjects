from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import random
import time
import copy
import math
import itertools
from collections import defaultdict

# ===========================
# 1. 常量定义区
# ===========================
MAX_THINK_TIME = 0.95  # 最大思考时间（秒）
EXPLORATION_WEIGHT = 1.2  # UCB公式中的探索参数
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]  # 中心热点位置
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]  # 角落位置（自由点）
SIMULATION_LIMIT = 200  # MCTS模拟的最大次数（提升）

# 预计算的方向向量和位置权重
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
POSITION_WEIGHTS = {}  # 位置权重缓存
ACTION_CACHE = {}  # 动作评估缓存

# 初始化位置权重
for i in range(10):
    for j in range(10):
        if (i, j) in HOTB_COORDS:
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
        self._card_cache = {}  # 卡牌评估缓存
        self._hand_diversity_cache = {}  # 手牌多样性缓存

    def _evaluate_card(self, card, state, consider_opponent=True):
        """评估卡牌在当前状态下的价值（增强版）"""
        # 生成缓存键
        board_hash = state.board.chips if hasattr(state.board, 'get_hash') else hash(
            tuple(tuple(row) for row in state.board.chips))
        cache_key = (str(card), board_hash, consider_opponent)

        if cache_key in self._card_cache:
            return self._card_cache[cache_key]

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

        self._card_cache[cache_key] = result
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
            r, c = pos
            # 检查位置是否可用
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
        """统计特定方向5个位置内的我方棋子数量（优化版）"""
        my_pieces = 0
        my_color = self.agent.my_color

        # 检查该方向前后各4个位置（共8个位置）
        for i in range(-4, 5):
            # 跳过中心位置（即将放置的位置）
            if i == 0:
                continue
            x, y = r + i * dx, c + i * dy
            # 边界检查和颜色检查合并
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == my_color:
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
        """检查位置是否可用"""
        if not (0 <= r < 10 and 0 <= c < 10):
            return False
        return board[r][c] == 0 or board[r][c] == '0'  # 空位


class ActionEvaluator:
    _evaluation_cache = {}  # 静态缓存
    _threat_cache = {}  # 威胁检测缓存

    @staticmethod
    def evaluate_action_quality(state, action):
        """评估动作的质量得分（越低越好）- 增强版"""
        if action.get('type') != 'place' or 'coords' not in action:
            return 100  # 非放置动作或无坐标

        r, c = action['coords']
        if (r, c) in CORNERS:
            return 100  # 角落位置

        # 生成缓存键
        board_hash = hash(tuple(tuple(row) for row in state.board.chips))
        cache_key = (board_hash, r, c)

        if cache_key in ActionEvaluator._evaluation_cache:
            return ActionEvaluator._evaluation_cache[cache_key]

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

        # 连续链评分（优化版）
        for dx, dy in DIRECTIONS:
            count = ActionEvaluator._count_consecutive_fast(board, r, c, dx, dy, color)
            # 根据连续长度评分
            if count >= 5:
                score += 200  # 形成序列
            elif count == 4:
                score += 100
            elif count == 3:
                score += 30
            elif count == 2:
                score += 10

        # 增强防守评分
        critical_block = False
        for dx, dy in DIRECTIONS:
            enemy_chain = ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy)
            if enemy_chain >= 4:
                score += 500  # 极高优先级阻断
                critical_block = True
            elif enemy_chain >= 3:
                score += 150  # 高优先级阻断

            # 检测双向威胁
            if ActionEvaluator._check_double_threat(board, r, c, dx, dy, enemy):
                score += 300

        # 特殊情况：如果是关键防守，大幅提升得分
        if critical_block:
            score *= 2

        # 中心控制评分（使用位置权重）
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if (x, y) == (r, c))
        score += hotb_controlled * 15

        # 转换为质量分数（越低越好）
        result = 100 - score
        ActionEvaluator._evaluation_cache[cache_key] = result
        return result

    @staticmethod
    def _check_double_threat(board, r, c, dx, dy, enemy):
        """检查是否存在双向威胁"""
        # 检查两端是否都可以延伸
        threats = 0

        # 正向检查
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == 0:
                    threats += 1
                    break
                elif board[x][y] != enemy:
                    break

        # 反向检查
        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == 0:
                    threats += 1
                    break
                elif board[x][y] != enemy:
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

        # 检查是否形成序列
        for dx, dy in DIRECTIONS:
            if ActionEvaluator._count_consecutive_fast(board_copy, r, c, dx, dy, color) >= 5:
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
            if ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy) >= 4:
                return True
        return False

    @staticmethod
    def _count_consecutive_fast(board, x, y, dx, dy, color):
        """优化的连续计算"""
        count = 1  # 起始位置算一个

        # 正向检查
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < 10 and 0 <= ny < 10 and board[nx][ny] == color:
                count += 1
            else:
                break

        # 反向检查
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if 0 <= nx < 10 and 0 <= ny < 10 and board[nx][ny] == color:
                count += 1
            else:
                break

        return min(count, 5)  # 最多返回5（形成一个序列）

    @staticmethod
    def _count_enemy_threat_fast(board, r, c, dx, dy, enemy):
        """优化的敌方威胁计算"""
        enemy_chain = 0
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
                enemy_chain += 1
            else:
                break
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
                enemy_chain += 1
            else:
                break
        return enemy_chain

    @staticmethod
    def _calculate_action_score(board, r, c, color, enemy):
        """计算动作分数"""
        score = 0

        # 创建假设棋盘
        board_copy = [row[:] for row in board]
        board_copy[r][c] = color

        # 中心偏好
        distance = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 5 - distance) * 2

        # 连续链评分
        for dx, dy in DIRECTIONS:
            count = ActionEvaluator._count_consecutive_fast(board_copy, r, c, dx, dy, color)
            if count >= 5:
                score += 200
            elif count == 4:
                score += 100
            elif count == 3:
                score += 30
            elif count == 2:
                score += 10

        # 防守评分
        for dx, dy in DIRECTIONS:
            enemy_threat = ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy)
            if enemy_threat >= 4:
                score += 500
            elif enemy_threat >= 3:
                score += 150

        # 中心控制
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if board_copy[x][y] == color)
        score += hotb_controlled * 15

        return score

    @staticmethod
    def _count_consecutive(board, x, y, dx, dy, color):
        """向后兼容的方法"""
        return ActionEvaluator._count_consecutive_fast(board, x, y, dx, dy, color)

    @staticmethod
    def _count_enemy_threat(board, r, c, dx, dy, enemy):
        """向后兼容的方法"""
        return ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy)


class StateEvaluator:
    _state_cache = {}  # 状态评估缓存

    @staticmethod
    def evaluate(state, last_action=None):
        """评估游戏状态的价值（带缓存）"""
        # 生成缓存键
        board_hash = hash(tuple(tuple(row) for row in state.board.chips))
        if board_hash in StateEvaluator._state_cache:
            return StateEvaluator._state_cache[board_hash]

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

        # 2. 序列潜力评分（批量处理）
        sequence_score = StateEvaluator._calculate_sequence_score_fast(board, my_color)

        # 3. 防御评分 - 阻止对手的序列
        defense_score = StateEvaluator._calculate_defense_score_fast(board, opp_color)

        # 4. 中心控制评分
        hotb_score = 0
        for x, y in HOTB_COORDS:
            cell = board[x][y]
            if cell == my_color:
                hotb_score += 5
            elif cell == opp_color:
                hotb_score -= 5

        # 5. 综合评分
        total_score = position_score + sequence_score + defense_score + hotb_score

        # 归一化到[-1, 1]区间
        result = max(-1, min(1, total_score / 200))
        StateEvaluator._state_cache[board_hash] = result
        return result

    @staticmethod
    def _calculate_sequence_score_fast(board, color):
        """优化的序列得分计算"""
        sequence_score = 0
        for i in range(10):
            for j in range(10):
                if board[i][j] == color:
                    for dx, dy in DIRECTIONS:
                        count = ActionEvaluator._count_consecutive_fast(board, i, j, dx, dy, color)
                        if count >= 5:
                            sequence_score += 100
                        elif count == 4:
                            sequence_score += 20
                        elif count == 3:
                            sequence_score += 5
                        elif count == 2:
                            sequence_score += 1
        return sequence_score

    @staticmethod
    def _calculate_defense_score_fast(board, opp_color):
        """优化的防御得分计算"""
        defense_score = 0
        for i in range(10):
            for j in range(10):
                if board[i][j] == opp_color:
                    for dx, dy in DIRECTIONS:
                        count = ActionEvaluator._count_consecutive_fast(board, i, j, dx, dy, opp_color)
                        if count >= 4:
                            defense_score -= 50
                        elif count == 3:
                            defense_score -= 10
        return defense_score

    @staticmethod
    def _calculate_sequence_score(board, color):
        """向后兼容的方法"""
        return StateEvaluator._calculate_sequence_score_fast(board, color)

    @staticmethod
    def _calculate_defense_score(board, opp_color):
        """向后兼容的方法"""
        return StateEvaluator._calculate_defense_score_fast(board, opp_color)


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
        color = self._get_current_color(state)
        state.board.chips[r][c] = color

        # 更新手牌
        self._update_hand(state, action)

    def _simulate_remove(self, state, action):
        """模拟移除动作"""
        if 'coords' not in action:
            return

        r, c = action['coords']
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
    MCTS搜索树节点，集成启发式评估
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
        # 动作管理（延迟初始化）
        self.untried_actions = None

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

    def select_child(self):
        """使用UCB公式选择最有希望的子节点（优化版）"""
        best_score = float('-inf')
        best_child = None
        log_visits = math.log(self.visits)

        for child in self.children:
            # UCB计算
            if child.visits == 0:
                score = float('inf')
            else:
                # 结合启发式评估的UCB计算
                exploitation = child.value / child.visits
                exploration = EXPLORATION_WEIGHT * math.sqrt(2 * log_visits / child.visits)
                # 启发式调整（缓存友好）
                if child.action:
                    heuristic_score = ActionEvaluator.evaluate_action_quality(self.state, child.action)
                    heuristic_factor = 1.0 / (1.0 + heuristic_score / 100)
                else:
                    heuristic_factor = 1.0

                score = exploitation + exploration * heuristic_factor

            # 更新最佳节点
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
        """更新节点统计信息"""
        self.visits += 1
        self.value += result


class AdaptiveTimeManager(TimeManager):
    """自适应时间管理器"""

    def __init__(self):
        super().__init__()
        self.move_count = 0
        self.phase = GamePhase.OPENING

    def start_timing(self):
        super().start_timing()
        self.move_count += 1

    def get_time_budget(self, state, importance=0.5):
        """根据游戏阶段和重要性动态分配时间"""
        # 判断游戏阶段
        piece_count = sum(1 for i in range(10) for j in range(10)
                          if state.board.chips[i][j] not in [0, '0'])

        if piece_count < 15:
            self.phase = GamePhase.OPENING
            base_time = 0.4
        elif piece_count < 35:
            self.phase = GamePhase.MIDDLE
            base_time = 0.6
        else:
            self.phase = GamePhase.ENDGAME
            base_time = 0.8

        # 检查是否为关键状态
        if self._is_critical_state(state):
            self.phase = GamePhase.CRITICAL
            base_time = 0.9

        # 根据重要性调整
        return min(MAX_THINK_TIME, base_time * (1 + importance * 0.5))

    def _is_critical_state(self, state):
        """检查是否为关键状态"""
        board = state.board.chips
        # 检查是否有4连的情况
        for i in range(10):
            for j in range(10):
                if board[i][j] not in [0, '0']:
                    color = board[i][j]
                    for dx, dy in DIRECTIONS:
                        if ActionEvaluator._count_consecutive_fast(board, i, j, dx, dy, color) >= 4:
                            return True
        return False


class myAgent(Agent):
    """智能体 myAgent - 增强版"""

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
        self.simulation_depth = 4  # 稍微减少深度但提高质量
        self.candidate_limit = 12  # 稍微增加候选数

        # 时间控制
        self.start_time = 0

        # 新增：开局库和模式识别
        self._initialize_opening_book()
        self.move_history = []

    def _initialize_opening_book(self):
        """初始化开局库"""
        self.opening_book = {
            # 第一手优先抢占中心
            0: [(4, 4), (4, 5), (5, 4), (5, 5)],
            # 第二手根据对手选择
            1: {
                (4, 4): [(5, 5), (4, 5), (5, 4)],
                (4, 5): [(5, 4), (4, 4), (5, 5)],
                (5, 4): [(4, 5), (5, 5), (4, 4)],
                (5, 5): [(4, 4), (5, 4), (4, 5)]
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

        if not hasattr(game_state, 'display_cards'):
            return random.choice(trade_actions)

        # 获取当前手牌
        current_hand = game_state.agents[self.id].hand if hasattr(game_state.agents[self.id], 'hand') else []

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

        return random.choice(trade_actions)

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
        """主决策函数 - 融合启发式筛选和MCTS"""
        self.time_manager.start_timing()
        self._initialize_colors(game_state)

        if self._is_card_selection(actions):
            return self._select_strategic_card(actions, game_state)

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
        if move_num < 2 and self._check_opening_book(actions, game_state, move_num):
            return self._get_opening_move(actions, game_state, move_num)

        # 启发式筛选候选动作
        candidates = self._heuristic_filter(actions, game_state)

        # 动态时间分配
        importance = self._evaluate_move_importance(game_state, candidates)
        time_budget = self.time_manager.get_time_budget(game_state, importance)

        # 时间检查
        if self.time_manager.get_remaining_time() < time_budget:
            return candidates[0] if candidates else random.choice(actions)

        # MCTS深度搜索
        try:
            return self._mcts_search(candidates, game_state)
        except:
            return candidates[0] if candidates else random.choice(actions)

    def _check_opening_book(self, actions, game_state, move_num):
        """检查是否可以使用开局库"""
        if move_num >= len(self.opening_book):
            return False

        # 检查推荐位置是否可用
        if move_num == 0:
            recommendations = self.opening_book[0]
        else:
            last_opp_move = self._get_last_opponent_move(game_state)
            if last_opp_move in self.opening_book[1]:
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
        """从开局库获取动作"""
        if move_num == 0:
            recommendations = self.opening_book[0]
        else:
            last_opp_move = self._get_last_opponent_move(game_state)
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
        """评估当前移动的重要性"""
        if not candidates:
            return 0.5

        # 获取最佳候选的评分
        best_score = ActionEvaluator.evaluate_action_quality(game_state, candidates[0])

        # 评分越低（质量越高），重要性越大
        if best_score < -100:  # 极高质量动作
            return 0.9
        elif best_score < 0:  # 高质量动作
            return 0.7
        elif best_score < 50:  # 中等质量
            return 0.5
        else:  # 低质量
            return 0.3

    def _heuristic_filter(self, actions, game_state):
        """使用启发式评估筛选最有前途的动作（优化版）"""
        # 排除角落位置
        valid_actions = [a for a in actions if 'coords' not in a or a['coords'] not in CORNERS]
        if not valid_actions:
            return actions[:1]  # 如果没有有效动作，返回第一个动作

        # 批量评估并排序
        scored_actions = [(a, ActionEvaluator.evaluate_action_quality(game_state, a)) for a in valid_actions]
        scored_actions.sort(key=lambda x: x[1])

        # 动态调整候选数量
        if self.time_manager.phase == GamePhase.CRITICAL:
            limit = min(20, len(scored_actions))  # 关键时刻考虑更多选项
        else:
            limit = self.candidate_limit

        # 返回前N个候选动作
        return [a for a, _ in scored_actions[:limit]]

    def _mcts_search(self, candidate_actions, game_state):
        """使用MCTS分析候选动作（优化版）"""
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
            max_iterations = min(SIMULATION_LIMIT * 2, 400)
            self.simulation_depth = 6
        else:
            max_iterations = SIMULATION_LIMIT
            self.simulation_depth = 4

        # MCTS主循环
        iterations = 0
        time_check_interval = 10  # 每10次迭代检查一次时间

        while not self.time_manager.is_timeout() and iterations < max_iterations:
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
            value = self._heuristic_guided_simulate(node.state)

            # 4. 回溯阶段
            while node:
                node.update(value)
                node = node.parent

            # 定期时间检查（减少系统调用）
            if iterations % time_check_interval == 0 and self.time_manager.is_timeout():
                break

        # 选择最佳动作（访问次数最多的子节点）
        if not root.children:
            return candidate_actions[0] if candidate_actions else None

        # 选择访问次数最多的子节点
        best_child = max(root.children, key=lambda c: c.visits)

        # 记录移动历史
        if best_child.action and 'coords' in best_child.action:
            self.move_history.append(best_child.action['coords'])

        return best_child.action

    def _heuristic_guided_simulate(self, state):
        """启发式引导的MCTS模拟（优化版）"""
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

            # 根据游戏阶段调整启发式比例
            if self.time_manager.phase == GamePhase.CRITICAL:
                heuristic_prob = 0.95  # 关键时刻更依赖启发式
            else:
                heuristic_prob = 0.85

            # 选择动作
            if random.random() < heuristic_prob:
                # 使用启发式选择动作
                scored_actions = [(a, ActionEvaluator.evaluate_action_quality(state_copy, a)) for a in actions]
                scored_actions.sort(key=lambda x: x[1])
                action = scored_actions[0][0] if scored_actions else random.choice(actions)
            else:
                action = random.choice(actions)

            # 应用动作
            state_copy = self.fast_simulate(state_copy, action)

            # 模拟卡牌选择（专门针对5张展示牌变体）
            self._simulate_card_selection(state_copy)

        # 评估最终状态
        return StateEvaluator.evaluate(state_copy)

    def fast_simulate(self, state, action):
        """快速模拟执行动作（优化版）"""
        new_state = self.custom_shallow_copy(state)

        # 处理放置动作
        if action['type'] == 'place' and 'coords' in action:
            r, c = action['coords']
            color = self.my_color
            if hasattr(state, 'current_player_id'):
                color = state.agents[state.current_player_id].colour
            new_state.board.chips[r][c] = color
            self._update_hand(new_state, action)

        # 处理移除动作
        elif action['type'] == 'remove' and 'coords' in action:
            r, c = action['coords']
            new_state.board.chips[r][c] = 0
            self._update_hand(new_state, action)

        return new_state

    def _update_hand(self, state, action):
        """更新手牌"""
        if 'play_card' not in action:
            return
        card = action['play_card']
        try:
            if (hasattr(state, 'agents') and
                    hasattr(state.agents[self.id], 'hand')):
                state.agents[self.id].hand.remove(card)
        except:
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
                if 0 <= player_id < len(state.agents):
                    state.agents[player_id].hand.append(best_card)

            # 从展示区移除所选卡牌
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