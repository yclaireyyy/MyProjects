from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import random
import time
import copy
import math
import itertools
from collections import defaultdict, deque

# ===========================
# 1. 常量定义区（保持不变）
# ===========================
MAX_THINK_TIME = 0.95
EXPLORATION_WEIGHT = 1.414
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]
SIMULATION_LIMIT = 300

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
POSITION_WEIGHTS = {}
ACTION_CACHE = {}

# 初始化位置权重（保持不变）
for i in range(10):
    for j in range(10):
        if (i, j) in CORNERS:
            POSITION_WEIGHTS[(i, j)] = 2.0
        elif (i, j) in HOTB_COORDS:
            POSITION_WEIGHTS[(i, j)] = 1.5
        elif i in [0, 9] or j in [0, 9]:
            POSITION_WEIGHTS[(i, j)] = 0.8
        else:
            POSITION_WEIGHTS[(i, j)] = 1.0


# ===========================
# 2. 游戏阶段定义（保持不变）
# ===========================
class GamePhase:
    OPENING = "opening"
    MIDDLE = "middle"
    CRITICAL = "critical"
    ENDGAME = "endgame"


# ===========================
# 3. 优化的缓存系统
# ===========================
class IntelligentCache:
    """多级智能缓存系统 - 根据数据类型优化缓存策略"""

    def __init__(self):
        # 按数据特性分类的缓存
        self.position_cache = {}  # 位置评估缓存（使用简单dict，访问更快）
        self.sequence_cache = {}  # 序列识别缓存
        self.board_pattern_cache = {}  # 棋盘模式缓存

        # 缓存统计和管理
        self.cache_size_limits = {
            'position': 8000,
            'sequence': 5000,
            'pattern': 3000
        }
        self.access_order = {
            'position': deque(),
            'sequence': deque(),
            'pattern': deque()
        }

    def get_position_score(self, key):
        """获取位置评分缓存"""
        if key in self.position_cache:
            self._update_access('position', key)
            return self.position_cache[key]
        return None

    def set_position_score(self, key, value):
        """设置位置评分缓存"""
        self._manage_cache_size('position')
        self.position_cache[key] = value
        self.access_order['position'].append(key)

    def get_board_features(self, board_hash):
        """获取棋盘特征缓存"""
        if board_hash in self.board_pattern_cache:
            self._update_access('pattern', board_hash)
            return self.board_pattern_cache[board_hash]
        return None

    def set_board_features(self, board_hash, features):
        """设置棋盘特征缓存"""
        self._manage_cache_size('pattern')
        self.board_pattern_cache[board_hash] = features
        self.access_order['pattern'].append(board_hash)

    def _update_access(self, cache_type, key):
        """更新访问顺序（简化的LRU）"""
        order_queue = self.access_order[cache_type]
        if key in order_queue:
            # 简化操作：不移动位置，只在满时清理旧项
            pass
        else:
            order_queue.append(key)

    def _manage_cache_size(self, cache_type):
        """管理缓存大小，防止内存溢出"""
        cache_dict = getattr(self, f'{cache_type}_cache')
        limit = self.cache_size_limits[cache_type]
        order_queue = self.access_order[cache_type]

        while len(cache_dict) >= limit and order_queue:
            old_key = order_queue.popleft()
            cache_dict.pop(old_key, None)

    def clear_all(self):
        """清空所有缓存"""
        self.position_cache.clear()
        self.sequence_cache.clear()
        self.board_pattern_cache.clear()
        for queue in self.access_order.values():
            queue.clear()


# 全局缓存实例
GLOBAL_CACHE = IntelligentCache()


# ===========================
# 4. 轻量级棋盘状态
# ===========================
class LightweightBoardState:
    """轻量级棋盘状态 - 减少复制开销"""

    def __init__(self, board_data, is_reference=False):
        if is_reference:
            self.chips = board_data  # 直接引用，不复制
            self._is_copy = False
        else:
            self.chips = [row[:] for row in board_data]  # 浅拷贝
            self._is_copy = True
        self._hash = None

    def get_hash(self):
        """快速哈希计算 - 只考虑关键位置"""
        if self._hash is None:
            # 只计算中心区域和边角的哈希，减少计算量
            key_positions = HOTB_COORDS + CORNERS
            key_positions += [(i, j) for i in [0, 9] for j in range(10)]  # 边界
            key_values = []
            for i, j in key_positions:
                if 0 <= i < 10 and 0 <= j < 10:
                    key_values.append(self.chips[i][j])
            self._hash = hash(tuple(key_values))
        return self._hash

    def copy(self):
        """高效复制"""
        return LightweightBoardState(self.chips, is_reference=False)

    def __getitem__(self, key):
        return self.chips[key]


# ===========================
# 5. 优化的卡牌评估器
# ===========================
class OptimizedCardEvaluator:
    """优化的卡牌评估器 - 使用批处理和智能缓存"""

    def __init__(self, agent):
        self.agent = agent

    def evaluate_card_batch(self, cards, state):
        """批量评估多张卡牌 - 共享计算提升效率"""
        if not cards:
            return {}

        # 预计算棋盘的全局特征，所有卡牌共享使用
        board_features = self._compute_board_features_once(state)

        results = {}
        for card in cards:
            results[card] = self._evaluate_single_card(card, state, board_features)

        return results

    def _compute_board_features_once(self, state):
        """一次性计算棋盘特征，供多个卡牌评估使用"""
        board = state.board.chips
        board_hash = self._get_board_hash(board)

        # 尝试从缓存获取
        cached_features = GLOBAL_CACHE.get_board_features(board_hash)
        if cached_features:
            return cached_features

        # 计算新特征
        features = {
            'my_positions': [],
            'opp_positions': [],
            'empty_near_pieces': set(),  # 棋子附近的空位
            'center_control': 0,
            'edge_positions': []
        }

        for i in range(10):
            for j in range(10):
                cell = board[i][j]
                if cell == self.agent.my_color:
                    features['my_positions'].append((i, j))
                    self._add_adjacent_empty(features['empty_near_pieces'], board, i, j)
                elif cell == self.agent.opp_color:
                    features['opp_positions'].append((i, j))
                    self._add_adjacent_empty(features['empty_near_pieces'], board, i, j)
                elif cell in [0, '0']:
                    if i in [0, 9] or j in [0, 9]:
                        features['edge_positions'].append((i, j))

        # 计算中心控制
        for pos in HOTB_COORDS:
            if board[pos[0]][pos[1]] == self.agent.my_color:
                features['center_control'] += 1
            elif board[pos[0]][pos[1]] == self.agent.opp_color:
                features['center_control'] -= 1

        # 缓存结果
        GLOBAL_CACHE.set_board_features(board_hash, features)
        return features

    def _add_adjacent_empty(self, empty_set, board, r, c):
        """添加指定位置周围的空位到集合中"""
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] in [0, '0']:
                empty_set.add((nr, nc))

    def _evaluate_single_card(self, card, state, board_features):
        """使用预计算特征评估单张卡牌"""
        # 双眼J和单眼J的评估保持不变
        if self._is_two_eyed_jack(card):
            return 10000
        elif self._is_one_eyed_jack(card):
            return 5000
        elif card not in COORDS:
            return 0

        # 对普通卡牌使用快速评估
        positions = COORDS[card] if isinstance(COORDS[card], list) else [COORDS[card]]
        total_score = 0
        valid_positions = 0

        for pos in positions:
            if isinstance(pos, list):
                pos = tuple(pos)
            r, c = pos

            # 使用预计算特征快速评估
            if not self._is_position_available_fast(state.board.chips, r, c):
                continue

            # 快速计算位置价值
            position_score = self._fast_position_evaluation(r, c, board_features)
            total_score += position_score
            valid_positions += 1

        return total_score / max(1, valid_positions) if valid_positions > 0 else 0

    def _fast_position_evaluation(self, r, c, board_features):
        """基于预计算特征的快速位置评估"""
        score = POSITION_WEIGHTS.get((r, c), 1.0) * 10

        # 如果位置在热点区域附近，额外加分
        if (r, c) in board_features['empty_near_pieces']:
            score += 50

        # 中心控制奖励
        if (r, c) in HOTB_COORDS:
            score += board_features['center_control'] * 20

        return score

    def _get_board_hash(self, board):
        """快速棋盘哈希"""
        # 只对关键区域计算哈希
        key_positions = HOTB_COORDS + [(i, j) for i in range(3, 7) for j in range(3, 7)]
        return hash(tuple(board[i][j] for i, j in key_positions))

    def _is_position_available_fast(self, board, r, c):
        """快速检查位置可用性"""
        if not (0 <= r < 10 and 0 <= c < 10):
            return False
        if (r, c) in CORNERS:
            return True
        return board[r][c] == 0 or board[r][c] == '0'

    # 保持向后兼容的方法
    def _evaluate_card(self, card, state, consider_opponent=True):
        """向后兼容接口"""
        board_features = self._compute_board_features_once(state)
        return self._evaluate_single_card(card, state, board_features)

    def _is_two_eyed_jack(self, card):
        try:
            card_str = str(card).lower()
            return card_str in ['jc', 'jd']
        except:
            return False

    def _is_one_eyed_jack(self, card):
        try:
            card_str = str(card).lower()
            return card_str in ['js', 'jh']
        except:
            return False


# ===========================
# 6. 智能动作筛选器
# ===========================
class IntelligentActionFilter:
    """智能动作筛选器 - 分层筛选减少无效计算"""

    @staticmethod
    def filter_actions_hierarchical(actions, game_state, agent, max_candidates=15):
        """分层筛选动作：几何预筛选 -> 启发式评估 -> 精确排序"""
        if len(actions) <= max_candidates:
            return IntelligentActionFilter._simple_sort(actions, game_state)

        # 第一层：几何预筛选（基于位置关系）
        geometric_candidates = IntelligentActionFilter._geometric_prescreening(
            actions, game_state, agent)

        if len(geometric_candidates) <= max_candidates:
            return IntelligentActionFilter._simple_sort(geometric_candidates, game_state)

        # 第二层：启发式快速评估
        return IntelligentActionFilter._heuristic_ranking(
            geometric_candidates, game_state, max_candidates)

    @staticmethod
    def _geometric_prescreening(actions, game_state, agent):
        """基于几何位置的快速预筛选"""
        board = game_state.board.chips

        # 识别活跃区域（有棋子的位置周围）
        active_zones = set()
        threat_zones = set()

        for i in range(10):
            for j in range(10):
                if board[i][j] != 0 and board[i][j] != '0':
                    # 添加周围位置到活跃区域
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 10 and 0 <= nj < 10:
                                active_zones.add((ni, nj))

                    # 如果是对手棋子，检查威胁
                    if board[i][j] == agent.opp_color:
                        threat_positions = IntelligentActionFilter._get_threat_positions(
                            board, i, j, agent.opp_color)
                        threat_zones.update(threat_positions)

        # 筛选在活跃区域或威胁区域的动作
        priority_actions = []
        other_actions = []

        for action in actions:
            if action.get('type') == 'place' and 'coords' in action:
                coords = action['coords']
                if coords in threat_zones:
                    priority_actions.insert(0, action)  # 威胁处理最优先
                elif coords in active_zones:
                    priority_actions.append(action)
                else:
                    other_actions.append(action)
            else:
                priority_actions.append(action)  # 非放置动作保持优先

        # 返回优先动作 + 部分其他动作
        result = priority_actions + other_actions[:5]
        return result[:25]  # 限制总数

    @staticmethod
    def _get_threat_positions(board, r, c, color):
        """获取可能的威胁位置"""
        threats = set()
        for dx, dy in DIRECTIONS:
            # 检查这个方向的连续长度
            count = IntelligentActionFilter._count_in_direction(board, r, c, dx, dy, color)
            if count >= 3:  # 如果已经有3连，周围位置都是威胁
                for i in range(-4, 5):
                    nx, ny = r + i * dx, c + i * dy
                    if (0 <= nx < 10 and 0 <= ny < 10 and
                            (board[nx][ny] in [0, '0'] or (nx, ny) in CORNERS)):
                        threats.add((nx, ny))
        return threats

    @staticmethod
    def _count_in_direction(board, r, c, dx, dy, color):
        """计算某方向的连续棋子数（简化版）"""
        count = 1
        # 正向
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if (0 <= x < 10 and 0 <= y < 10 and
                    (board[x][y] == color or (x, y) in CORNERS)):
                count += 1
            else:
                break
        # 反向
        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if (0 <= x < 10 and 0 <= y < 10 and
                    (board[x][y] == color or (x, y) in CORNERS)):
                count += 1
            else:
                break
        return min(count, 5)

    @staticmethod
    def _heuristic_ranking(actions, game_state, max_count):
        """启发式快速排序"""
        scored_actions = []
        for action in actions:
            score = IntelligentActionFilter._quick_action_score(action, game_state)
            scored_actions.append((action, score))

        scored_actions.sort(key=lambda x: x[1])
        return [action for action, _ in scored_actions[:max_count]]

    @staticmethod
    def _quick_action_score(action, game_state):
        """快速动作评分（简化版）"""
        if action.get('type') != 'place' or 'coords' not in action:
            return 50

        r, c = action['coords']

        # 基础位置权重
        score = 100 - POSITION_WEIGHTS.get((r, c), 1.0) * 20

        # 中心位置奖励
        center_distance = abs(r - 4.5) + abs(c - 4.5)
        score -= max(0, 5 - center_distance) * 5

        return score

    @staticmethod
    def _simple_sort(actions, game_state):
        """简单排序"""
        return IntelligentActionFilter._heuristic_ranking(actions, game_state, len(actions))


# ===========================
# 7. 自适应MCTS搜索
# ===========================
class AdaptiveMCTSNode:
    """自适应MCTS节点 - 根据局面复杂度调整搜索策略"""

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.is_terminal = False

        # 自适应参数
        self.complexity_level = self._assess_complexity()
        self.exploration_weight = self._adaptive_exploration_weight()

    def _assess_complexity(self):
        """评估当前局面复杂度"""
        if not hasattr(self.state, 'board'):
            return 0.5

        board = self.state.board.chips
        piece_count = sum(1 for i in range(10) for j in range(10)
                          if board[i][j] not in [0, '0'])

        # 基于棋子数量判断复杂度
        if piece_count < 10:
            return 0.3  # 简单
        elif piece_count < 25:
            return 0.6  # 中等
        else:
            return 0.9  # 复杂

    def _adaptive_exploration_weight(self):
        """根据复杂度调整探索权重"""
        base_weight = EXPLORATION_WEIGHT
        if self.complexity_level < 0.4:
            return base_weight * 1.2  # 简单局面多探索
        elif self.complexity_level > 0.7:
            return base_weight * 0.8  # 复杂局面少探索，多利用
        return base_weight

    def get_untried_actions(self):
        """获取未尝试动作（延迟初始化）"""
        if self.untried_actions is None:
            if hasattr(self.state, 'available_actions'):
                self.untried_actions = list(self.state.available_actions)
            else:
                self.untried_actions = []
        return self.untried_actions

    def is_fully_expanded(self):
        return len(self.get_untried_actions()) == 0

    def select_child(self):
        """选择子节点（简化的UCB）"""
        if not self.children:
            return None

        best_score = float('-inf')
        best_child = None
        log_visits = math.log(self.visits) if self.visits > 0 else 0

        for child in self.children:
            if child.visits == 0:
                return child  # 优先选择未访问的节点

            exploitation = child.value / child.visits
            exploration = self.exploration_weight * math.sqrt(log_visits / child.visits)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, agent):
        """扩展节点"""
        untried = self.get_untried_actions()
        if not untried:
            return None

        action = untried.pop(0)
        new_state = agent.fast_simulate_lightweight(self.state, action)
        child = AdaptiveMCTSNode(new_state, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, result):
        """更新节点统计"""
        self.visits += 1
        self.value += result

    def get_best_child(self):
        """获取最佳子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)


# ===========================
# 8. 优化的状态评估器
# ===========================
class FastStateEvaluator:
    """快速状态评估器 - 使用缓存和简化计算"""

    @staticmethod
    def evaluate_fast(state, agent):
        """快速评估状态（使用缓存和简化算法）"""
        if not hasattr(state, 'board'):
            return 0.0

        board = state.board.chips
        board_hash = FastStateEvaluator._get_state_hash(board)

        # 尝试从缓存获取
        cached_score = GLOBAL_CACHE.get_position_score(board_hash)
        if cached_score is not None:
            return cached_score

        # 计算新分数
        score = FastStateEvaluator._compute_state_score(board, agent)

        # 缓存结果
        GLOBAL_CACHE.set_position_score(board_hash, score)
        return score

    @staticmethod
    def _compute_state_score(board, agent):
        """计算状态分数（优化版）"""
        my_color = agent.my_color
        opp_color = agent.opp_color

        score = 0

        # 1. 快速位置评分
        for pos, weight in POSITION_WEIGHTS.items():
            i, j = pos
            cell = board[i][j]
            if cell == my_color:
                score += weight * 2
            elif cell == opp_color:
                score -= weight * 2

        # 2. 序列潜力评分（简化版）
        sequence_score = FastStateEvaluator._fast_sequence_evaluation(board, my_color, opp_color)
        score += sequence_score

        # 归一化
        return max(-1, min(1, score / 200))

    @staticmethod
    def _fast_sequence_evaluation(board, my_color, opp_color):
        """快速序列评估（只检查关键区域）"""
        score = 0

        # 只检查中心区域和热点附近
        check_positions = HOTB_COORDS + [(i, j) for i in range(2, 8) for j in range(2, 8)]

        for r, c in check_positions:
            if board[r][c] == my_color or (r, c) in CORNERS:
                # 只检查水平和垂直方向（减少计算）
                for dx, dy in [(0, 1), (1, 0)]:
                    count = FastStateEvaluator._count_consecutive_fast(board, r, c, dx, dy, my_color)
                    if count >= 3:
                        score += count * 10

            elif board[r][c] == opp_color:
                # 检查对手威胁
                for dx, dy in [(0, 1), (1, 0)]:
                    count = FastStateEvaluator._count_consecutive_fast(board, r, c, dx, dy, opp_color)
                    if count >= 4:
                        score -= count * 15

        return score

    @staticmethod
    def _count_consecutive_fast(board, x, y, dx, dy, color):
        """快速计算连续棋子（简化版）"""
        count = 1

        # 正向检查（最多3步）
        for i in range(1, 4):
            nx, ny = x + i * dx, y + i * dy
            if (0 <= nx < 10 and 0 <= ny < 10 and
                    (board[nx][ny] == color or (nx, ny) in CORNERS)):
                count += 1
            else:
                break

        # 反向检查（最多3步）
        for i in range(1, 4):
            nx, ny = x - i * dx, y - i * dy
            if (0 <= nx < 10 and 0 <= ny < 10 and
                    (board[nx][ny] == color or (nx, ny) in CORNERS)):
                count += 1
            else:
                break

        return min(count, 5)

    @staticmethod
    def _get_state_hash(board):
        """快速状态哈希"""
        return hash(tuple(tuple(row[2:8]) for row in board[2:8]))  # 只哈希中心区域


# ===========================
# 9. 优化的时间管理器
# ===========================
class OptimizedTimeManager:
    """优化的时间管理器 - 预测性时间分配"""

    def __init__(self):
        self.start_time = 0
        self.time_budget = MAX_THINK_TIME
        self.computation_history = []  # 记录计算耗时历史
        self.move_count = 0

    def start_timing(self):
        self.start_time = time.time()
        self.move_count += 1

    def get_remaining_time(self):
        return self.time_budget - (time.time() - self.start_time)

    def is_timeout(self, buffer=0.05):
        return self.get_remaining_time() < buffer

    def predict_computation_time(self, candidate_count, complexity):
        """预测所需计算时间"""
        if not self.computation_history:
            return 0.4  # 默认估计

        # 基于历史数据预测
        recent_times = self.computation_history[-5:]
        avg_time = sum(recent_times) / len(recent_times)

        # 根据候选数量和复杂度调整
        predicted = avg_time * (candidate_count / 10) * (1 + complexity)
        return min(predicted, self.time_budget * 0.8)

    def should_use_fast_mode(self, required_time):
        """判断是否应该使用快速模式"""
        available_time = self.get_remaining_time()
        return required_time > available_time * 0.7

    def record_computation_time(self, elapsed_time):
        """记录计算时间"""
        self.computation_history.append(elapsed_time)
        if len(self.computation_history) > 20:
            self.computation_history.pop(0)  # 保持最近20次记录


# ===========================
# 10. 主Agent类（优化版）
# ===========================
class myAgent(Agent):
    """优化版智能体 - 保持所有原有接口和功能"""

    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.counter = itertools.count()

        # 使用优化的组件
        self.card_evaluator = OptimizedCardEvaluator(self)
        self.time_manager = OptimizedTimeManager()

        # 颜色信息
        self.my_color = None
        self.opp_color = None

        # 搜索参数（动态调整）
        self.base_simulation_depth = 4
        self.base_candidate_limit = 12

        # 开局库（保持不变）
        self._initialize_opening_book()
        self.move_history = []

        # 性能统计
        self.performance_stats = {
            'fast_decisions': 0,
            'full_searches': 0,
            'cache_usage': 0
        }

    def _initialize_opening_book(self):
        """初始化开局库（保持不变）"""
        self.opening_book = {
            0: [(4, 4), (4, 5), (5, 4), (5, 5)],
            1: {
                (4, 4): [(5, 5), (4, 5), (5, 4), (3, 3)],
                (4, 5): [(5, 4), (4, 4), (5, 5), (3, 4)],
                (5, 4): [(4, 5), (5, 5), (4, 4), (4, 3)],
                (5, 5): [(4, 4), (5, 4), (4, 5), (3, 3)]
            },
            2: {
                'center_control': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 6),
                                   (5, 3), (5, 6), (6, 3), (6, 4), (6, 5)]
            }
        }

    def _initialize_colors(self, game_state):
        """初始化颜色信息（保持不变）"""
        if self.my_color is None:
            self.my_color = game_state.agents[self.id].colour
            self.opp_color = game_state.agents[1 - self.id].colour

    def SelectAction(self, actions, game_state):
        """主决策函数（保持接口不变，内部优化）"""
        decision_start = time.time()
        self.time_manager.start_timing()
        self._initialize_colors(game_state)

        # 边界检查
        if not actions:
            return None

        # 卡牌选择逻辑（保持不变）
        if self._is_card_selection(actions):
            result = self._select_strategic_card(actions, game_state)
            self._record_decision_time(decision_start)
            return result

        # 即时获胜检测（保持不变）
        for action in actions:
            if self._is_winning_move_fast(game_state, action):
                return action

        # 阻止对手获胜（保持不变）
        for action in actions:
            if self._blocks_opponent_win_fast(game_state, action):
                return action

        # 开局库查询（保持不变）
        move_num = len(self.move_history)
        if move_num < 3:
            opening_move = self._try_opening_book(actions, game_state, move_num)
            if opening_move:
                self._record_decision_time(decision_start)
                return opening_move

        # 智能动作筛选（优化版）
        candidates = IntelligentActionFilter.filter_actions_hierarchical(
            actions, game_state, self, self.base_candidate_limit)

        if not candidates:
            candidates = actions[:1]

        # 预测计算时间并选择策略
        complexity = self._assess_game_complexity(game_state)
        predicted_time = self.time_manager.predict_computation_time(len(candidates), complexity)

        if self.time_manager.should_use_fast_mode(predicted_time):
            # 快速模式：简化搜索
            result = self._fast_decision_mode(candidates, game_state)
            self.performance_stats['fast_decisions'] += 1
        else:
            # 完整搜索模式
            result = self._full_search_mode(candidates, game_state)
            self.performance_stats['full_searches'] += 1

        self._record_decision_time(decision_start)
        return result

    def _fast_decision_mode(self, candidates, game_state):
        """快速决策模式 - 减少搜索深度"""
        if len(candidates) == 1:
            return candidates[0]

        # 使用简化评估快速排序
        scored_candidates = []
        for action in candidates[:8]:  # 只考虑前8个候选
            score = self._quick_evaluate_action(action, game_state)
            scored_candidates.append((action, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def _full_search_mode(self, candidates, game_state):
        """完整搜索模式 - 使用优化的MCTS"""
        return self._optimized_mcts_search(candidates, game_state)

    def _optimized_mcts_search(self, candidates, game_state):
        """优化的MCTS搜索"""
        if not candidates:
            return None

        # 准备MCTS状态
        mcts_state = self._prepare_mcts_state(game_state, candidates)
        root = AdaptiveMCTSNode(mcts_state)

        # 直接创建候选子节点
        for action in candidates:
            next_state = self.fast_simulate_lightweight(mcts_state, action)
            child = AdaptiveMCTSNode(next_state, parent=root, action=action)
            root.children.append(child)

        # 自适应搜索参数
        complexity = root.complexity_level
        if complexity < 0.4:
            max_iterations = 80  # 简单局面快速决策
        elif complexity > 0.7:
            max_iterations = min(SIMULATION_LIMIT, 200)  # 复杂局面深度搜索
        else:
            max_iterations = 120  # 中等复杂度

        # MCTS主循环
        iterations = 0
        while not self.time_manager.is_timeout() and iterations < max_iterations:
            iterations += 1

            # 选择
            node = root
            while node.children and node.is_fully_expanded():
                node = node.select_child()
                if node is None:
                    break

            if node is None:
                continue

            # 扩展
            if node.visits > 0 and not node.is_fully_expanded():
                child = node.expand(self)
                if child:
                    node = child

            # 模拟
            value = self._lightweight_simulation(node.state)

            # 回溯
            while node:
                node.update(value)
                node = node.parent

        # 选择最佳动作
        best_child = root.get_best_child()
        return best_child.action if best_child else candidates[0]

    def _lightweight_simulation(self, state):
        """轻量级模拟"""
        # 对于简单局面，直接使用状态评估
        complexity = self._assess_state_complexity(state)
        if complexity < 0.4:
            return FastStateEvaluator.evaluate_fast(state, self)

        # 对于复杂局面，进行1-2步的快速模拟
        simulation_depth = 2 if complexity > 0.7 else 1
        current_state = state

        for _ in range(simulation_depth):
            if hasattr(current_state, 'available_actions'):
                actions = current_state.available_actions
            else:
                try:
                    actions = self.rule.getLegalActions(current_state, self.id)
                except:
                    break

            if not actions:
                break

            # 快速选择动作
            action = self._quick_action_selection(actions, current_state)
            current_state = self.fast_simulate_lightweight(current_state, action)

        return FastStateEvaluator.evaluate_fast(current_state, self)

    def _quick_action_selection(self, actions, state):
        """快速动作选择"""
        if len(actions) <= 3:
            return random.choice(actions)

        # 简单启发式选择
        scored_actions = []
        sample_size = min(5, len(actions))
        sampled_actions = random.sample(actions, sample_size)

        for action in sampled_actions:
            score = self._quick_evaluate_action(action, state)
            scored_actions.append((action, score))

        scored_actions.sort(key=lambda x: x[1], reverse=True)
        return scored_actions[0][0]

    def _quick_evaluate_action(self, action, state):
        """快速动作评估"""
        if action.get('type') != 'place' or 'coords' not in action:
            return 0

        r, c = action['coords']

        # 基础位置分数
        score = POSITION_WEIGHTS.get((r, c), 1.0)

        # 中心偏好
        center_dist = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 5 - center_dist)

        return score

    def fast_simulate_lightweight(self, state, action):
        """轻量级快速模拟"""
        # 创建轻量级状态副本
        new_state = self._create_lightweight_copy(state)

        if action['type'] == 'place' and 'coords' in action:
            r, c = action['coords']
            if (r, c) not in CORNERS and new_state.board.chips[r][c] in [0, '0']:
                new_state.board.chips[r][c] = self.my_color
        elif action['type'] == 'remove' and 'coords' in action:
            r, c = action['coords']
            if (r, c) not in CORNERS:
                new_state.board.chips[r][c] = 0

        return new_state

    def _create_lightweight_copy(self, state):
        """创建轻量级状态副本"""
        new_state = copy.copy(state)
        new_state.board = LightweightBoardState(state.board.chips)
        return new_state

    # 保持所有原有的辅助方法不变
    def _is_card_selection(self, actions):
        return any(a.get('type') == 'trade' for a in actions)

    def _select_strategic_card(self, actions, game_state):
        """战略卡牌选择（保持原有逻辑）"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']
        if not trade_actions:
            return None

        if not hasattr(game_state, 'display_cards') or not game_state.display_cards:
            return random.choice(trade_actions)

        # 使用优化的卡牌评估器
        card_scores = self.card_evaluator.evaluate_card_batch(
            game_state.display_cards, game_state)

        best_card = max(card_scores.keys(), key=lambda c: card_scores[c])

        for action in trade_actions:
            if action.get('draft_card') == best_card:
                return action

        return random.choice(trade_actions)

    def _is_winning_move_fast(self, state, action):
        """快速获胜检测"""
        if action.get('type') != 'place' or 'coords' not in action:
            return False

        r, c = action['coords']
        board = state.board.chips

        # 简化检查：只检查主要方向
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = FastStateEvaluator._count_consecutive_fast(board, r, c, dx, dy, self.my_color)
            if count >= 5:
                return True
        return False

    def _blocks_opponent_win_fast(self, state, action):
        """快速阻止对手获胜检测"""
        if action.get('type') != 'place' or 'coords' not in action:
            return False

        r, c = action['coords']
        board = state.board.chips

        # 检查该位置是否阻止对手形成5连
        for dx, dy in [(0, 1), (1, 0)]:  # 只检查主要方向
            opp_count = FastStateEvaluator._count_consecutive_fast(board, r, c, dx, dy, self.opp_color)
            if opp_count >= 4:
                return True
        return False

    def _try_opening_book(self, actions, game_state, move_num):
        """尝试使用开局库"""
        if move_num >= len(self.opening_book):
            return None

        if move_num == 0:
            recommendations = self.opening_book[0]
        elif move_num == 2:
            recommendations = self.opening_book[2]['center_control']
        else:
            last_opp_move = self._get_last_opponent_move(game_state)
            if not last_opp_move or last_opp_move not in self.opening_book[1]:
                return None
            recommendations = self.opening_book[1][last_opp_move]

        for r, c in recommendations:
            for action in actions:
                if action.get('coords') == (r, c):
                    return action
        return None

    def _get_last_opponent_move(self, game_state):
        """获取对手最后一步"""
        board = game_state.board.chips
        for r, c in HOTB_COORDS:
            if board[r][c] == self.opp_color:
                return (r, c)
        return None

    def _assess_game_complexity(self, game_state):
        """评估游戏复杂度"""
        board = game_state.board.chips
        piece_count = sum(1 for i in range(10) for j in range(10)
                          if board[i][j] not in [0, '0'])
        return min(1.0, piece_count / 50)

    def _assess_state_complexity(self, state):
        """评估状态复杂度"""
        if not hasattr(state, 'board'):
            return 0.5
        return self._assess_game_complexity(state)

    def _prepare_mcts_state(self, game_state, actions):
        """准备MCTS状态"""
        mcts_state = self._create_lightweight_copy(game_state)
        mcts_state.my_color = self.my_color
        mcts_state.opp_color = self.opp_color
        mcts_state.current_player_id = self.id
        mcts_state.available_actions = actions
        return mcts_state

    def _record_decision_time(self, start_time):
        """记录决策时间"""
        elapsed = time.time() - start_time
        self.time_manager.record_computation_time(elapsed)

    # 向后兼容方法
    def fast_simulate(self, state, action):
        """向后兼容的快速模拟方法"""
        return self.fast_simulate_lightweight(state, action)

    def custom_shallow_copy(self, state):
        """向后兼容的状态复制方法"""
        return self._create_lightweight_copy(state)