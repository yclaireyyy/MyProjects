from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import random
import time
import copy
import math
import itertools

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

    def _evaluate_card(self, card, state):
        """评估卡牌在当前状态下的价值（带缓存）"""
        # 生成缓存键
        board_hash = state.board.chips if hasattr(state.board, 'get_hash') else hash(
            tuple(tuple(row) for row in state.board.chips))
        cache_key = (str(card), board_hash)

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
            result = self._exponential_card_evaluation(card, state)
        else:
            result = 0

        self._card_cache[cache_key] = result
        return result

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

    @staticmethod
    def evaluate_action_quality(state, action):
        """评估动作的质量得分（越低越好）- 带缓存优化"""
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

        # 阻止对手评分（优化版）
        for dx, dy in DIRECTIONS:
            enemy_chain = ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy)
            if enemy_chain >= 3:
                score += 50  # 高优先级阻断

        # 中心控制评分（使用位置权重）
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if (x, y) == (r, c))
        score += hotb_controlled * 15

        # 转换为质量分数（越低越好）
        result = 100 - score
        ActionEvaluator._evaluation_cache[cache_key] = result
        return result

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
            if enemy_threat >= 3:
                score += 50

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


class TimeManager:
    def __init__(self):
        self.start_time = 0
        self.time_budget = MAX_THINK_TIME

    def start_timing(self):
        self.start_time = time.time()

    def get_remaining_time(self):
        elapsed = time.time() - self.start_time
        return self.time_budget - elapsed

    def is_timeout(self, buffer=0.05):
        return self.get_remaining_time() < buffer

    def should_use_quick_mode(self):
        return self.get_remaining_time() < 0.25  # 稍微降低阈值


class myAgent(Agent):
    """智能体 myAgent"""

    def __init__(self, _id):
        """初始化Agent"""
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)  # 2人游戏
        self.counter = itertools.count()  # 用于搜索的唯一标识符

        self.card_evaluator = CardEvaluator(self)
        self.time_manager = TimeManager()

        # 玩家颜色初始化
        self.my_color = None
        self.opp_color = None

        # 搜索参数（优化后）
        self.simulation_depth = 4  # 稍微减少深度但提高质量
        self.candidate_limit = 12  # 稍微增加候选数

        # 时间控制
        self.start_time = 0

    def _initialize_colors(self, game_state):
        """初始化颜色信息"""
        if self.my_color is None:
            self.my_color = game_state.agents[self.id].colour
            self.opp_color = game_state.agents[1 - self.id].colour

    def _is_card_selection(self, actions):
        """判断是否为卡牌选择"""
        return any(a.get('type') == 'trade' for a in actions)

    def _select_strategic_card(self, actions, game_state):
        """卡牌选择逻辑"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']

        if not hasattr(game_state, 'display_cards'):
            return random.choice(trade_actions)

        # 评估所有展示牌
        best_card = None
        best_score = float('-inf')

        for card in game_state.display_cards:
            score = self.card_evaluator._evaluate_card(card, game_state)
            if score > best_score:
                best_score = score
                best_card = card

        # 找到对应动作
        for action in trade_actions:
            if action.get('draft_card') == best_card:
                return action

        return random.choice(trade_actions)

    def SelectAction(self, actions, game_state):
        """主决策函数 - 融合启发式筛选和MCTS"""
        self.time_manager.start_timing()
        self._initialize_colors(game_state)

        if self._is_card_selection(actions):
            return self._select_strategic_card(actions, game_state)

        # 启发式筛选候选动作
        candidates = self._heuristic_filter(actions, game_state)

        # 时间检查
        if self.time_manager.should_use_quick_mode():
            return candidates[0] if candidates else random.choice(actions)
        # MCTS深度搜索
        try:
            return self._mcts_search(candidates, game_state)
        except:
            return candidates[0] if candidates else random.choice(actions)

    def _heuristic_filter(self, actions, game_state):
        """使用启发式评估筛选最有前途的动作（优化版）"""
        # 排除角落位置
        valid_actions = [a for a in actions if 'coords' not in a or a['coords'] not in CORNERS]
        if not valid_actions:
            return actions[:1]  # 如果没有有效动作，返回第一个动作

        # 批量评估并排序
        scored_actions = [(a, ActionEvaluator.evaluate_action_quality(game_state, a)) for a in valid_actions]
        scored_actions.sort(key=lambda x: x[1])

        # 返回前N个候选动作
        return [a for a, _ in scored_actions[:self.candidate_limit]]

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

        # MCTS主循环
        iterations = 0
        time_check_interval = 10  # 每10次迭代检查一次时间

        while not self.time_manager.is_timeout() and iterations < SIMULATION_LIMIT:
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

            # 85%概率使用启发式，15%随机选择（稍微提高启发式比例）
            if random.random() < 0.85:
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