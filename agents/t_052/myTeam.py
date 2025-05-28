from collections import namedtuple
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
from Sequence.sequence_model import COORDS
import heapq
import time
import itertools
import random
import math
from collections import defaultdict

MAX_THINK_TIME = 0.95
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]

TTEntry = namedtuple('TTEntry', 'depth score flag best_move')


# ================================================================================================
# 优化配置类 - 预计算和快速访问
# ================================================================================================

class GameConfig:
    """游戏配置和权重管理 - 优化版"""

    def __init__(self):
        # 预计算所有权重配置
        self.phase_weights = {
            'opening': {'center': 2.0, 'chain': 0.8, 'block': 0.5, 'corner': 1.5},
            'middle': {'center': 1.0, 'chain': 1.5, 'block': 1.2, 'corner': 1.8},
            'endgame': {'center': 0.3, 'chain': 2.0, 'block': 1.8, 'corner': 2.5}
        }

        self.evaluation_weights = {
            'chain_win': 1.0, 'chain_threat_4': 1.2, 'chain_threat_3': 1.0,
            'chain_threat_2': 1.0, 'compound_bonus': 1.0, 'fork_attack': 1.5,
            'chain_threat': 1.0, 'block_enemy_win': 1.0, 'block_enemy_threat': 1.0,
            'block_compound': 1.3, 'hotb_control': 1.1, 'center_bias': 1.0,
            'corner_strategic': 1.0, 'tempo': 0.9, 'space_control': 1.0,
            'flexibility': 1.0
        }

        self.card_selection_weights = {
            'immediate_play': 3.0, 'strategic_position': 2.0, 'blocking_value': 2.5,
            'flexibility': 1.0, 'opponent_denial': 2.8, 'opponent_expectation': 2.2
        }

        self.compound_threat_config = {
            'fork_attack_bonus': 800, 'double_threat_bonus': 500,
            'chain_reaction_bonus': 600, 'cross_pattern_bonus': 400,
            'tempo_advantage_bonus': 300
        }

        # 预计算核心数据
        self._precompute_core_data()

    def _precompute_core_data(self):
        """预计算核心数据结构"""
        self.CORNER_POSITIONS = [(0, 0), (0, 9), (9, 0), (9, 9)]
        self.direction_vectors = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 预计算位置价值
        self.position_values = {}
        for r in range(10):
            for c in range(10):
                value = 0
                if (r, c) in HOTB_COORDS:
                    value += 120
                distance_to_center = abs(r - 4.5) + abs(c - 4.5)
                value += max(0, 25 - distance_to_center * 2.5)
                for corner_r, corner_c in self.CORNER_POSITIONS:
                    corner_distance = max(abs(r - corner_r), abs(c - corner_c))
                    if corner_distance <= 3:
                        value += max(0, 20 - corner_distance * 5)
                self.position_values[(r, c)] = value


# ================================================================================================
# 高效时间管理器
# ================================================================================================

class TimeManager:
    """优化时间管理系统 - 动态调整算法时间分配"""

    def __init__(self, total_time_limit=0.95):
        self.total_limit = total_time_limit
        self.start_time = None
        self.safety_buffer = 0.03  # 减少安全缓冲，更积极使用时间

        # 动态阶段时间分配
        self.base_phase_limits = {
            'quick_decision': 0.10,  # 快速决策
            'astar_search': 0.45,  # A*搜索
            'mcts_refinement': 0.85  # MCTS优化
        }

        self.phase_limits = self.base_phase_limits.copy()

    def start_turn(self):
        self.start_time = time.time()

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    def get_remaining_time(self):
        return max(0, self.total_limit - self.safety_buffer - self.get_elapsed_time())

    def should_continue_phase(self, phase_name):
        elapsed = self.get_elapsed_time()
        phase_limit = self.phase_limits.get(phase_name, 0.8)
        return elapsed < phase_limit

    def get_remaining_for_phase(self, phase_name):
        phase_limit = self.phase_limits.get(phase_name, 0.8)
        elapsed = self.get_elapsed_time()
        return max(0, phase_limit - elapsed)

    def adjust_phase_limits(self, complexity_factor):
        """根据局面复杂度动态调整时间分配"""
        if complexity_factor > 0.7:  # 复杂局面，更多时间给搜索
            self.phase_limits = {
                'quick_decision': 0.05,
                'astar_search': 0.35,
                'mcts_refinement': 0.85
            }
        elif complexity_factor < 0.3:  # 简单局面，快速决策
            self.phase_limits = {
                'quick_decision': 0.20,
                'astar_search': 0.50,
                'mcts_refinement': 0.80
            }
        else:
            self.phase_limits = self.base_phase_limits.copy()


# ================================================================================================
# 高效缓存管理器
# ================================================================================================

class CacheManager:
    """轻量级缓存管理 - 只缓存高价值计算"""

    def __init__(self, max_size=500):  # 减小缓存大小
        self.eval_cache = {}  # 评估缓存
        self.max_size = max_size
        self.hit_count = 0
        self.total_count = 0

    def get_eval_cache(self, board_hash, coords):
        """获取评估缓存"""
        self.total_count += 1
        cache_key = (board_hash, coords)
        if cache_key in self.eval_cache:
            self.hit_count += 1
            return self.eval_cache[cache_key]
        return None

    def set_eval_cache(self, board_hash, coords, result):
        """设置评估缓存"""
        if len(self.eval_cache) >= self.max_size:
            # 简单的FIFO清理
            oldest_keys = list(self.eval_cache.keys())[:self.max_size // 4]
            for key in oldest_keys:
                del self.eval_cache[key]

        cache_key = (board_hash, coords)
        self.eval_cache[cache_key] = result

    def _get_board_hash(self, board):
        """快速棋盘哈希"""
        return hash(''.join(''.join(row) for row in board))


# ================================================================================================
# 高效棋盘分析器
# ================================================================================================

class BoardAnalyzer:
    """棋盘分析核心功能 - 内联优化版"""

    def __init__(self, config):
        self.config = config
        self.CORNER_POSITIONS = config.CORNER_POSITIONS
        self.direction_vectors = config.direction_vectors

    def analyze_chain_pattern(self, board, r, c, dx, dy, color):
        """角落感知的连子模式分析 - 内联优化"""
        count = 1
        openings = 0

        # 正向计数 - 内联优化
        pos_open = False
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color or (x, y) in self.CORNER_POSITIONS:
                    count += 1
                elif board[x][y] == '0':
                    pos_open = True
                    break
                else:
                    break
            else:
                break

        # 负向计数 - 内联优化
        neg_open = False
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color or (x, y) in self.CORNER_POSITIONS:
                    count += 1
                elif board[x][y] == '0':
                    neg_open = True
                    break
                else:
                    break
            else:
                break

        openings = (1 if pos_open else 0) + (1 if neg_open else 0)
        return count, openings

    def is_corner_position(self, r, c):
        return (r, c) in self.CORNER_POSITIONS


# ================================================================================================
# 内联威胁分析器
# ================================================================================================

class ThreatAnalyzer:
    """威胁分析专用类 - 内联优化"""

    def __init__(self, board_analyzer, config):
        self.board_analyzer = board_analyzer
        self.config = config
        # 预计算威胁分类映射
        self.threat_urgency_map = {
            'win': 10000, 'critical_win': 9000, 'major_threat': 7000,
            'double_threat': 5000, 'active_threat': 3000, 'blocked_threat': 1500,
            'potential_threat': 800, 'development': 400, 'weak_connection': 100,
            'none': 0
        }

    def classify_threat_fast(self, count, openings, corner_support):
        """快速威胁分类 - 内联版本"""
        if count >= 5:
            return 'win', 10000
        elif count >= 4:
            if openings >= 2 or (openings >= 1 and corner_support):
                return 'critical_win', 9000
            elif openings >= 1 or corner_support:
                return 'major_threat', 7000
            else:
                return 'blocked_threat', 1500
        elif count >= 3:
            if openings >= 2:
                return 'double_threat', 5000
            elif openings >= 1 or corner_support:
                return 'active_threat', 3000
            else:
                return 'potential_threat', 800
        elif count >= 2:
            if openings >= 2:
                return 'development', 400
            else:
                return 'weak_connection', 100
        else:
            return 'none', 0

    def threat_analysis_fast(self, board, r, c, color, enemy_color):
        """快速完整威胁分析"""
        max_urgency = 0
        threat_count = 0

        for dx, dy in self.board_analyzer.direction_vectors:
            temp_board = [row[:] for row in board]
            temp_board[r][c] = color

            count, openings = self.board_analyzer.analyze_chain_pattern(temp_board, r, c, dx, dy, color)
            corner_support = self._check_corner_support_fast(temp_board, r, c, dx, dy, color)

            threat_type, urgency = self.classify_threat_fast(count, openings, corner_support)
            if threat_type != 'none':
                max_urgency = max(max_urgency, urgency)
                threat_count += 1

        return max_urgency, threat_count

    def _check_corner_support_fast(self, board, r, c, dx, dy, color):
        """快速角落支持检查"""
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if not (0 <= x < 10 and 0 <= y < 10):
                    break
                if (x, y) in self.board_analyzer.CORNER_POSITIONS:
                    return True
                if board[x][y] != color and board[x][y] != '0' and (x, y) not in self.board_analyzer.CORNER_POSITIONS:
                    break
        return False


# ================================================================================================
# 高效位置评估器
# ================================================================================================

class PositionEvaluator:
    """位置评估系统 - 高度优化版"""

    def __init__(self, board_analyzer, threat_analyzer, config):
        self.board_analyzer = board_analyzer
        self.threat_analyzer = threat_analyzer
        self.config = config
        self.position_values = config.position_values

    def fast_comprehensive_score(self, board, r, c, my_color, enemy_color):
        """快速综合评分 - 一次性计算所有维度"""
        # 基础位置价值
        total_score = self.position_values.get((r, c), 0)

        # 连子评分 - 内联计算
        for dx, dy in self.board_analyzer.direction_vectors:
            count, openings = self.board_analyzer.analyze_chain_pattern(board, r, c, dx, dy, my_color)
            corner_support = self.threat_analyzer._check_corner_support_fast(board, r, c, dx, dy, my_color)

            # 内联指数级评分逻辑
            if count >= 5:
                total_score += 100000
                break  # 获胜就不需要继续
            elif count == 4:
                if openings == 2:
                    total_score += 50000
                elif corner_support and openings >= 1:
                    total_score += 30000
                elif openings == 1:
                    total_score += 20000
                elif corner_support:
                    total_score += 15000
                else:
                    total_score += 3000
            elif count == 3:
                if openings == 2:
                    total_score += 2000
                elif corner_support and openings >= 1:
                    total_score += 1500
                elif openings == 1:
                    total_score += 600
                elif corner_support:
                    total_score += 800
                else:
                    total_score += 100
            elif count >= 2:
                if openings >= 2:
                    total_score += 200
                elif corner_support:
                    total_score += 150
                else:
                    total_score += 80

        # 阻断评分 - 内联计算
        for dx, dy in self.board_analyzer.direction_vectors:
            enemy_threat = self._calculate_enemy_threat_fast(board, r, c, dx, dy, enemy_color)
            if enemy_threat >= 5:
                total_score += 15000
            elif enemy_threat >= 4:
                total_score += 8000
            elif enemy_threat >= 3:
                total_score += 2000
            elif enemy_threat >= 2:
                total_score += 500

        # HOTB控制 - 快速计算
        if (r, c) in HOTB_COORDS:
            total_score += self._hotb_bonus_fast(board, my_color, enemy_color)

        return total_score

    def _calculate_enemy_threat_fast(self, board, r, c, dx, dy, enemy_color):
        """快速敌方威胁计算"""
        threat_level = 0
        for direction in [1, -1]:
            count = 0
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if board[x][y] == enemy_color or (x, y) in self.board_analyzer.CORNER_POSITIONS:
                        count += 1
                    elif board[x][y] == '0':
                        break
                    else:
                        break
                else:
                    break
            threat_level = max(threat_level, count)
        return threat_level

    def _hotb_bonus_fast(self, board, my_color, enemy_color):
        """快速HOTB奖励计算"""
        my_control = sum(1 for r, c in HOTB_COORDS if board[r][c] == my_color)
        enemy_control = sum(1 for r, c in HOTB_COORDS if board[r][c] == enemy_color)

        bonus = my_control * 50 - enemy_control * 80
        if my_control >= 3:
            bonus += 200
        if my_control == 4:
            bonus += 500
        return bonus


# ================================================================================================
# 快速卡片选择器
# ================================================================================================

class CardSelector:
    """卡片选择策略 - 高效版"""

    def __init__(self, board_analyzer, threat_analyzer, position_evaluator, config):
        self.board_analyzer = board_analyzer
        self.threat_analyzer = threat_analyzer
        self.position_evaluator = position_evaluator
        self.config = config

    def select_optimal_card(self, card_actions, game_state):
        """选择最优卡片 - 快速版本"""
        if not card_actions:
            return None

        agent, board, is_valid = self._safe_get_game_state_info(game_state)
        if not is_valid:
            return card_actions[0]

        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        best_card = None
        best_score = float('-inf')

        # 限制评估数量以节省时间
        for card_action in card_actions[:min(len(card_actions), 5)]:
            score = self._fast_card_evaluation(card_action, board, my_color, enemy_color, game_state)
            if score > best_score:
                best_score = score
                best_card = card_action

        return best_card or card_actions[0]

    def _fast_card_evaluation(self, card_action, board, my_color, enemy_color, game_state):
        """快速卡片评估"""
        card_positions = self._get_card_board_positions(card_action, game_state)
        if not card_positions:
            return 0

        total_score = 0

        # 快速胜利检查
        for r, c in card_positions[:3]:  # 只检查前3个位置
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 快速获胜检查
                temp_board = [row[:] for row in board]
                temp_board[r][c] = my_color
                for dx, dy in self.board_analyzer.direction_vectors:
                    count, _ = self.board_analyzer.analyze_chain_pattern(temp_board, r, c, dx, dy, my_color)
                    if count >= 5:
                        return 100000

        # 快速综合评估
        available_count = 0
        for r, c in card_positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                available_count += 1
                total_score += self.position_evaluator.fast_comprehensive_score(board, r, c, my_color,
                                                                                enemy_color) * 0.3

        total_score += available_count * 50  # 灵活性奖励
        return total_score

    def _get_card_board_positions(self, card_action, game_state):
        """获取卡片对应的棋盘位置 - 优化版"""
        try:
            card = None
            if isinstance(card_action, dict):
                card = (card_action.get('draft_card') or
                        card_action.get('card') or
                        card_action.get('play_card'))

            if card and isinstance(card, str) and card in COORDS:
                return COORDS[card]

            if isinstance(card, (tuple, list)) and len(card) == 2:
                card_str1 = f"{card[1]}{card[0]}"
                card_str2 = f"{card[0]}{card[1]}"

                if card_str1 in COORDS:
                    return COORDS[card_str1]
                elif card_str2 in COORDS:
                    return COORDS[card_str2]

            return []
        except Exception:
            return []

    def _safe_get_game_state_info(self, game_state):
        """安全获取游戏状态信息"""
        try:
            agent = game_state.agents[self.board_analyzer.agent_id] if hasattr(self.board_analyzer, 'agent_id') else \
                game_state.agents[0]
            board = game_state.board.chips
            return agent, board, True
        except (AttributeError, KeyError, IndexError):
            return None, None, False


# ================================================================================================
# 高效A*搜索引擎
# ================================================================================================

class SearchEngine:
    """高效A*搜索算法引擎"""

    def __init__(self, position_evaluator, time_manager, config, cache_manager):
        self.position_evaluator = position_evaluator
        self.time_manager = time_manager
        self.config = config
        self.cache_manager = cache_manager
        self.counter = itertools.count()

    def a_star_search(self, initial_state, candidate_moves, game_rule, agent_id):
        """高效A*搜索算法"""
        if not candidate_moves:
            return None

        pending = []
        seen_states = set()
        best_sequence = []
        top_reward = float('-inf')

        # 动态参数调整
        remaining_time = self.time_manager.get_remaining_for_phase('astar_search')
        if remaining_time > 0.3:
            max_candidates, max_expansions, depth_limit = 4, 15, 2
        elif remaining_time > 0.15:
            max_candidates, max_expansions, depth_limit = 3, 8, 1
        else:
            max_candidates, max_expansions, depth_limit = 2, 5, 1

        # 快速预排序候选动作
        candidate_moves = self._smart_sort_candidates(initial_state, candidate_moves, agent_id)

        # 初始化搜索
        for move in candidate_moves[:max_candidates]:
            g = 1
            h = self._fast_heuristic(initial_state, move, agent_id)
            f = g + h
            heapq.heappush(pending, (f, next(self.counter), g, h,
                                     self._fast_simulate(initial_state, move, game_rule, agent_id), [move]))

        expansions_count = 0
        while (pending and expansions_count < max_expansions and
               self.time_manager.should_continue_phase('astar_search')):

            f, _, g, h, current_state, move_history = heapq.heappop(pending)

            if g > depth_limit:
                continue

            # 简化的状态去重
            state_sig = id(current_state)
            if state_sig in seen_states:
                continue
            seen_states.add(state_sig)

            expansions_count += 1

            # 快速状态评估
            reward = self._fast_evaluate_state(current_state, move_history[-1], agent_id)
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history

            # 扩展搜索
            if (g < depth_limit and
                    self.time_manager.get_remaining_for_phase('astar_search') > 0.05):

                try:
                    next_steps = game_rule.getLegalActions(current_state, agent_id)
                    if next_steps:
                        next_steps = self._smart_sort_candidates(current_state, next_steps, agent_id)

                        for next_move in next_steps[:2]:  # 限制分支因子
                            next_g = g + 1
                            next_h = self._fast_heuristic(current_state, next_move, agent_id)
                            heapq.heappush(pending, (
                                next_g + next_h, next(self.counter),
                                next_g, next_h,
                                self._fast_simulate(current_state, next_move, game_rule, agent_id),
                                move_history + [next_move]
                            ))
                except:
                    continue

        return best_sequence[0] if best_sequence else None

    def _smart_sort_candidates(self, state, candidates, agent_id):
        """智能排序候选动作 - 优化版"""
        if len(candidates) <= 3:
            return candidates

        def move_priority(action):
            if not action.get('coords'):
                return 0
            r, c = action['coords']

            # 快速评分
            score = self.config.position_values.get((r, c), 0)

            # 快速威胁检查
            agent = state.agents[agent_id]
            my_color = agent.colour
            board = state.board.chips

            # 简化的威胁评估
            temp_board = [row[:] for row in board]
            temp_board[r][c] = my_color

            for dx, dy in [(0, 1), (1, 0)]:  # 只检查水平和垂直
                count, _ = self.position_evaluator.board_analyzer.analyze_chain_pattern(
                    temp_board, r, c, dx, dy, my_color)
                if count >= 5:
                    score += 10000
                    break
                elif count >= 4:
                    score += 3000
                elif count >= 3:
                    score += 800

            return score

        return sorted(candidates, key=move_priority, reverse=True)

    def _fast_heuristic(self, state, action, agent_id):
        """快速启发式评估函数"""
        if action.get('type') != 'place' or not action.get('coords'):
            return 1000

        r, c = action['coords']
        board = state.board.chips
        agent = state.agents[agent_id]
        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        # 使用缓存
        board_hash = self.cache_manager._get_board_hash(board)
        cached_result = self.cache_manager.get_eval_cache(board_hash, (r, c))
        if cached_result is not None:
            return cached_result

        # 快速评分
        score = self.position_evaluator.fast_comprehensive_score(board, r, c, my_color, enemy_color)
        result = max(1, 1000 - score // 10)

        # 缓存结果
        self.cache_manager.set_eval_cache(board_hash, (r, c), result)
        return result

    def _fast_simulate(self, state, action, game_rule, agent_id):
        """快速状态模拟"""
        try:
            from copy import deepcopy
            new_state = deepcopy(state)
            agent = new_state.agents[agent_id]

            if action['type'] == 'place' and 'coords' in action:
                r, c = action['coords']
                new_state.board.chips[r][c] = agent.colour

            return new_state
        except:
            return state

    def _fast_evaluate_state(self, state, action, agent_id):
        """快速状态评估"""
        if not action or not action.get('coords'):
            return 0

        r, c = action['coords']
        agent = state.agents[agent_id]
        board = state.board.chips
        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        return self.position_evaluator.fast_comprehensive_score(board, r, c, my_color, enemy_color)


# ================================================================================================
# 高效MCTS实现
# ================================================================================================

class MCTSNode:
    """轻量级MCTS节点"""

    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = []
        self.player_id = None

    def ucb1_score(self, exploration_weight=1.414):
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return float('inf')

        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        if not self.children:
            return self
        return max(self.children, key=lambda c: c.ucb1_score())

    def add_child(self, action, state):
        child = MCTSNode(state, action, self)
        if self.player_id in [0, 1]:
            child.player_id = 1 - self.player_id
        else:
            child.player_id = 1 if self.player_id == 0 else 0
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def is_terminal(self):
        """快速终端检查"""
        try:
            board = self.state.board.chips
            # 简化的获胜检查
            for r in range(10):
                for c in range(10):
                    if board[r][c] in ['r', 'b']:
                        color = board[r][c]
                        # 只检查水平和垂直方向
                        for dx, dy in [(0, 1), (1, 0)]:
                            count = 1
                            for direction in [1, -1]:
                                for i in range(1, 5):
                                    x, y = r + dx * direction * i, c + dy * direction * i
                                    if (0 <= x < 10 and 0 <= y < 10 and board[x][y] == color):
                                        count += 1
                                    else:
                                        break
                            if count >= 5:
                                return True
            return False
        except:
            return True


class MCTSEngine:
    """高效MCTS引擎"""

    def __init__(self, position_evaluator, time_manager, config):
        self.position_evaluator = position_evaluator
        self.time_manager = time_manager
        self.config = config

    def mcts_refinement(self, initial_state, candidate_actions, game_rule, agent_id):
        """MCTS改进A*结果"""
        if not candidate_actions or not self.time_manager.should_continue_phase('mcts_refinement'):
            return candidate_actions[0] if candidate_actions else None

        # 限制MCTS的候选动作数量
        top_candidates = candidate_actions[:min(len(candidate_actions), 3)]

        # 为每个候选动作创建MCTS根节点
        action_scores = {}
        remaining_time = self.time_manager.get_remaining_for_phase('mcts_refinement')
        iterations_per_action = max(10, int(remaining_time * 50))  # 动态调整迭代次数

        for action in top_candidates:
            if not self.time_manager.should_continue_phase('mcts_refinement'):
                break

            # 模拟执行动作
            simulated_state = self._fast_simulate(initial_state, action, game_rule, agent_id)
            if not simulated_state:
                continue

            root = MCTSNode(simulated_state, action)
            root.player_id = agent_id

            # 获取合法动作
            try:
                legal_actions = game_rule.getLegalActions(simulated_state, agent_id)
                root.untried_actions = legal_actions[:10] if legal_actions else []  # 限制动作数量
            except:
                root.untried_actions = []

            # 执行MCTS迭代
            for _ in range(iterations_per_action):
                if not self.time_manager.should_continue_phase('mcts_refinement'):
                    break
                self._mcts_iteration(root, game_rule)

            # 计算最终分数
            if root.visits > 0:
                action_scores[action] = root.wins / root.visits
            else:
                action_scores[action] = 0

        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]

        return candidate_actions[0]

    def _mcts_iteration(self, root, game_rule):
        """单次MCTS迭代 - 优化版"""
        # 1. 选择
        node = self._mcts_select(root)

        # 2. 扩展
        if not node.is_terminal() and node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state = self._fast_simulate(node.state, action, game_rule, node.player_id)
            if new_state:
                node = node.add_child(action, new_state)
                # 设置新节点的合法动作
                try:
                    other_player = 1 - node.parent.player_id if node.parent.player_id in [0, 1] else (
                        1 if node.parent.player_id == 0 else 0)
                    legal_actions = game_rule.getLegalActions(new_state, other_player)
                    node.untried_actions = legal_actions[:5] if legal_actions else []  # 限制数量
                except:
                    node.untried_actions = []

        # 3. 模拟 - 简化版
        result = self._fast_simulate_playout(node)

        # 4. 回传
        self._mcts_backpropagate(node, result)

    def _mcts_select(self, root):
        """MCTS选择阶段"""
        node = root
        depth = 0
        while node.children and node.is_fully_expanded() and depth < 3:  # 限制选择深度
            node = node.select_child()
            depth += 1
        return node

    def _fast_simulate_playout(self, node):
        """快速模拟游戏结束"""
        # 简化的评估 - 直接使用启发式评估
        try:
            if not node.action or not node.action.get('coords'):
                return 0.5

            r, c = node.action['coords']
            board = node.state.board.chips
            agent = node.state.agents[node.player_id]
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            score = self.position_evaluator.fast_comprehensive_score(board, r, c, my_color, enemy_color)
            # 归一化到[0,1]
            normalized_score = min(1.0, max(0.0, (score + 1000) / 3000.0))
            return normalized_score
        except:
            return 0.5

    def _mcts_backpropagate(self, node, result):
        """MCTS回传阶段"""
        while node is not None:
            node.update(result)
            node = node.parent

    def _fast_simulate(self, state, action, game_rule, agent_id):
        """快速状态模拟"""
        try:
            from copy import deepcopy
            new_state = deepcopy(state)
            agent = new_state.agents[agent_id]

            if action['type'] == 'place' and 'coords' in action:
                r, c = action['coords']
                new_state.board.chips[r][c] = agent.colour

            return new_state
        except:
            return None


# ================================================================================================
# 主代理类 - A*+MCTS双算法优化版
# ================================================================================================

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)

        # 快速初始化配置和组件
        self.config = GameConfig()
        self.cache_manager = CacheManager()
        self.board_analyzer = BoardAnalyzer(self.config)
        self.board_analyzer.agent_id = _id
        self.threat_analyzer = ThreatAnalyzer(self.board_analyzer, self.config)
        self.position_evaluator = PositionEvaluator(self.board_analyzer, self.threat_analyzer, self.config)
        self.card_selector = CardSelector(self.board_analyzer, self.threat_analyzer, self.position_evaluator,
                                          self.config)

        # 创建搜索引擎
        self.time_manager = TimeManager()
        self.search_engine = SearchEngine(self.position_evaluator, self.time_manager, self.config, self.cache_manager)
        self.mcts_engine = MCTSEngine(self.position_evaluator, self.time_manager, self.config)

        # 时间管理和计数器
        self.counter = itertools.count()
        self.startup_start_time = time.time()
        self.startup_time_limit = 15.0
        self.startup_time_used = False
        self.turn_count = 0

        # 游戏相关
        self.COORDS = COORDS
        self.move_history = []

        # 启动预计算
        if not self.startup_time_used:
            self._precompute_startup_data()

    def _precompute_startup_data(self):
        """利用启动时间进行预计算"""
        if self.startup_time_used:
            return

        startup_elapsed = time.time() - self.startup_start_time
        remaining_startup_time = self.startup_time_limit - startup_elapsed

        if remaining_startup_time <= 2.0:
            self.startup_time_used = True
            return

        # 预计算开局优先位置
        self.opening_priority_positions = [
            (4, 4), (4, 5), (5, 4), (5, 5),  # HOTB
            (3, 3), (3, 6), (6, 3), (6, 6),  # HOTB扩展
            (1, 1), (1, 8), (8, 1), (8, 8),  # 角落附近
            (2, 2), (2, 7), (7, 2), (7, 7)  # 次级战略位置
        ]

        # 预计算常用评估
        self._precompute_common_evaluations()

        self.startup_time_used = True

    def _precompute_common_evaluations(self):
        """预计算常用评估数据"""
        # 预计算开局阶段的位置评估
        self.opening_position_scores = {}
        for r in range(10):
            for c in range(10):
                # 预计算开局位置的基础评分
                score = self.config.position_values.get((r, c), 0)
                if (r, c) in HOTB_COORDS:
                    score += 50  # 开局HOTB额外奖励
                self.opening_position_scores[(r, c)] = score

    def SelectAction(self, actions, game_state):
        """主要决策接口 - A*+MCTS双算法版"""
        self.turn_count += 1

        # 时间管理初始化
        if not self.startup_time_used:
            startup_elapsed = time.time() - self.startup_start_time
            remaining_startup = self.startup_time_limit - startup_elapsed
            time_limit = min(remaining_startup - 0.1, 0.95) if remaining_startup > 0.2 else 0.95
            if remaining_startup <= 0.2:
                self.startup_time_used = True
        else:
            time_limit = 0.95

        self.time_manager = TimeManager(time_limit)
        self.time_manager.start_turn()

        # 评估局面复杂度并调整时间分配
        complexity = self._assess_game_complexity(game_state, actions)
        self.time_manager.adjust_phase_limits(complexity)

        if time_limit < 0.05:
            return self.emergency_quick_decision(actions, game_state)

        # 判断动作类型
        action_type = self.identify_action_type(actions, game_state)
        if action_type == 'card_selection':
            return self.card_selector.select_optimal_card(actions, game_state)
        elif action_type == 'place':
            return self.dual_algorithm_strategy(actions, game_state)
        else:
            return self.dual_algorithm_strategy(actions, game_state)

    def _assess_game_complexity(self, game_state, actions):
        """评估游戏复杂度"""
        try:
            complexity = 0.5  # 基础复杂度

            # 动作数量影响复杂度
            if len(actions) > 20:
                complexity += 0.2
            elif len(actions) < 5:
                complexity -= 0.2

            # 游戏阶段影响复杂度
            if self.turn_count <= 10:
                complexity -= 0.1  # 开局相对简单
            elif self.turn_count > 30:
                complexity += 0.2  # 残局更复杂

            # 棋盘密度影响复杂度
            if hasattr(game_state, 'board'):
                board = game_state.board.chips
                piece_count = sum(1 for row in board for cell in row if cell != '0')
                if piece_count > 40:
                    complexity += 0.1

            return max(0.1, min(1.0, complexity))
        except:
            return 0.5

    def dual_algorithm_strategy(self, actions, game_state):
        """A*+MCTS双算法策略"""
        # 第一阶段：快速决策保底
        quick_decision = self.get_quick_decision(actions, game_state)

        if not self.time_manager.should_continue_phase('astar_search'):
            return quick_decision

        # 第二阶段：A*搜索获得候选方案
        try:
            astar_result = self.search_engine.a_star_search(game_state, actions, self.rule, self.id)
            if not astar_result:
                astar_result = quick_decision
        except Exception:
            astar_result = quick_decision

        # 第三阶段：MCTS优化A*结果
        if self.time_manager.should_continue_phase('mcts_refinement'):
            try:
                # 将A*结果和快速决策结果组合作为MCTS的候选
                candidates = [astar_result]
                if astar_result != quick_decision:
                    candidates.append(quick_decision)

                # 添加其他高分候选
                if len(actions) > 2:
                    additional_candidates = self._get_additional_candidates(actions, game_state, 2)
                    candidates.extend(additional_candidates)

                mcts_result = self.mcts_engine.mcts_refinement(game_state, candidates, self.rule, self.id)
                return mcts_result if mcts_result else astar_result
            except Exception:
                return astar_result

        return astar_result

    def _get_additional_candidates(self, actions, game_state, count):
        """获取额外的高分候选动作"""
        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            scored_actions = []
            for action in actions:
                if not action.get('coords'):
                    continue
                r, c = action['coords']
                score = self.position_evaluator.fast_comprehensive_score(board, r, c, my_color, enemy_color)
                scored_actions.append((score, action))

            scored_actions.sort(key=lambda x: x[0], reverse=True)
            return [action for _, action in scored_actions[:count]]
        except:
            return actions[:count]

    def identify_action_type(self, actions, game_state):
        """判断动作类型：选牌或放牌"""
        if not actions:
            return None

        # 检查游戏状态
        if game_state:
            try:
                agent = game_state.agents[self.id]
                if hasattr(agent, 'hand') and len(agent.hand) < 7:
                    for action in actions:
                        if isinstance(action, dict):
                            if ('draft_card' in action or 'card' in action or
                                    action.get('type') == 'draw' or action.get('type') == 'select_card'):
                                return 'card_selection'
            except:
                pass

        # 检查是否为放置动作
        for action in actions:
            if isinstance(action, dict):
                if 'coords' in action and action.get('type') == 'place':
                    return 'place'
                elif hasattr(action, 'coords') or 'coords' in str(action):
                    return 'place'

        return 'place'

    def get_quick_decision(self, actions, game_state):
        """快速决策方法 - 优化版"""
        if not actions:
            return None

        best_action = actions[0]
        best_score = float('-inf')

        agent, board, is_valid = self.safe_get_game_state_info(game_state)
        if not is_valid:
            return actions[0]

        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        # 开局检查 - 使用预计算数据
        if self.turn_count <= 6:
            for action in actions[:min(len(actions), 5)]:
                if not action.get('coords'):
                    continue
                r, c = action['coords']
                if (r, c) in self.opening_priority_positions:
                    return action

        # 快速评估候选动作
        for action in actions[:min(len(actions), 8)]:  # 增加候选数量
            if not action.get('coords'):
                continue

            r, c = action['coords']

            # 使用预计算的快速评分
            if self.turn_count <= 10 and (r, c) in self.opening_position_scores:
                score = self.opening_position_scores[(r, c)]
            else:
                score = self.position_evaluator.fast_comprehensive_score(board, r, c, my_color, enemy_color) // 10

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def emergency_quick_decision(self, actions, game_state):
        """紧急快速决策 - 进一步优化"""
        if not actions:
            return None

        # 优先选择HOTB位置
        for action in actions[:min(len(actions), 3)]:
            if action.get('coords') and action['coords'] in HOTB_COORDS:
                return action

        # 使用预计算的开局位置
        if hasattr(self, 'opening_priority_positions'):
            for action in actions[:min(len(actions), 5)]:
                if action.get('coords') and action['coords'] in self.opening_priority_positions:
                    return action

        return actions[0]

    def safe_get_game_state_info(self, game_state):
        """安全获取游戏状态信息"""
        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            return agent, board, True
        except (AttributeError, KeyError, IndexError):
            return None, None, False


# ================================================================================================
# 保持兼容性的辅助类
# ================================================================================================

class WeightLearningSystem:
    """权重学习系统（保持接口兼容）"""

    def __init__(self):
        self.decision_records = []
        self.game_outcomes = []
        self.learning_rate = 0.1


class GameStatistics:
    """游戏统计系统（保持接口兼容）"""

    def __init__(self):
        self.move_count = 0
        self.decision_times = []
        self.evaluation_history = []


class OpponentModel:
    """对手模型（保持接口兼容）"""

    def __init__(self):
        self.opponent_moves = []
        self.position_preferences = {}
        self.style_indicators = {
            'aggressive': 0.5,
            'defensive': 0.5,
            'strategic': 0.5
        }


class OpeningBook:
    """开局库（保持接口兼容）"""

    def __init__(self):
        self.HOTB_CARDS = ['5h', '4h', '2h', '3h']
        self.HOTB_POSITIONS = [(4, 4), (4, 5), (5, 4), (5, 5)]