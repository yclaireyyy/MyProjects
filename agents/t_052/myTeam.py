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
# 核心配置类 - 游戏参数和权重管理
# ================================================================================================

class GameConfig:
    """游戏配置和权重管理"""

    def __init__(self):
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


# ================================================================================================
# 缓存和时间管理系统
# ================================================================================================

class CacheManager:
    """评估结果缓存管理"""

    def __init__(self, max_size=2000):
        self.cache = {}
        self.max_size = max_size
        self.hit_count = 0
        self.total_count = 0

    def get(self, board, action, cache_type='heuristic'):
        try:
            if not action.get('coords'):
                return None
            r, c = action['coords']
            board_hash = self._get_board_hash(board)
            cache_key = (board_hash, r, c, cache_type)

            self.total_count += 1
            if cache_key in self.cache:
                self.hit_count += 1
                return self.cache[cache_key]
            return None
        except:
            return None

    def set(self, board, action, result, cache_type='heuristic'):
        try:
            if not action.get('coords') or result is None:
                return
            r, c = action['coords']
            board_hash = self._get_board_hash(board)
            cache_key = (board_hash, r, c, cache_type)

            if len(self.cache) >= self.max_size:
                keys_to_remove = list(self.cache.keys())[:self.max_size // 3]
                for key in keys_to_remove:
                    del self.cache[key]

            self.cache[cache_key] = result
        except:
            pass

    def _get_board_hash(self, board):
        return hash(''.join(''.join(row) for row in board))


class TimeManager:
    """时间管理系统"""

    def __init__(self, total_time_limit=0.95):
        self.total_limit = total_time_limit
        self.start_time = None
        self.safety_buffer = 0.05
        self.phase_limits = {
            'quick_decision': 0.15,
            'heuristic_search': 0.50,
            'monte_carlo': 0.80
        }

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


# ================================================================================================
# 棋盘分析和评估系统
# ================================================================================================

class BoardAnalyzer:
    """棋盘分析核心功能"""

    def __init__(self, config):
        self.config = config
        self.CORNER_POSITIONS = [(0, 0), (0, 9), (9, 0), (9, 9)]
        self.direction_vectors = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def analyze_chain_pattern(self, board, r, c, dx, dy, color):
        """角落感知的连子模式分析"""
        count = 1  # 包含当前位置
        openings = 0

        # 正向计数
        pos_open = False
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color or self.is_corner_position(x, y):
                    count += 1
                elif board[x][y] == '0':
                    pos_open = True
                    break
                else:
                    break
            else:
                break

        # 负向计数
        neg_open = False
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color or self.is_corner_position(x, y):
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

    def check_corner_support(self, board, r, c, dx, dy, color):
        """检查连子方向上是否包含角落位置"""
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if not (0 <= x < 10 and 0 <= y < 10):
                    break
                if self.is_corner_position(x, y):
                    return True
                if board[x][y] != color and board[x][y] != '0' and not self.is_corner_position(x, y):
                    break
        return False

    def is_corner_position(self, r, c):
        return (r, c) in self.CORNER_POSITIONS


class ThreatAnalyzer:
    """威胁分析专用类"""

    def __init__(self, board_analyzer, config):
        self.board_analyzer = board_analyzer
        self.config = config

    def classify_threat(self, count, openings, corner_support):
        """威胁分类"""
        if count >= 5:
            return 'win'
        elif count >= 4:
            if openings >= 2 or (openings >= 1 and corner_support):
                return 'critical_win'
            elif openings >= 1 or corner_support:
                return 'major_threat'
            else:
                return 'blocked_threat'
        elif count >= 3:
            if openings >= 2:
                return 'double_threat'
            elif openings >= 1 or corner_support:
                return 'active_threat'
            else:
                return 'potential_threat'
        elif count >= 2:
            if openings >= 2:
                return 'development'
            else:
                return 'weak_connection'
        else:
            return 'none'

    def calculate_threat_urgency(self, threat_type, count, openings):
        """计算威胁紧迫性"""
        urgency_map = {
            'win': 10000, 'critical_win': 9000, 'major_threat': 7000,
            'double_threat': 5000, 'active_threat': 3000, 'blocked_threat': 1500,
            'potential_threat': 800, 'development': 400, 'weak_connection': 100,
            'none': 0
        }
        return urgency_map.get(threat_type, 0)

    def threat_analysis(self, board, r, c, color, enemy_color):
        """完整威胁分析"""
        threats = []
        for dx, dy in self.board_analyzer.direction_vectors:
            temp_board = [row[:] for row in board]
            temp_board[r][c] = color

            count, openings = self.board_analyzer.analyze_chain_pattern(temp_board, r, c, dx, dy, color)
            corner_support = self.board_analyzer.check_corner_support(temp_board, r, c, dx, dy, color)

            threat_type = self.classify_threat(count, openings, corner_support)
            if threat_type != 'none':
                threats.append({
                    'direction': (dx, dy),
                    'type': threat_type,
                    'count': count,
                    'openings': openings,
                    'corner_support': corner_support,
                    'urgency': self.calculate_threat_urgency(threat_type, count, openings)
                })
        return threats


# ================================================================================================
# 评估系统
# ================================================================================================

class PositionEvaluator:
    """位置评估系统"""

    def __init__(self, board_analyzer, threat_analyzer, config):
        self.board_analyzer = board_analyzer
        self.threat_analyzer = threat_analyzer
        self.config = config
        self._precompute_position_values()

    def _precompute_position_values(self):
        """预计算位置价值"""
        self.position_values = {}
        for r in range(10):
            for c in range(10):
                self.position_values[(r, c)] = self._calculate_base_position_value(r, c)

    def _calculate_base_position_value(self, r, c):
        """计算位置基础价值"""
        value = 0
        if (r, c) in HOTB_COORDS:
            value += 120

        distance_to_center = abs(r - 4.5) + abs(c - 4.5)
        value += max(0, 25 - distance_to_center * 2.5)

        for corner_r, corner_c in self.board_analyzer.CORNER_POSITIONS:
            corner_distance = max(abs(r - corner_r), abs(c - corner_c))
            if corner_distance <= 3:
                value += max(0, 20 - corner_distance * 5)

        return value

    def chain_score(self, board, r, c, color):
        """连子评分系统"""
        total_score = 0
        for dx, dy in self.board_analyzer.direction_vectors:
            count, openings = self.board_analyzer.analyze_chain_pattern(board, r, c, dx, dy, color)
            corner_support = self.board_analyzer.check_corner_support(board, r, c, dx, dy, color)

            # 指数级评分逻辑
            if count >= 5:
                total_score += 100000
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
            elif count == 2:
                if openings >= 2:
                    total_score += 200
                elif corner_support and openings >= 1:
                    total_score += 150
                elif openings >= 1:
                    total_score += 80
                elif corner_support:
                    total_score += 100
                else:
                    total_score += 20
            else:
                if corner_support:
                    total_score += 30
                else:
                    total_score += 10

        return total_score

    def block_enemy_score(self, board, r, c, enemy_color):
        """阻断敌方评分"""
        score = 0
        for dx, dy in self.board_analyzer.direction_vectors:
            enemy_threat = self.calculate_enemy_threat_in_direction(board, r, c, dx, dy, enemy_color)

            if enemy_threat >= 5:
                score += 15000
            elif enemy_threat >= 4:
                score += 8000
            elif enemy_threat >= 3:
                score += 2000
            elif enemy_threat >= 2:
                score += 500

        return score

    def calculate_enemy_threat_in_direction(self, board, r, c, dx, dy, enemy_color):
        """计算特定方向的敌方威胁等级"""
        threat_level = 0
        for direction in [1, -1]:
            count = 0
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    cell_value = board[x][y]
                    if x == r and y == c:
                        cell_value = enemy_color
                    if cell_value == enemy_color or self.board_analyzer.is_corner_position(x, y):
                        count += 1
                    elif cell_value == '0':
                        break
                    else:
                        break
                else:
                    break
            threat_level = max(threat_level, count)

        return threat_level

    def hotb_score(self, board, color):
        """HOTB控制评分"""
        enemy = 'r' if color == 'b' else 'b'
        score = 0
        controlled = 0
        enemy_controlled = 0

        for r, c in HOTB_COORDS:
            if board[r][c] == color:
                controlled += 1
                score += 50
            elif board[r][c] == enemy:
                enemy_controlled += 1
                score -= 80

        if controlled >= 3:
            score += 200
        if controlled == 4:
            score += 500
        if enemy_controlled >= 2:
            score -= 150

        return score


# ================================================================================================
# 卡片选择系统
# ================================================================================================

class CardSelector:
    """卡片选择策略"""

    def __init__(self, board_analyzer, threat_analyzer, position_evaluator, config):
        self.board_analyzer = board_analyzer
        self.threat_analyzer = threat_analyzer
        self.position_evaluator = position_evaluator
        self.config = config

    def select_optimal_card(self, card_actions, game_state):
        """选择最优卡片"""
        if not card_actions:
            return None

        agent, board, is_valid = self._safe_get_game_state_info(game_state)
        if not is_valid:
            return card_actions[0]

        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        best_card = None
        best_score = float('-inf')

        for card_action in card_actions:
            score = self._evaluate_card_value(card_action, board, my_color, enemy_color, game_state)
            if score > best_score:
                best_score = score
                best_card = card_action

        return best_card or card_actions[0]

    def _evaluate_card_value(self, card_action, board, my_color, enemy_color, game_state):
        """评估单张卡片价值"""
        card_positions = self._get_card_board_positions(card_action, game_state)
        if not card_positions:
            return 0

        total_score = 0

        # 胜利检查
        if self._check_immediate_win_potential(card_positions, board, my_color):
            return 100000

        # 各种评估维度
        total_score += self._calculate_immediate_play_value(card_positions, board, my_color)
        total_score += self._calculate_strategic_position_value(card_positions, board)
        total_score += self._calculate_blocking_value(card_positions, board, enemy_color)
        total_score += self._calculate_flexibility_value(card_positions, board)

        return total_score

    def _get_card_board_positions(self, card_action, game_state):
        """获取卡片对应的棋盘位置"""
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

    def _check_immediate_win_potential(self, positions, board, color):
        """检查立即获胜潜力"""
        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                temp_board = [row[:] for row in board]
                temp_board[r][c] = color

                for dx, dy in self.board_analyzer.direction_vectors:
                    count, _ = self.board_analyzer.analyze_chain_pattern(temp_board, r, c, dx, dy, color)
                    if count >= 5:
                        return True
        return False

    def _calculate_immediate_play_value(self, positions, board, color):
        """计算立即可用价值"""
        value = 0
        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                position_value = self.position_evaluator.position_values.get((r, c), 0)
                chain_value = self.position_evaluator.chain_score(board, r, c, color)
                value += position_value + chain_value
        return value

    def _calculate_strategic_position_value(self, positions, board):
        """计算战略位置价值"""
        value = 0
        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10:
                value += self.position_evaluator.position_values.get((r, c), 0)
        return value

    def _calculate_blocking_value(self, positions, board, enemy_color):
        """计算阻断价值"""
        value = 0
        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                blocking_score = self.position_evaluator.block_enemy_score(board, r, c, enemy_color)
                value += blocking_score * 0.8
        return value

    def _calculate_flexibility_value(self, positions, board):
        """计算灵活性价值"""
        available_positions = sum(1 for r, c in positions
                                  if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0')
        return available_positions * 10


# ================================================================================================
# 搜索引擎
# ================================================================================================

class SearchEngine:
    """搜索算法引擎"""

    def __init__(self, position_evaluator, time_manager, config):
        self.position_evaluator = position_evaluator
        self.time_manager = time_manager
        self.config = config
        self.counter = itertools.count()

    def a_star_search(self, initial_state, candidate_moves, game_rule, agent_id):
        """A*搜索算法"""
        pending = []
        seen_states = set()
        best_sequence = []
        top_reward = float('-inf')

        # 自适应参数
        remaining_time = self.time_manager.get_remaining_for_phase('heuristic_search')
        if remaining_time > 0.4:
            max_candidates, max_expansions, depth_limit = 6, 40, 3
        elif remaining_time > 0.2:
            max_candidates, max_expansions, depth_limit = 4, 20, 2
        else:
            max_candidates, max_expansions, depth_limit = 3, 10, 1

        # 预排序候选动作
        candidate_moves = self._smart_sort_candidates(initial_state, candidate_moves, agent_id)

        for move in candidate_moves[:max_candidates]:
            g = 1
            h = self._heuristic(initial_state, move, agent_id)
            f = g + h
            heapq.heappush(pending, (f, next(self.counter), g, h,
                                     self._fast_simulate(initial_state, move, game_rule, agent_id), [move]))

        expansions_count = 0
        while (pending and expansions_count < max_expansions and
               self.time_manager.should_continue_phase('heuristic_search')):

            f, _, g, h, current_state, move_history = heapq.heappop(pending)

            if g > depth_limit:
                continue

            state_sig = self._create_state_signature(current_state, move_history[-1])
            if state_sig in seen_states:
                continue
            seen_states.add(state_sig)

            expansions_count += 1

            reward = self._evaluate_state(current_state, move_history[-1], agent_id)
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history

            if (g < depth_limit and
                    self.time_manager.get_remaining_for_phase('heuristic_search') > 0.05):

                next_steps = game_rule.getLegalActions(current_state, agent_id)
                next_steps = self._smart_sort_candidates(current_state, next_steps, agent_id)

                for next_move in next_steps[:2]:
                    next_g = g + 1
                    next_h = self._heuristic(current_state, next_move, agent_id)
                    heapq.heappush(pending, (
                        next_g + next_h, next(self.counter),
                        next_g, next_h,
                        self._fast_simulate(current_state, next_move, game_rule, agent_id),
                        move_history + [next_move]
                    ))

        return best_sequence[0] if best_sequence else None

    def _smart_sort_candidates(self, state, candidates, agent_id):
        """智能排序候选动作"""

        def move_priority(action):
            if not action.get('coords'):
                return 0
            r, c = action['coords']

            score = self.position_evaluator.position_values.get((r, c), 0)

            # 快速威胁评估
            agent = state.agents[agent_id]
            my_color = agent.colour
            temp_board = [row[:] for row in state.board.chips]
            temp_board[r][c] = my_color

            for dx, dy in self.position_evaluator.board_analyzer.direction_vectors:
                count, _ = self.position_evaluator.board_analyzer.analyze_chain_pattern(
                    temp_board, r, c, dx, dy, my_color)

                if count >= 5:
                    score += 10000
                    break
                elif count >= 4:
                    score += 3000
                elif count >= 3:
                    score += 800
                elif count >= 2:
                    score += 200

            return score

        return sorted(candidates, key=move_priority, reverse=True)

    def _heuristic(self, state, action, agent_id):
        """启发式评估函数"""
        if action.get('type') != 'place' or not action.get('coords'):
            return 1000

        r, c = action['coords']
        board = [row[:] for row in state.board.chips]
        me = state.agents[agent_id]
        color = me.colour
        enemy = 'r' if color == 'b' else 'b'

        board[r][c] = color
        score = 0

        # 基础评估
        score += self.position_evaluator.position_values.get((r, c), 0)
        score += self.position_evaluator.chain_score(board, r, c, color)
        score += self.position_evaluator.block_enemy_score(board, r, c, enemy)
        score += self.position_evaluator.hotb_score(board, color)

        return max(1, 1000 - score)

    def _fast_simulate(self, state, action, game_rule, agent_id):
        """快速状态模拟"""
        new_state = self._custom_copy(state)
        agent = new_state.agents[agent_id]

        if action['type'] == 'place' and 'coords' in action:
            r, c = action['coords']
            new_state.board.chips[r][c] = agent.colour

        return new_state

    def _custom_copy(self, state):
        """自定义状态拷贝"""
        from copy import deepcopy
        return deepcopy(state)

    def _create_state_signature(self, state, last_move):
        """创建状态签名"""
        try:
            board = state.board.chips
            board_str = ''.join(''.join(row) for row in board)
            board_hash = hash(board_str)

            move_info = None
            if last_move and 'coords' in last_move:
                move_info = last_move['coords']

            return (board_hash, move_info)
        except:
            return (hash(str(state)), str(last_move))

    def _evaluate_state(self, state, action, agent_id):
        """评估状态"""
        agent = state.agents[agent_id]
        board = [row[:] for row in state.board.chips]

        if action and action.get('coords'):
            r, c = action['coords']
            board[r][c] = agent.colour

        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        score = 0
        score += self._score_all_chains(board, my_color) * 2.5
        score -= self._score_all_chains(board, enemy_color) * 2.0
        score += self.position_evaluator.hotb_score(board, my_color)

        return score

    def _score_all_chains(self, board, color):
        """评估所有连子"""
        total_score = 0
        processed = set()

        for r in range(10):
            for c in range(10):
                if (r, c) in processed or board[r][c] != color:
                    continue

                max_chain_score = 0
                for dx, dy in self.position_evaluator.board_analyzer.direction_vectors:
                    count, openings = self.position_evaluator.board_analyzer.analyze_chain_pattern(
                        board, r, c, dx, dy, color)
                    corner_support = self.position_evaluator.board_analyzer.check_corner_support(
                        board, r, c, dx, dy, color)

                    chain_score = self._convert_chain_to_score(count, openings, corner_support)
                    max_chain_score = max(max_chain_score, chain_score)

                total_score += max_chain_score
                processed.add((r, c))

        return total_score

    def _convert_chain_to_score(self, count, openings, corner_support):
        """连子转换为分数"""
        if count >= 5:
            return 10000
        elif count == 4:
            return 2000 if (openings > 0 or corner_support) else 400
        elif count == 3:
            return 300 if (openings > 0 or corner_support) else 60
        elif count == 2:
            return 40 if openings > 0 else 10
        else:
            return 0


# ================================================================================================
# MCTS相关类
# ================================================================================================

class MCTSNode:
    """MCTS节点"""

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
        """检查是否为终端节点"""
        try:
            if not hasattr(self, 'state') or not self.state:
                return True

            # 简化的终端检查
            board = self.state.board.chips
            for r in range(10):
                for c in range(10):
                    if board[r][c] in ['r', 'b']:
                        # 简单检查是否有5连
                        color = board[r][c]
                        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                            count = 1
                            for direction in [1, -1]:
                                for i in range(1, 5):
                                    x, y = r + dx * direction * i, c + dy * direction * i
                                    if (0 <= x < 10 and 0 <= y < 10 and
                                            board[x][y] == color):
                                        count += 1
                                    else:
                                        break
                            if count >= 5:
                                return True
            return False
        except:
            return True


# ================================================================================================
# 主代理类
# ================================================================================================

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)

        # 初始化配置和组件
        self.config = GameConfig()
        self.cache_manager = CacheManager()
        self.board_analyzer = BoardAnalyzer(self.config)
        self.board_analyzer.agent_id = _id  # 设置agent_id
        self.threat_analyzer = ThreatAnalyzer(self.board_analyzer, self.config)
        self.position_evaluator = PositionEvaluator(self.board_analyzer, self.threat_analyzer, self.config)
        self.card_selector = CardSelector(self.board_analyzer, self.threat_analyzer, self.position_evaluator,
                                          self.config)

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

        self.startup_time_used = True

    # ===========================================================================================
    # 主要接口方法
    # ===========================================================================================

    def SelectAction(self, actions, game_state):
        """主要决策接口"""
        self.turn_count += 1

        # 时间管理初始化
        if not self.startup_time_used:
            startup_elapsed = time.time() - self.startup_start_time
            remaining_startup = self.startup_time_limit - startup_elapsed
            time_limit = min(remaining_startup - 0.1, 0.98) if remaining_startup > 0.2 else 0.98
            if remaining_startup <= 0.2:
                self.startup_time_used = True
        else:
            time_limit = 0.98

        self.time_manager = TimeManager(time_limit)
        self.time_manager.start_turn()

        if time_limit < 0.05:
            return self.emergency_quick_decision(actions, game_state)

        # 判断动作类型
        action_type = self.identify_action_type(actions, game_state)
        if action_type == 'card_selection':
            return self.card_selector.select_optimal_card(actions, game_state)
        elif action_type == 'place':
            return self.place_strategy(actions, game_state)
        else:
            return self.place_strategy(actions, game_state)

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

    def place_strategy(self, actions, game_state):
        """放牌策略"""
        # 快速决策作为保底
        quick_decision = self.get_quick_decision(actions, game_state)

        if not self.time_manager.should_continue_phase('heuristic_search'):
            return quick_decision

        # 尝试更好的决策
        try:
            search_engine = SearchEngine(self.position_evaluator, self.time_manager, self.config)
            better_decision = search_engine.a_star_search(game_state, actions, self.rule, self.id)
            return better_decision if better_decision else quick_decision
        except Exception:
            return quick_decision

    def get_quick_decision(self, actions, game_state):
        """快速决策方法"""
        if not actions:
            return None

        best_action = actions[0]
        best_score = float('-inf')

        agent, board, is_valid = self.safe_get_game_state_info(game_state)
        if not is_valid:
            return actions[0]

        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        # 开局检查
        if self.turn_count <= 6:
            for action in actions[:min(len(actions), 5)]:
                if not action.get('coords'):
                    continue
                r, c = action['coords']
                if (r, c) in self.opening_priority_positions:
                    return action

        for action in actions[:min(len(actions), 6)]:
            if not action.get('coords'):
                continue

            r, c = action['coords']
            score = 0

            # 快速评估
            if (r, c) in HOTB_COORDS:
                score += 150
            elif 3 <= r <= 6 and 3 <= c <= 6:
                score += 80

            # 快速连子检查
            temp_board = [row[:] for row in board]
            temp_board[r][c] = my_color

            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count = 1
                for direction in [1, -1]:
                    for i in range(1, 5):
                        x, y = r + dx * direction * i, c + dy * direction * i
                        if (0 <= x < 10 and 0 <= y < 10 and
                                (temp_board[x][y] == my_color or self.board_analyzer.is_corner_position(x, y))):
                            count += 1
                        else:
                            break

                if count >= 5:
                    return action
                elif count >= 4:
                    score += 2000
                elif count >= 3:
                    score += 500

            # 快速阻断检查
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                enemy_temp = [row[:] for row in board]
                enemy_temp[r][c] = enemy_color

                enemy_count = 1
                for direction in [1, -1]:
                    for i in range(1, 5):
                        x, y = r + dx * direction * i, c + dy * direction * i
                        if (0 <= x < 10 and 0 <= y < 10 and
                                (enemy_temp[x][y] == enemy_color or self.board_analyzer.is_corner_position(x, y))):
                            enemy_count += 1
                        else:
                            break

                if enemy_count >= 4:
                    score += 1500

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def emergency_quick_decision(self, actions, game_state):
        """紧急快速决策"""
        if not actions:
            return None

        agent, board, is_valid = self.safe_get_game_state_info(game_state)
        if not is_valid:
            return actions[0]

        my_color = agent.colour
        best_action = actions[0]
        best_score = float('-inf')

        for action in actions[:min(len(actions), 3)]:
            if not action.get('coords'):
                continue

            r, c = action['coords']
            score = 0

            # HOTB最高优先级
            if (r, c) in HOTB_COORDS:
                score += 300

            # 检查直接获胜
            temp_board = [row[:] for row in board]
            temp_board[r][c] = my_color
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count = 1
                for direction in [1, -1]:
                    for i in range(1, 4):
                        x, y = r + dx * direction * i, c + dy * direction * i
                        if (0 <= x < 10 and 0 <= y < 10 and
                                (temp_board[x][y] == my_color or self.board_analyzer.is_corner_position(x, y))):
                            count += 1
                        else:
                            break
                if count >= 5:
                    return action

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def safe_get_game_state_info(self, game_state):
        """安全获取游戏状态信息"""
        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            return agent, board, True
        except (AttributeError, KeyError, IndexError):
            return None, None, False


# ================================================================================================
# 辅助类和函数（保持兼容性）
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