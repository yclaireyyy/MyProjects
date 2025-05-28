from collections import namedtuple
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
from Sequence.sequence_model import COORDS
import heapq
import time
import itertools
import random
import math

MAX_THINK_TIME = 0.95
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]
CORNER_POSITIONS = [(0, 0), (0, 9), (9, 0), (9, 9)]


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.counter = itertools.count()

        # 时间管理
        self.startup_start = time.time()
        self.startup_used = False
        self.turn_count = 0

        # 预计算核心数据
        self._precompute_data()

        # 智能缓存系统
        self.eval_cache = {}
        self.threat_cache = {}
        self.cache_size = 0
        self.max_cache = 300

    def _precompute_data(self):
        """预计算关键数据"""
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 增强的位置权重系统
        self.position_weights = {}
        for r in range(10):
            for c in range(10):
                weight = 1
                # HOTB权重
                if (r, c) in HOTB_COORDS:
                    weight += 8
                # 中心权重 - 增强
                center_dist = abs(r - 4.5) + abs(c - 4.5)
                weight += max(0, 4 - center_dist * 0.4)
                # 角落战略权重
                for cr, cc in CORNER_POSITIONS:
                    corner_dist = max(abs(r - cr), abs(c - cc))
                    if corner_dist <= 3:
                        weight += max(0, 3 - corner_dist * 0.8)
                self.position_weights[(r, c)] = weight

        # 威胁等级映射 - 保留原版本的复杂度
        self.threat_levels = {
            'win': 50000, 'critical_win': 30000, 'major_threat': 15000,
            'double_threat': 8000, 'active_threat': 4000, 'blocked_threat': 1000,
            'potential_threat': 500, 'development': 200, 'weak_connection': 50,
            'none': 0
        }

        # 开局优先位置
        self.opening_priority = HOTB_COORDS + [
            (3, 3), (3, 6), (6, 3), (6, 6),
            (2, 2), (2, 7), (7, 2), (7, 7),
            (1, 1), (1, 8), (8, 1), (8, 8)
        ]

    def SelectAction(self, actions, game_state):
        """主决策入口"""
        self.turn_count += 1
        start_time = time.time()

        # 动态时间管理
        if not self.startup_used:
            elapsed = time.time() - self.startup_start
            remaining = 15.0 - elapsed
            time_limit = min(remaining - 0.1, 0.95) if remaining > 0.2 else 0.95
            if remaining <= 0.2:
                self.startup_used = True
        else:
            time_limit = 0.95

        if not actions or time_limit < 0.05:
            return self._emergency_decision(actions, game_state)

        # 判断动作类型
        if self._is_card_selection(actions, game_state):
            return self._enhanced_card_selection(actions, game_state)

        return self._balanced_dual_search(actions, game_state, start_time, time_limit)

    def _balanced_dual_search(self, actions, game_state, start_time, time_limit):
        """平衡的双算法搜索"""
        # 评估局面复杂度
        complexity = self._assess_complexity(game_state, actions)

        # 动态时间分配
        if complexity > 0.7:  # 复杂局面
            quick_ratio, astar_ratio = 0.15, 0.70
        elif complexity < 0.3:  # 简单局面
            quick_ratio, astar_ratio = 0.25, 0.60
        else:  # 中等复杂度
            quick_ratio, astar_ratio = 0.20, 0.65

        # 阶段1：增强快速评估
        quick_result = self._enhanced_quick_evaluation(actions, game_state)

        if time.time() - start_time > time_limit * quick_ratio:
            return quick_result

        # 阶段2：增强A*搜索
        try:
            astar_result = self._enhanced_astar(actions, game_state, start_time, time_limit * astar_ratio, complexity)
            if not astar_result:
                astar_result = quick_result
        except:
            astar_result = quick_result

        if time.time() - start_time > time_limit * astar_ratio:
            return astar_result

        # 阶段3：改进MCTS
        try:
            candidates = self._select_mcts_candidates(actions, game_state, astar_result, quick_result)
            mcts_result = self._improved_mcts(candidates, game_state, start_time, time_limit)
            return mcts_result if mcts_result else astar_result
        except:
            return astar_result

    def _enhanced_quick_evaluation(self, actions, game_state):
        """增强的快速评估"""
        if not actions:
            return None

        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'
        except:
            return actions[0]

        # 开局策略增强
        if self.turn_count <= 10:
            for action in actions:
                coords = action.get('coords')
                if coords and coords in self.opening_priority:
                    # 验证这是个好的开局位置
                    if self._quick_position_check(board, coords, my_color):
                        return action

        # 紧急威胁检查
        for action in actions:
            coords = action.get('coords')
            if coords:
                if self._check_immediate_win(board, coords, my_color):
                    return action
                if self._check_critical_block(board, coords, enemy_color):
                    return action

        # 增强评估
        best_action = actions[0]
        best_score = -999999

        eval_limit = min(len(actions), 8)
        for action in actions[:eval_limit]:
            coords = action.get('coords')
            if not coords:
                continue

            score = self._comprehensive_position_score(board, coords, my_color, enemy_color)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _comprehensive_position_score(self, board, coords, my_color, enemy_color):
        """综合位置评分 - 保留原版本核心逻辑"""
        r, c = coords
        if not (0 <= r < 10 and 0 <= c < 10):
            return -999999

        # 缓存检查
        cache_key = (self._board_hash(board), r, c, my_color)
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        score = self.position_weights.get((r, c), 1)

        # 增强的连子分析
        for dx, dy in self.directions:
            # 我方连子分析 - 保留原版本的指数级评分
            my_count, my_openings = self._analyze_chain_enhanced(board, r, c, dx, dy, my_color)
            corner_support = self._check_corner_support(board, r, c, dx, dy, my_color)

            # 威胁分类 - 使用原版本逻辑
            threat_type, threat_score = self._classify_threat(my_count, my_openings, corner_support)
            score += threat_score

            # 阻断评分 - 增强
            enemy_threat = self._analyze_enemy_threat(board, r, c, dx, dy, enemy_color)
            if enemy_threat >= 4:
                score += 12000
            elif enemy_threat >= 3:
                score += 3000
            elif enemy_threat >= 2:
                score += 800

        # HOTB控制增强
        if (r, c) in HOTB_COORDS:
            score += self._enhanced_hotb_bonus(board, my_color, enemy_color)

        # 缓存管理
        if self.cache_size >= self.max_cache:
            self._smart_cache_cleanup()

        self.eval_cache[cache_key] = score
        self.cache_size += 1

        return score

    def _analyze_chain_enhanced(self, board, r, c, dx, dy, color):
        """增强的连子分析 - 保留原版本精度"""
        count = 1
        openings = 0

        # 正向分析
        pos_open = False
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color or (x, y) in CORNER_POSITIONS:
                    count += 1
                elif board[x][y] == '0':
                    pos_open = True
                    break
                else:
                    break
            else:
                break

        # 负向分析
        neg_open = False
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color or (x, y) in CORNER_POSITIONS:
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

    def _classify_threat(self, count, openings, corner_support):
        """威胁分类 - 保留原版本逻辑"""
        if count >= 5:
            return 'win', self.threat_levels['win']
        elif count >= 4:
            if openings >= 2 or (openings >= 1 and corner_support):
                return 'critical_win', self.threat_levels['critical_win']
            elif openings >= 1 or corner_support:
                return 'major_threat', self.threat_levels['major_threat']
            else:
                return 'blocked_threat', self.threat_levels['blocked_threat']
        elif count >= 3:
            if openings >= 2:
                return 'double_threat', self.threat_levels['double_threat']
            elif openings >= 1 or corner_support:
                return 'active_threat', self.threat_levels['active_threat']
            else:
                return 'potential_threat', self.threat_levels['potential_threat']
        elif count >= 2:
            if openings >= 2:
                return 'development', self.threat_levels['development']
            else:
                return 'weak_connection', self.threat_levels['weak_connection']
        else:
            return 'none', self.threat_levels['none']

    def _check_corner_support(self, board, r, c, dx, dy, color):
        """角落支持检查 - 保留原版本逻辑"""
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if not (0 <= x < 10 and 0 <= y < 10):
                    break
                if (x, y) in CORNER_POSITIONS:
                    return True
                if (board[x][y] != color and board[x][y] != '0' and
                        (x, y) not in CORNER_POSITIONS):
                    break
        return False

    def _enhanced_astar(self, actions, game_state, start_time, time_limit, complexity):
        """增强A*搜索"""
        if not actions:
            return None

        # 根据复杂度动态调整参数
        if complexity > 0.7:
            max_candidates, max_expansions, depth_limit = 4, 18, 2
        elif complexity > 0.4:
            max_candidates, max_expansions, depth_limit = 3, 12, 1
        else:
            max_candidates, max_expansions, depth_limit = 2, 8, 1

        heap = []
        visited = set()

        # 智能排序
        sorted_actions = self._intelligent_sort(actions, game_state)

        # 初始化
        for action in sorted_actions[:max_candidates]:
            h_score = self._enhanced_heuristic(game_state, action)
            heapq.heappush(heap, (h_score, next(self.counter), 0, action, [action]))

        best_sequence = []
        best_reward = float('-inf')
        expansions = 0

        while (heap and expansions < max_expansions and
               time.time() - start_time < time_limit):

            f_score, _, depth, action, sequence = heapq.heappop(heap)

            # 改进的状态去重
            state_key = self._get_state_signature(game_state, sequence)
            if state_key in visited:
                continue
            visited.add(state_key)

            expansions += 1

            # 评估当前序列
            reward = self._evaluate_sequence(game_state, sequence)
            if reward > best_reward:
                best_reward = reward
                best_sequence = sequence

            # 深度控制
            if depth >= depth_limit:
                continue

            # 智能扩展
            try:
                next_state = self._simulate_action(game_state, action)
                if next_state:
                    next_actions = self.rule.getLegalActions(next_state, self.id)
                    if next_actions:
                        sorted_next = self._intelligent_sort(next_actions, next_state)
                        expand_count = 3 if complexity > 0.6 else 2

                        for next_action in sorted_next[:expand_count]:
                            new_sequence = sequence + [next_action]
                            h_score = self._enhanced_heuristic(next_state, next_action)
                            heapq.heappush(heap, (
                                depth + 1 + h_score, next(self.counter),
                                depth + 1, next_action, new_sequence
                            ))
            except:
                continue

        return best_sequence[0] if best_sequence else actions[0]

    def _improved_mcts(self, candidates, game_state, start_time, time_limit):
        """改进的MCTS"""
        if not candidates:
            return None

        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.03:
            return candidates[0]

        action_scores = {}
        iterations_per_action = max(8, int(remaining_time * 40))

        for action in candidates:
            if time.time() - start_time >= time_limit:
                break

            total_reward = 0
            simulations = 0

            for _ in range(iterations_per_action):
                if time.time() - start_time >= time_limit:
                    break

                # 改进的模拟
                reward = self._enhanced_simulation(game_state, action)
                total_reward += reward
                simulations += 1

            if simulations > 0:
                action_scores[action] = total_reward / simulations
            else:
                action_scores[action] = 0

        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]
        return candidates[0]

    def _enhanced_simulation(self, state, action):
        """增强的模拟评估"""
        try:
            coords = action.get('coords')
            if not coords:
                return 0

            agent = state.agents[self.id]
            board = state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 使用完整的位置评分
            score = self._comprehensive_position_score(board, coords, my_color, enemy_color)

            # 添加随机性以模拟不确定性
            noise = random.uniform(0.9, 1.1)
            return (score / 1000.0) * noise
        except:
            return 0

    # ============ 辅助函数 ============

    def _assess_complexity(self, game_state, actions):
        """评估局面复杂度"""
        try:
            complexity = 0.5

            if len(actions) > 25:
                complexity += 0.2
            elif len(actions) < 8:
                complexity -= 0.1

            if self.turn_count <= 8:
                complexity -= 0.1
            elif self.turn_count > 25:
                complexity += 0.15

            if hasattr(game_state, 'board'):
                board = game_state.board.chips
                piece_count = sum(1 for row in board for cell in row if cell in ['r', 'b'])
                if piece_count > 35:
                    complexity += 0.1

            return max(0.1, min(0.9, complexity))
        except:
            return 0.5

    def _check_immediate_win(self, board, coords, my_color):
        """检查即时获胜"""
        r, c = coords
        for dx, dy in self.directions:
            count, _ = self._analyze_chain_enhanced(board, r, c, dx, dy, my_color)
            if count >= 5:
                return True
        return False

    def _check_critical_block(self, board, coords, enemy_color):
        """检查关键阻断"""
        r, c = coords
        for dx, dy in self.directions:
            enemy_count = self._analyze_enemy_threat(board, r, c, dx, dy, enemy_color)
            if enemy_count >= 4:
                return True
        return False

    def _analyze_enemy_threat(self, board, r, c, dx, dy, enemy_color):
        """分析敌方威胁"""
        max_threat = 0
        for direction in [1, -1]:
            count = 0
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if board[x][y] == enemy_color or (x, y) in CORNER_POSITIONS:
                        count += 1
                    else:
                        break
                else:
                    break
            max_threat = max(max_threat, count)
        return max_threat

    def _enhanced_hotb_bonus(self, board, my_color, enemy_color):
        """增强HOTB奖励"""
        my_control = sum(1 for r, c in HOTB_COORDS if board[r][c] == my_color)
        enemy_control = sum(1 for r, c in HOTB_COORDS if board[r][c] == enemy_color)

        bonus = my_control * 80 - enemy_control * 120
        if my_control >= 3:
            bonus += 400
        if my_control == 4:
            bonus += 1000
        return bonus

    def _intelligent_sort(self, actions, game_state):
        """智能动作排序"""
        if len(actions) <= 2:
            return actions

        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            def action_priority(action):
                coords = action.get('coords')
                if not coords:
                    return 0
                return self._comprehensive_position_score(board, coords, my_color, enemy_color)

            return sorted(actions, key=action_priority, reverse=True)
        except:
            return actions

    def _enhanced_card_selection(self, actions, game_state):
        """增强卡片选择"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']
        if not trade_actions:
            return random.choice(actions) if actions else None

        try:
            # 优先J牌
            for action in trade_actions:
                draft_card = action.get('draft_card', '').lower()
                if draft_card in ['jc', 'jd', 'js', 'jh']:
                    return action

            # 评估卡片价值
            best_action = None
            best_value = -1

            for action in trade_actions:
                draft_card = action.get('draft_card', '')
                if draft_card in COORDS:
                    value = self._evaluate_card_value(draft_card, game_state)
                    if value > best_value:
                        best_value = value
                        best_action = action

            return best_action if best_action else random.choice(trade_actions)
        except:
            return random.choice(trade_actions)

    def _evaluate_card_value(self, card, game_state):
        """评估卡片价值"""
        try:
            positions = COORDS.get(card, [])
            if not positions:
                return 0

            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            total_value = 0
            available_positions = 0

            for pos in positions:
                if len(pos) == 2:
                    r, c = pos
                    if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                        available_positions += 1
                        value = self._comprehensive_position_score(board, (r, c), my_color, enemy_color)
                        total_value += value

            return total_value + available_positions * 100
        except:
            return 0

    def _smart_cache_cleanup(self):
        """智能缓存清理"""
        # 保留最近使用的缓存项
        if len(self.eval_cache) > self.max_cache // 2:
            # 简单FIFO清理
            items_to_remove = list(self.eval_cache.keys())[:self.max_cache // 4]
            for key in items_to_remove:
                del self.eval_cache[key]
        self.cache_size = len(self.eval_cache)

    def _board_hash(self, board):
        """棋盘哈希"""
        return hash(tuple(tuple(row) for row in board))

    def _get_state_signature(self, state, sequence):
        """状态签名"""
        try:
            board_hash = self._board_hash(state.board.chips)
            move_hash = hash(tuple(str(move) for move in sequence))
            return (board_hash, move_hash)
        except:
            return (id(state), tuple(sequence))

    def _select_mcts_candidates(self, actions, game_state, astar_result, quick_result):
        """选择MCTS候选"""
        candidates = [astar_result, quick_result]
        if len(actions) > 2:
            sorted_actions = self._intelligent_sort(actions, game_state)
            candidates.extend(sorted_actions[:2])
        return list(set(candidates))  # 去重

    def _quick_position_check(self, board, coords, my_color):
        """快速位置检查"""
        r, c = coords
        if not (0 <= r < 10 and 0 <= c < 10) or board[r][c] != '0':
            return False
        return True

    def _simulate_action(self, state, action):
        """快速动作模拟"""
        return state  # 简化版，避免深拷贝

    def _evaluate_sequence(self, state, sequence):
        """序列评估"""
        if not sequence:
            return 0
        return self._enhanced_simulation(state, sequence[-1])

    def _enhanced_heuristic(self, state, action):
        """增强启发式"""
        try:
            coords = action.get('coords')
            if not coords:
                return 1000

            agent = state.agents[self.id]
            board = state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            score = self._comprehensive_position_score(board, coords, my_color, enemy_color)
            return max(1, 1000 - score // 20)
        except:
            return 1000

    def _is_card_selection(self, actions, game_state):
        """判断卡片选择"""
        try:
            return any(action.get('type') == 'trade' for action in actions)
        except:
            return False

    def _emergency_decision(self, actions, game_state):
        """紧急决策"""
        if not actions:
            return None

        for action in actions:
            coords = action.get('coords')
            if coords and coords in HOTB_COORDS:
                return action

        for action in actions:
            coords = action.get('coords')
            if coords and coords in self.opening_priority:
                return action

        return actions[0]