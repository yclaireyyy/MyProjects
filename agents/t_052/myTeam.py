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

        # 时间管理 - 更保守
        self.startup_start = time.time()
        self.startup_used = False
        self.turn_count = 0

        # 预计算核心数据
        self._precompute_data()

        # 优化的缓存系统
        self.eval_cache = {}
        self.pattern_cache = {}
        self.jack_cache = {}  # 新增J牌专用缓存
        self.cache_size = 0
        self.max_cache = 150  # 进一步减小缓存

        # J牌快速查找表 - O(1)访问
        self._build_jack_lookup_tables()

    def _build_jack_lookup_tables(self):
        """构建J牌快速查找表 - 一次性O(1)预计算"""
        # 预计算每个位置的威胁影响范围
        self.threat_zones = {}  # position -> 影响的所有方向线
        self.position_lines = {}  # position -> 所在的所有线

        for r in range(10):
            for c in range(10):
                zones = []
                lines = []

                for dx, dy in self.directions:
                    # 该位置参与的所有连线
                    line_positions = set()
                    line_positions.add((r, c))

                    # 正向扩展
                    for i in range(1, 5):
                        x, y = r + dx * i, c + dy * i
                        if 0 <= x < 10 and 0 <= y < 10:
                            line_positions.add((x, y))
                        else:
                            break

                    # 负向扩展
                    for i in range(1, 5):
                        x, y = r - dx * i, c - dy * i
                        if 0 <= x < 10 and 0 <= y < 10:
                            line_positions.add((x, y))
                        else:
                            break

                    if len(line_positions) >= 5:  # 有效连线
                        lines.append((dx, dy, frozenset(line_positions)))
                        zones.extend(list(line_positions))

                self.threat_zones[(r, c)] = list(set(zones))
                self.position_lines[(r, c)] = lines

        # 预计算高价值位置列表 - 避免遍历整个棋盘
        self.high_value_positions = []
        self.hotb_positions = set(HOTB_COORDS)
        self.strategic_positions = set()

        for r in range(10):
            for c in range(10):
                weight = self.position_weights.get((r, c), 0)
                if weight > 3 or (r, c) in HOTB_COORDS:
                    self.high_value_positions.append((r, c))
                if weight > 2:
                    self.strategic_positions.add((r, c))

    def _precompute_data(self):
        """预计算关键数据 - 进一步优化"""
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 更快的位置权重计算
        self.position_weights = {}
        center = 4.5

        for r in range(10):
            for c in range(10):
                weight = 1
                # HOTB权重
                if (r, c) in HOTB_COORDS:
                    weight += 5
                # 简化的中心权重
                center_dist = max(abs(r - center), abs(c - center))
                weight += max(0, 3 - center_dist * 0.4)
                # 简化的角落权重
                min_corner_dist = min(
                    max(abs(r - cr), abs(c - cc))
                    for cr, cc in CORNER_POSITIONS
                )
                if min_corner_dist <= 2:
                    weight += max(0, 2 - min_corner_dist * 0.5)

                self.position_weights[(r, c)] = weight

        # 威胁等级映射 - 保持不变
        self.threat_levels = {
            'win': 40000, 'critical_win': 20000, 'major_threat': 10000,
            'double_threat': 5000, 'active_threat': 2500, 'blocked_threat': 800,
            'potential_threat': 300, 'development': 150, 'weak_connection': 40,
            'none': 0
        }

        # 预计算方向查找表 - 保持不变
        self.direction_cache = {}
        for r in range(10):
            for c in range(10):
                for dx, dy in self.directions:
                    self.direction_cache[(r, c, dx, dy)] = self._precompute_direction_data(r, c, dx, dy)

    def _precompute_direction_data(self, r, c, dx, dy):
        """预计算方向数据"""
        positions = []
        # 正向
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                positions.append((x, y, 1))
            else:
                break
        # 负向
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                positions.append((x, y, -1))
            else:
                break
        return positions

    def SelectAction(self, actions, game_state):
        """主决策入口 - 添加更严格的时间控制"""
        self.turn_count += 1
        start_time = time.time()

        # 超保守的时间管理
        if not self.startup_used:
            elapsed = time.time() - self.startup_start
            remaining = 15.0 - elapsed
            time_limit = min(remaining - 0.3, 0.75) if remaining > 0.4 else 0.75
            if remaining <= 0.4:
                self.startup_used = True
        else:
            time_limit = 0.75  # 更保守

        if not actions or time_limit < 0.05:
            return self._emergency_decision(actions, game_state)

        # 判断动作类型
        if self._is_card_selection(actions, game_state):
            return self._ultra_fast_card_selection(actions, game_state, start_time, time_limit)

        return self._optimized_search_strategy(actions, game_state, start_time, time_limit)

    def _ultra_fast_card_selection(self, actions, game_state, start_time, time_limit):
        """超快速卡片选择 - 专门优化J牌评估"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']
        if not trade_actions:
            return random.choice(actions) if actions else None

        # 时间分配：给J牌评估留更少时间
        j_time_limit = time_limit * 0.4  # 只用40%时间评估J牌

        try:
            best_action = trade_actions[0]
            best_value = -999999

            for action in trade_actions:
                if time.time() - start_time > j_time_limit:
                    break

                draft_card = action.get('draft_card', '').lower()

                if draft_card in ['js', 'jh']:  # 单眼J
                    value = self._fast_one_eyed_jack_eval(game_state, start_time, j_time_limit)
                elif draft_card in ['jc', 'jd']:  # 双眼J
                    value = self._fast_two_eyed_jack_eval(game_state, start_time, j_time_limit)
                elif draft_card in COORDS:
                    value = self._quick_card_value(draft_card, game_state)
                else:
                    value = 0

                if value > best_value:
                    best_value = value
                    best_action = action

            return best_action

        except:
            return random.choice(trade_actions)

    def _fast_one_eyed_jack_eval(self, game_state, start_time, time_limit):
        """快速单眼J评估 - 避免全盘扫描"""
        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 创建缓存键
            board_hash = self._ultra_fast_board_hash(board, enemy_color)
            cache_key = ('one_eyed', board_hash, my_color)

            if cache_key in self.jack_cache:
                return self.jack_cache[cache_key]

            max_value = 0

            # 优先检查高价值位置的敌方棋子
            priority_targets = []

            # 1. HOTB区域的敌方棋子 - O(1)
            for pos in self.hotb_positions:
                if board[pos[0]][pos[1]] == enemy_color:
                    priority_targets.append(pos)

            # 2. 战略位置的敌方棋子 - O(k), k远小于100
            for pos in self.strategic_positions:
                if time.time() - start_time > time_limit * 0.8:
                    break
                if board[pos[0]][pos[1]] == enemy_color and pos not in priority_targets:
                    priority_targets.append(pos)

            # 快速评估优先目标
            for target in priority_targets[:8]:  # 最多评估8个目标
                if time.time() - start_time > time_limit * 0.9:
                    break

                removal_value = self._ultra_fast_removal_eval(board, target, my_color, enemy_color)
                max_value = max(max_value, removal_value)

            # 如果时间充足且没找到好目标，快速扫描其他位置
            if max_value < 2000 and time.time() - start_time < time_limit * 0.7:
                for r in range(1, 9, 2):  # 跳跃式扫描
                    for c in range(1, 9, 2):
                        if time.time() - start_time > time_limit * 0.9:
                            break
                        if board[r][c] == enemy_color:
                            removal_value = self._ultra_fast_removal_eval(board, (r, c), my_color, enemy_color)
                            max_value = max(max_value, removal_value)

            # 缓存结果
            if len(self.jack_cache) < 50:
                self.jack_cache[cache_key] = max_value

            return max_value

        except:
            return 3000

    def _fast_two_eyed_jack_eval(self, game_state, start_time, time_limit):
        """快速双眼J评估 - 只评估高价值位置"""
        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 创建缓存键
            board_hash = self._ultra_fast_board_hash(board, '0')
            cache_key = ('two_eyed', board_hash, my_color)

            if cache_key in self.jack_cache:
                return self.jack_cache[cache_key]

            max_value = 0
            available_count = 0

            # 优先评估高价值位置
            for pos in self.high_value_positions:
                if time.time() - start_time > time_limit * 0.8:
                    break

                r, c = pos
                if board[r][c] == '0':
                    available_count += 1

                    # 快速评估此位置价值
                    value = self.position_weights.get((r, c), 1) * 50

                    # 快速威胁检查 - 使用预计算的线
                    urgency = self._ultra_fast_urgency_check(board, pos, my_color, enemy_color)
                    value += urgency

                    max_value = max(max_value, value)

            # 灵活性奖励 - 简化计算
            flexibility_bonus = min(available_count * 40, 800)
            total_value = max_value + flexibility_bonus

            # 缓存结果
            if len(self.jack_cache) < 50:
                self.jack_cache[cache_key] = total_value

            return total_value

        except:
            return 4000

    def _ultra_fast_removal_eval(self, board, coords, my_color, enemy_color):
        """超快速移除评估 - 使用预计算查找表"""
        r, c = coords
        total_impact = self.position_weights.get(coords, 0) * 20

        # HOTB奖励 - O(1)
        if coords in self.hotb_positions:
            total_impact += 1200

        # 使用预计算的威胁线进行快速评估
        if coords in self.position_lines:
            for dx, dy, line_positions in self.position_lines[coords]:
                # 快速计算该线上的威胁
                enemy_count = sum(1 for pos in line_positions if board[pos[0]][pos[1]] == enemy_color)
                my_count = sum(1 for pos in line_positions if board[pos[0]][pos[1]] == my_color)

                # 移除该敌方棋子的影响
                if enemy_count >= 4:
                    total_impact += 6000
                elif enemy_count >= 3:
                    total_impact += 2000
                elif enemy_count >= 2:
                    total_impact += 500

                # 移除后对己方的帮助
                if my_count >= 3:
                    total_impact += 3000
                elif my_count >= 2:
                    total_impact += 800

        return total_impact

    def _ultra_fast_urgency_check(self, board, coords, my_color, enemy_color):
        """超快速紧急性检查"""
        urgency = 0
        r, c = coords

        # 使用预计算的线进行快速检查
        if coords in self.position_lines:
            for dx, dy, line_positions in self.position_lines[coords][:2]:  # 只检查前2条线
                my_count = sum(1 for pos in line_positions if board[pos[0]][pos[1]] == my_color) + 1
                enemy_count = sum(1 for pos in line_positions if board[pos[0]][pos[1]] == enemy_color)

                if my_count >= 5:
                    urgency += 12000  # 立即获胜
                elif my_count >= 4:
                    urgency += 4000  # 强威胁

                if enemy_count >= 4:
                    urgency += 8000  # 阻断关键威胁
                elif enemy_count >= 3:
                    urgency += 2000  # 阻断威胁

        return urgency

    def _ultra_fast_board_hash(self, board, target_color):
        """超快速棋盘哈希 - 只哈希相关颜色"""
        # 只对目标颜色的棋子位置进行哈希
        relevant_positions = []
        for r in range(0, 10, 2):  # 跳跃式采样
            for c in range(0, 10, 2):
                if board[r][c] == target_color:
                    relevant_positions.append(r * 10 + c)
        return hash(tuple(sorted(relevant_positions)))

    def _quick_card_value(self, card, game_state):
        """快速卡片价值评估 - 修复缺失函数"""
        try:
            positions = COORDS.get(card, [])
            if not positions:
                return 0

            board = game_state.board.chips
            total_value = 0
            available_count = 0

            for pos in positions:
                if len(pos) == 2:
                    r, c = pos
                    if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                        available_count += 1
                        # 使用简化的位置权重
                        total_value += self.position_weights.get((r, c), 1)

            return total_value + available_count * 50
        except:
            return 0

    # ========== 保持原有的其他优化函数 ==========

    def _optimized_search_strategy(self, actions, game_state, start_time, time_limit):
        """优化的搜索策略"""
        # 快速评估局面复杂度
        complexity = self._quick_assess_complexity(game_state, actions)

        # 阶段1：增强快速评估 (30%时间)
        quick_result = self._enhanced_quick_evaluation(actions, game_state)

        if time.time() - start_time > time_limit * 0.3:
            return quick_result

        # 阶段2：轻量级搜索 (60%时间)
        try:
            search_result = self._lightweight_search(actions, game_state, start_time, time_limit * 0.6, complexity)
            if not search_result:
                search_result = quick_result
        except:
            search_result = quick_result

        if time.time() - start_time > time_limit * 0.6:
            return search_result

        # 阶段3：精简MCTS (85%时间)
        try:
            candidates = [search_result, quick_result]
            if len(actions) > 3:
                sorted_actions = self._intelligent_sort(actions, game_state)
                candidates.extend(sorted_actions[:2])
            candidates = list(set(candidates))

            mcts_result = self._lightweight_mcts(candidates, game_state, start_time, time_limit * 0.85)
            return mcts_result if mcts_result else search_result
        except:
            return search_result

    def _lightweight_search(self, actions, game_state, start_time, time_limit, complexity):
        """轻量级搜索"""
        if not actions:
            return None

        # 根据复杂度调整参数 - 更保守
        if complexity > 0.7:
            max_candidates, max_evaluations = 2, 6
        elif complexity > 0.4:
            max_candidates, max_evaluations = 3, 8
        else:
            max_candidates, max_evaluations = 4, 10

        sorted_actions = self._intelligent_sort(actions, game_state)
        candidates = sorted_actions[:max_candidates]

        best_action = candidates[0]
        best_score = float('-inf')
        evaluations = 0

        for action in candidates:
            if time.time() - start_time >= time_limit or evaluations >= max_evaluations:
                break

            score = self._cached_action_evaluation(action, game_state)

            if evaluations < max_evaluations // 2:
                lookahead_bonus = self._quick_lookahead(action, game_state)
                score += lookahead_bonus * 0.3

            if score > best_score:
                best_score = score
                best_action = action

            evaluations += 1

        return best_action

    def _quick_lookahead(self, action, game_state):
        """快速前瞻评估"""
        try:
            coords = action.get('coords')
            if not coords:
                return 0

            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            bonus = 0
            for dx, dy in self.directions:
                my_threat = self._quick_threat_after_move(board, coords, dx, dy, my_color)
                enemy_threat = self._quick_threat_after_move(board, coords, dx, dy, enemy_color)

                if my_threat >= 4:
                    bonus += 5000
                elif my_threat >= 3:
                    bonus += 1000

                if enemy_threat >= 4:
                    bonus += 3000
                elif enemy_threat >= 3:
                    bonus += 500

            return bonus
        except:
            return 0

    def _quick_threat_after_move(self, board, coords, dx, dy, color):
        """快速计算移动后的威胁"""
        r, c = coords
        count = 1

        cache_key = (r, c, dx, dy)
        if cache_key in self.direction_cache:
            positions = self.direction_cache[cache_key]
            for x, y, direction in positions:
                if board[x][y] == color or (x, y) in CORNER_POSITIONS:
                    count += 1
                elif board[x][y] != '0':
                    break

        return min(count, 5)

    def _lightweight_mcts(self, candidates, game_state, start_time, time_limit):
        """轻量级MCTS"""
        if not candidates:
            return None

        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.02:
            return candidates[0]

        action_scores = {}
        iterations_per_action = max(2, int(remaining_time * 12))  # 进一步减少

        for action in candidates:
            if time.time() - start_time >= time_limit:
                break

            total_reward = 0
            simulations = 0

            for _ in range(iterations_per_action):
                if time.time() - start_time >= time_limit:
                    break

                reward = self._cached_action_evaluation(action, game_state)
                reward *= random.uniform(0.95, 1.05)
                total_reward += reward
                simulations += 1

            if simulations > 0:
                action_scores[action] = total_reward / simulations
            else:
                action_scores[action] = 0

        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]
        return candidates[0]

    def _cached_action_evaluation(self, action, game_state):
        """缓存的动作评估"""
        coords = action.get('coords')
        if not coords:
            return 0

        try:
            board_hash = self._quick_board_hash(game_state.board.chips)
            cache_key = (board_hash, coords, self.id)

            if cache_key in self.eval_cache:
                return self.eval_cache[cache_key]

            agent = game_state.agents[self.id]
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            score = self._optimized_position_score(game_state.board.chips, coords, my_color, enemy_color)

            if self.cache_size >= self.max_cache:
                self._efficient_cache_cleanup()

            self.eval_cache[cache_key] = score
            self.cache_size += 1

            return score
        except:
            return self._optimized_position_score(game_state.board.chips, coords,
                                                  game_state.agents[self.id].colour,
                                                  'r' if game_state.agents[self.id].colour == 'b' else 'b')

    def _optimized_position_score(self, board, coords, my_color, enemy_color):
        """优化的位置评分"""
        r, c = coords
        if not (0 <= r < 10 and 0 <= c < 10):
            return -999999

        score = self.position_weights.get((r, c), 1)

        for dx, dy in self.directions:
            cache_key = (r, c, dx, dy)
            if cache_key in self.direction_cache:
                positions = self.direction_cache[cache_key]

                my_count, my_openings = self._fast_chain_analysis(board, r, c, positions, my_color)
                corner_support = self._quick_corner_check(board, positions, my_color)

                threat_type, threat_score = self._classify_threat(my_count, my_openings, corner_support)
                score += threat_score

                enemy_threat = self._fast_enemy_threat(board, positions, enemy_color)
                if enemy_threat >= 4:
                    score += 10000
                elif enemy_threat >= 3:
                    score += 2500
                elif enemy_threat >= 2:
                    score += 600

        if (r, c) in HOTB_COORDS:
            score += self._quick_hotb_bonus(board, my_color, enemy_color)

        return score

    def _fast_chain_analysis(self, board, r, c, positions, color):
        """快速连子分析"""
        count = 1
        openings = 0

        pos_open = False
        neg_open = False

        for x, y, direction in positions:
            if board[x][y] == color or (x, y) in CORNER_POSITIONS:
                count += 1
            elif board[x][y] == '0':
                if direction > 0:
                    pos_open = True
                else:
                    neg_open = True
                break
            else:
                break

        openings = (1 if pos_open else 0) + (1 if neg_open else 0)
        return min(count, 5), openings

    def _quick_corner_check(self, board, positions, color):
        """快速角落检查"""
        for x, y, direction in positions:
            if (x, y) in CORNER_POSITIONS:
                return True
            if board[x][y] != color and board[x][y] != '0' and (x, y) not in CORNER_POSITIONS:
                break
        return False

    def _fast_enemy_threat(self, board, positions, enemy_color):
        """快速敌方威胁评估"""
        max_threat = 0
        current_count = 0

        for x, y, direction in positions:
            if board[x][y] == enemy_color or (x, y) in CORNER_POSITIONS:
                current_count += 1
                max_threat = max(max_threat, current_count)
            else:
                current_count = 0

        return min(max_threat + 1, 5)

    def _quick_hotb_bonus(self, board, my_color, enemy_color):
        """快速HOTB奖励计算"""
        my_control = sum(1 for r, c in HOTB_COORDS if board[r][c] == my_color)
        enemy_control = sum(1 for r, c in HOTB_COORDS if board[r][c] == enemy_color)

        bonus = my_control * 60 - enemy_control * 100
        if my_control >= 3:
            bonus += 300
        if my_control == 4:
            bonus += 800
        return bonus

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

        if self.turn_count <= 8:
            for action in actions:
                coords = action.get('coords')
                if coords and coords in HOTB_COORDS and board[coords[0]][coords[1]] == '0':
                    return action

        for action in actions:
            coords = action.get('coords')
            if coords:
                if self._quick_win_check(board, coords, my_color):
                    return action
                if self._quick_block_check(board, coords, enemy_color):
                    return action

        best_action = actions[0]
        best_score = -999999

        eval_limit = min(len(actions), 5)
        for action in actions[:eval_limit]:
            coords = action.get('coords')
            if not coords:
                continue

            score = self._cached_action_evaluation(action, game_state)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _quick_win_check(self, board, coords, my_color):
        """快速获胜检查"""
        r, c = coords
        for dx, dy in self.directions:
            if self._quick_threat_after_move(board, coords, dx, dy, my_color) >= 5:
                return True
        return False

    def _quick_block_check(self, board, coords, enemy_color):
        """快速阻断检查"""
        r, c = coords
        for dx, dy in self.directions:
            if self._quick_threat_after_move(board, coords, dx, dy, enemy_color) >= 4:
                return True
        return False

    # ============ 保持原有接口的其他函数 ============

    def _classify_threat(self, count, openings, corner_support):
        """威胁分类"""
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

    def _intelligent_sort(self, actions, game_state):
        """智能动作排序"""
        if len(actions) <= 3:
            return actions

        try:
            scored_actions = []
            for action in actions:
                score = self._cached_action_evaluation(action, game_state)
                scored_actions.append((action, score))

            scored_actions.sort(key=lambda x: x[1], reverse=True)
            return [action for action, score in scored_actions]
        except:
            return actions

    def _quick_assess_complexity(self, game_state, actions):
        """快速评估局面复杂度"""
        complexity = 0.5

        if len(actions) > 20:
            complexity += 0.15
        elif len(actions) < 10:
            complexity -= 0.1

        if self.turn_count <= 6:
            complexity -= 0.1
        elif self.turn_count > 20:
            complexity += 0.1

        return max(0.2, min(0.8, complexity))

    def _efficient_cache_cleanup(self):
        """高效缓存清理"""
        keys_to_remove = list(self.eval_cache.keys())[:self.max_cache // 3]
        for key in keys_to_remove:
            self.eval_cache.pop(key, None)
        self.cache_size = len(self.eval_cache)

    def _quick_board_hash(self, board):
        """快速棋盘哈希"""
        return hash(tuple(tuple(row) for row in board[::2]))

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

        return actions[0]