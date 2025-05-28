from collections import namedtuple
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
from Sequence.sequence_model import COORDS
import heapq
import time
import itertools
import random
import math
from copy import deepcopy

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

        # 核心优化：借鉴第三版思路但重新设计
        self._build_strategic_tables()

        # 轻量缓存
        self.position_cache = {}
        self.cache_hits = 0

    def _build_strategic_tables(self):
        """构建战略查找表 - 借鉴第三版预计算思路但重新实现"""
        # 1. 位置战略价值表 - 受第三版POSITION_WEIGHTS启发
        self.strategic_values = {}
        for r in range(10):
            for c in range(10):
                value = self._calculate_strategic_value(r, c)
                self.strategic_values[(r, c)] = value

        # 2. 连子奖励表 - 受第三版exp_weight启发但简化
        self.chain_rewards = {0: 0, 1: 5, 2: 25, 3: 200, 4: 2000, 5: 50000}

        # 3. HOTB控制策略表 - 受第三版heart_weight启发但重新设计
        self.hotb_strategy = self._build_hotb_strategy_table()

        # 4. 方向评估模板 - 受第三版滑动窗口启发
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 5. 开局模式库
        self.opening_patterns = self._build_opening_patterns()

    def _calculate_strategic_value(self, r, c):
        """计算位置战略价值 - 借鉴第三版思路但用不同算法"""
        if (r, c) in CORNER_POSITIONS:
            return 0  # 角落无价值

        value = 10  # 基础价值

        # HOTB核心区域 - 指数加权
        if (r, c) in HOTB_COORDS:
            value += 50

        # 中心影响力 - 使用高斯分布而非第三版的指数衰减
        center_r, center_c = 4.5, 4.5
        distance_sq = (r - center_r) ** 2 + (c - center_c) ** 2
        center_influence = 30 * math.exp(-distance_sq / 8.0)  # 高斯分布
        value += center_influence

        # 边缘惩罚
        edge_penalty = 0
        if r in [0, 9] or c in [0, 9]:
            edge_penalty = 15
        value -= edge_penalty

        return value

    def _build_hotb_strategy_table(self):
        """构建HOTB战略表 - 借鉴第三版思路但用自己的策略"""
        # 不直接复制第三版的硬编码表，而是用规则生成
        strategy = {}
        for my_count in range(5):
            for opp_count in range(5):
                if my_count + opp_count > 4:
                    continue

                # 自己的策略逻辑
                place_value = 0
                remove_value = 0

                if my_count >= 3:
                    place_value = 1000 * (2 ** my_count)  # 指数增长
                elif my_count >= 1:
                    place_value = 100 * my_count

                if opp_count >= 3:
                    remove_value = 800 * (2 ** opp_count)  # 防御指数增长
                elif opp_count >= 2:
                    remove_value = 200 * opp_count

                strategy[(my_count, opp_count)] = (place_value, remove_value)

        return strategy

    def _build_opening_patterns(self):
        """构建开局模式 - 原创设计"""
        patterns = {
            'center_control': HOTB_COORDS,
            'expansion_ring': [(3, 3), (3, 6), (6, 3), (6, 6)],
            'strategic_corners': [(2, 2), (2, 7), (7, 2), (7, 7)],
            'fallback_positions': [(1, 1), (1, 8), (8, 1), (8, 8)]
        }

        # 按优先级展开
        ordered_positions = []
        for pattern_positions in patterns.values():
            ordered_positions.extend(pattern_positions)

        return ordered_positions

    def SelectAction(self, actions, game_state):
        """主决策入口"""
        self.turn_count += 1
        start_time = time.time()

        # 动态时间管理
        if not self.startup_used:
            elapsed = time.time() - self.startup_start
            remaining = 15.0 - elapsed
            time_limit = min(remaining - 0.05, 0.95) if remaining > 0.15 else 0.95
            if remaining <= 0.15:
                self.startup_used = True
        else:
            time_limit = 0.95

        if not actions or time_limit < 0.02:
            return self._emergency_action(actions, game_state)

        if self._is_card_selection(actions, game_state):
            return self._intelligent_card_selection(actions, game_state)

        return self._hybrid_search_strategy(actions, game_state, start_time, time_limit)

    def _hybrid_search_strategy(self, actions, game_state, start_time, time_limit):
        """混合搜索策略 - A*+MCTS但融入第三版的快速评估思路"""

        # 阶段1：战略快评（15%时间）- 借鉴第三版直接评估但保持搜索框架
        strategic_result = self._strategic_fast_evaluation(actions, game_state)

        if time.time() - start_time > time_limit * 0.15:
            return strategic_result

        # 阶段2：增强A*搜索（70%时间）- 融入第三版的滑动窗口思路
        enhanced_astar_result = self._enhanced_astar_with_patterns(
            actions, game_state, start_time, time_limit * 0.85
        )

        if time.time() - start_time > time_limit * 0.85:
            return enhanced_astar_result or strategic_result

        # 阶段3：模式MCTS（15%时间）- 结合第三版的评估准确性
        pattern_mcts_result = self._pattern_based_mcts(
            [enhanced_astar_result, strategic_result],
            game_state, start_time, time_limit
        )

        return pattern_mcts_result or enhanced_astar_result or strategic_result

    def _strategic_fast_evaluation(self, actions, game_state):
        """战略快速评估 - 借鉴第三版思路但用于A*框架"""
        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'
        except:
            return actions[0] if actions else None

        # 开局模式匹配 - 受第三版启发但重新设计
        if self.turn_count <= 10:
            for action in actions[:5]:
                coords = action.get('coords')
                if coords and coords in self.opening_patterns:
                    if self._validate_opening_move(board, coords, my_color):
                        return action

        # 紧急情况处理
        for action in actions[:8]:
            coords = action.get('coords')
            if coords and self._is_playable_position(board, coords):
                # 获胜检查
                if self._rapid_win_check(board, coords, my_color):
                    return action
                # 防御检查
                if self._rapid_defense_check(board, coords, enemy_color):
                    return action

        # 战略评分选择
        return self._select_by_strategic_score(actions, board, my_color, enemy_color)

    def _select_by_strategic_score(self, actions, board, my_color, enemy_color):
        """基于战略评分选择 - 融合第三版高效评估"""
        best_action = actions[0]
        best_score = -999999

        for action in actions[:min(8, len(actions))]:
            coords = action.get('coords')
            if not coords or not self._is_playable_position(board, coords):
                continue

            # 使用缓存加速
            cache_key = (tuple(map(tuple, board)), coords, my_color)
            if cache_key in self.position_cache:
                score = self.position_cache[cache_key]
                self.cache_hits += 1
            else:
                score = self._comprehensive_position_evaluation(board, coords, my_color, enemy_color)
                if len(self.position_cache) < 200:  # 控制缓存大小
                    self.position_cache[cache_key] = score

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _comprehensive_position_evaluation(self, board, coords, my_color, enemy_color):
        """综合位置评估 - 借鉴第三版滑动窗口思路但重新实现"""
        r, c = coords

        # 基础战略价值
        base_score = self.strategic_values.get((r, c), 0)

        # 连子价值分析 - 受第三版启发但用不同实现
        chain_score = 0
        defense_score = 0

        for dx, dy in self.directions:
            # 我方连子分析 - 使用滑动评估思路
            my_chain_length = self._sliding_chain_analysis(board, r, c, dx, dy, my_color)
            chain_score += self.chain_rewards.get(my_chain_length, 0)

            # 防御价值分析
            enemy_threat = self._sliding_threat_analysis(board, r, c, dx, dy, enemy_color)
            if enemy_threat >= 4:
                defense_score += 5000
            elif enemy_threat >= 3:
                defense_score += 1500
            elif enemy_threat >= 2:
                defense_score += 300

        # HOTB战略加成 - 使用自建策略表
        hotb_bonus = self._calculate_hotb_bonus(board, r, c, my_color, enemy_color)

        return base_score + chain_score + defense_score + hotb_bonus

    def _sliding_chain_analysis(self, board, r, c, dx, dy, color):
        """滑动连子分析 - 借鉴第三版滑动窗口思路但简化"""
        max_length = 1  # 包含当前位置

        # 正向滑动
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if (0 <= x < 10 and 0 <= y < 10 and
                    (board[x][y] == color or (x, y) in CORNER_POSITIONS)):
                max_length += 1
            else:
                break

        # 反向滑动
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if (0 <= x < 10 and 0 <= y < 10 and
                    (board[x][y] == color or (x, y) in CORNER_POSITIONS)):
                max_length += 1
            else:
                break

        return min(max_length, 5)

    def _sliding_threat_analysis(self, board, r, c, dx, dy, enemy_color):
        """滑动威胁分析"""
        max_threat = 0

        for direction in [1, -1]:
            threat_count = 0
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if (0 <= x < 10 and 0 <= y < 10 and
                        (board[x][y] == enemy_color or (x, y) in CORNER_POSITIONS)):
                    threat_count += 1
                else:
                    break
            max_threat = max(max_threat, threat_count)

        return max_threat

    def _calculate_hotb_bonus(self, board, r, c, my_color, enemy_color):
        """计算HOTB奖励 - 使用自建策略表"""
        if (r, c) not in HOTB_COORDS:
            return 0

        # 统计当前HOTB控制情况
        my_count = sum(1 for hr, hc in HOTB_COORDS if board[hr][hc] == my_color)
        opp_count = sum(1 for hr, hc in HOTB_COORDS if board[hr][hc] == enemy_color)

        # 使用策略表
        place_value, _ = self.hotb_strategy.get((my_count, opp_count), (0, 0))
        return place_value

    def _enhanced_astar_with_patterns(self, actions, game_state, start_time, time_limit):
        """增强A*搜索 - 融入模式识别"""
        if not actions:
            return None

        # 动态搜索参数
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 0.3:
            max_candidates, max_expansions, depth_limit = 4, 12, 2
        else:
            max_candidates, max_expansions, depth_limit = 3, 8, 1

        heap = []
        visited = set()

        # 模式感知的动作排序
        pattern_sorted_actions = self._pattern_aware_sorting(actions, game_state)

        # 初始化搜索
        for action in pattern_sorted_actions[:max_candidates]:
            h_score = self._pattern_enhanced_heuristic(game_state, action)
            heapq.heappush(heap, (h_score, next(self.counter), 0, action, [action]))

        best_sequence = []
        best_reward = float('-inf')
        expansions = 0

        while (heap and expansions < max_expansions and
               time.time() - start_time < time_limit):

            f_score, _, depth, action, sequence = heapq.heappop(heap)

            # 模式感知的状态去重
            state_key = self._pattern_state_signature(game_state, sequence)
            if state_key in visited:
                continue
            visited.add(state_key)

            expansions += 1

            # 模式增强的序列评估
            reward = self._pattern_sequence_evaluation(game_state, sequence)
            if reward > best_reward:
                best_reward = reward
                best_sequence = sequence

            if depth >= depth_limit:
                continue

            # 状态模拟和扩展
            try:
                next_state = self._accurate_state_simulation(game_state, action)
                if next_state:
                    next_actions = self.rule.getLegalActions(next_state, self.id)
                    if next_actions:
                        sorted_next = self._pattern_aware_sorting(next_actions, next_state)

                        for next_action in sorted_next[:2]:
                            new_sequence = sequence + [next_action]
                            h_score = self._pattern_enhanced_heuristic(next_state, next_action)
                            heapq.heappush(heap, (
                                depth + 1 + h_score, next(self.counter),
                                depth + 1, next_action, new_sequence
                            ))
            except:
                continue

        return best_sequence[0] if best_sequence else actions[0]

    def _pattern_aware_sorting(self, actions, game_state):
        """模式感知的动作排序"""
        if len(actions) <= 3:
            return actions

        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            scored_actions = []
            for action in actions[:min(12, len(actions))]:
                coords = action.get('coords')
                if coords and self._is_playable_position(board, coords):
                    score = self._comprehensive_position_evaluation(board, coords, my_color, enemy_color)
                    scored_actions.append((score, action))

            scored_actions.sort(key=lambda x: x[0], reverse=True)
            return [action for _, action in scored_actions]
        except:
            return actions

    def _accurate_state_simulation(self, state, action):
        """精确状态模拟 - 修复关键bug"""
        try:
            new_state = deepcopy(state)

            if action.get('type') == 'place' and action.get('coords'):
                r, c = action['coords']
                agent = new_state.agents[self.id]
                if 0 <= r < 10 and 0 <= c < 10:
                    new_state.board.chips[r][c] = agent.colour
            elif action.get('type') == 'remove' and action.get('coords'):
                r, c = action['coords']
                if 0 <= r < 10 and 0 <= c < 10:
                    new_state.board.chips[r][c] = '0'

            return new_state
        except:
            return None

    def _pattern_based_mcts(self, candidates, game_state, start_time, time_limit):
        """基于模式的MCTS"""
        if not candidates:
            return None

        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.02:
            return candidates[0]

        action_scores = {}
        simulations_per_action = max(4, int(remaining_time * 20))

        for action in candidates[:3]:  # 限制候选数量
            if time.time() - start_time >= time_limit:
                break

            total_reward = 0
            simulations = 0

            for _ in range(simulations_per_action):
                if time.time() - start_time >= time_limit:
                    break

                reward = self._pattern_simulation(game_state, action)
                total_reward += reward
                simulations += 1

            if simulations > 0:
                action_scores[action] = total_reward / simulations

        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]
        return candidates[0]

    def _pattern_simulation(self, state, action):
        """模式模拟"""
        try:
            coords = action.get('coords')
            if not coords:
                return 0

            agent = state.agents[self.id]
            board = state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            score = self._comprehensive_position_evaluation(board, coords, my_color, enemy_color)
            return score / 1000.0 + random.uniform(-0.05, 0.05)
        except:
            return 0

    # ============ 辅助函数 ============

    def _rapid_win_check(self, board, coords, my_color):
        """快速获胜检查"""
        r, c = coords
        for dx, dy in self.directions:
            if self._sliding_chain_analysis(board, r, c, dx, dy, my_color) >= 5:
                return True
        return False

    def _rapid_defense_check(self, board, coords, enemy_color):
        """快速防御检查"""
        r, c = coords
        for dx, dy in self.directions:
            if self._sliding_threat_analysis(board, r, c, dx, dy, enemy_color) >= 3:
                return True
        return False

    def _validate_opening_move(self, board, coords, my_color):
        """验证开局走法"""
        return self._is_playable_position(board, coords)

    def _is_playable_position(self, board, coords):
        """检查位置可下"""
        r, c = coords
        return (0 <= r < 10 and 0 <= c < 10 and
                board[r][c] == '0' and (r, c) not in CORNER_POSITIONS)

    def _pattern_enhanced_heuristic(self, state, action):
        """模式增强启发式"""
        try:
            coords = action.get('coords')
            if not coords:
                return 1000

            agent = state.agents[self.id]
            board = state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            score = self._comprehensive_position_evaluation(board, coords, my_color, enemy_color)
            return max(1, 1000 - score // 10)
        except:
            return 1000

    def _pattern_state_signature(self, state, sequence):
        """模式状态签名"""
        try:
            board_sig = hash(tuple(tuple(row) for row in state.board.chips))
            seq_sig = hash(tuple(str(action.get('coords')) for action in sequence))
            return (board_sig, seq_sig)
        except:
            return (id(state), len(sequence))

    def _pattern_sequence_evaluation(self, state, sequence):
        """模式序列评估"""
        if not sequence:
            return 0
        return self._pattern_simulation(state, sequence[-1]) * 1000

    def _intelligent_card_selection(self, actions, game_state):
        """智能卡片选择"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']
        if not trade_actions:
            return random.choice(actions) if actions else None

        # 优先J牌
        for action in trade_actions:
            draft_card = action.get('draft_card', '').lower()
            if draft_card in ['jc', 'jd', 'js', 'jh']:
                return action

        # 评估普通卡片价值
        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            best_action = None
            best_value = -1

            for action in trade_actions:
                draft_card = action.get('draft_card', '')
                if draft_card in COORDS:
                    total_value = 0
                    positions = COORDS[draft_card]
                    for pos in positions:
                        if len(pos) == 2:
                            r, c = pos
                            if self._is_playable_position(board, (r, c)):
                                value = self._comprehensive_position_evaluation(
                                    board, (r, c), my_color, enemy_color)
                                total_value += value

                    if total_value > best_value:
                        best_value = total_value
                        best_action = action

            return best_action if best_action else random.choice(trade_actions)
        except:
            return random.choice(trade_actions)

    def _is_card_selection(self, actions, game_state):
        """判断是否为卡片选择"""
        return any(action.get('type') == 'trade' for action in actions)

    def _emergency_action(self, actions, game_state):
        """紧急动作"""
        if not actions:
            return None

        # HOTB优先
        for action in actions[:3]:
            coords = action.get('coords')
            if coords and coords in HOTB_COORDS:
                return action

        # 开局模式优先
        for action in actions[:5]:
            coords = action.get('coords')
            if coords and coords in self.opening_patterns:
                return action

        return actions[0]