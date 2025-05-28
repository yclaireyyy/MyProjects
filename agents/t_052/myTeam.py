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

        # 智能时间管理
        self.startup_start = time.time()
        self.startup_used = False
        self.turn_count = 0
        self.total_thinking_time = 0
        self.avg_turn_time = 0.5  # 动态平均时间

        # 预计算核心数据
        self._precompute_data()

        # 增强缓存系统 - 分层缓存
        self.eval_cache = {}
        self.pattern_cache = {}
        self.jack_cache = {}
        self.urgency_cache = {}  # 紧急情况缓存
        self.cache_size = 0
        self.max_cache = 150

        # 性能监控
        self.decision_quality_history = []
        self.time_pressure_level = 0  # 0-低压，1-中压，2-高压

        # 快速查找表
        self._build_lookup_tables()

    def _precompute_data(self):
        """预计算关键数据"""
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 改进的位置权重 - 基于距离的数学模型
        self.position_weights = {}
        center = 4.5

        for r in range(10):
            for c in range(10):
                # 中心距离权重
                center_dist = ((r - center) ** 2 + (c - center) ** 2) ** 0.5
                weight = max(1, 8 - center_dist * 0.8)

                # HOTB特殊权重
                if (r, c) in HOTB_COORDS:
                    weight += 6

                # 角落战略权重
                corner_dists = [max(abs(r - cr), abs(c - cc)) for cr, cc in CORNER_POSITIONS]
                min_corner_dist = min(corner_dists)
                if min_corner_dist <= 2:
                    weight += max(0, 2 - min_corner_dist * 0.5)

                self.position_weights[(r, c)] = weight

        # 威胁等级映射
        self.threat_levels = {
            'win': 50000, 'critical_win': 25000, 'major_threat': 12000,
            'double_threat': 6000, 'active_threat': 3000, 'blocked_threat': 1000,
            'potential_threat': 400, 'development': 200, 'weak_connection': 50,
            'none': 0
        }

        # 预计算方向查找表
        self.direction_cache = {}
        for r in range(10):
            for c in range(10):
                for dx, dy in self.directions:
                    self.direction_cache[(r, c, dx, dy)] = self._precompute_direction_data(r, c, dx, dy)

    def _build_lookup_tables(self):
        """构建快速查找表"""
        # 所有可能的5连线段
        self.all_lines = []

        # 水平线
        for r in range(10):
            for c in range(6):  # 0-5列开始，长度为5
                self.all_lines.append([(r, c + i) for i in range(5)])

        # 垂直线
        for r in range(6):  # 0-5行开始
            for c in range(10):
                self.all_lines.append([(r + i, c) for i in range(5)])

        # 主对角线
        for r in range(6):
            for c in range(6):
                self.all_lines.append([(r + i, c + i) for i in range(5)])

        # 副对角线
        for r in range(6):
            for c in range(4, 10):
                self.all_lines.append([(r + i, c - i) for i in range(5)])

        # 位置到线段的映射
        self.position_to_lines = {}
        for r in range(10):
            for c in range(10):
                self.position_to_lines[(r, c)] = []

        for line_idx, line in enumerate(self.all_lines):
            for pos in line:
                self.position_to_lines[pos].append(line_idx)

        # 高价值位置预筛选
        self.priority_positions = []
        self.strategic_positions = []
        for r in range(10):
            for c in range(10):
                weight = self.position_weights.get((r, c), 0)
                if weight > 6:
                    self.priority_positions.append((r, c))
                if weight > 4:
                    self.strategic_positions.append((r, c))

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
        """主决策入口 - 智能时间分配"""
        self.turn_count += 1
        start_time = time.time()

        # 动态时间管理
        time_limit, self.time_pressure_level = self._intelligent_time_allocation()

        if not actions or time_limit < 0.03:
            return self._emergency_decision(actions, game_state)

        try:
            # 判断动作类型
            if self._is_card_selection(actions, game_state):
                result = self._balanced_card_selection(actions, game_state, start_time, time_limit)
            else:
                result = self._adaptive_search_strategy(actions, game_state, start_time, time_limit)

            # 记录性能数据
            elapsed = time.time() - start_time
            self.total_thinking_time += elapsed
            self.avg_turn_time = self.total_thinking_time / self.turn_count

            return result

        except Exception:
            return self._emergency_decision(actions, game_state)

    def _intelligent_time_allocation(self):
        """智能时间分配 - 根据游戏状态和历史表现动态调整"""
        if not self.startup_used:
            elapsed = time.time() - self.startup_start
            remaining = 15.0 - elapsed

            if remaining <= 0.5:
                self.startup_used = True
                return 0.7, 2  # 高压模式

            # 启动阶段的渐进式时间使用
            if self.turn_count <= 3:
                time_limit = min(remaining - 0.4, 1.2)  # 早期可以用更多时间
                pressure = 0
            else:
                time_limit = min(remaining - 0.3, 0.9)
                pressure = 1 if remaining < 3.0 else 0

            return time_limit, pressure
        else:
            # 根据历史表现调整
            base_time = 0.8

            # 如果平均用时过高，降低时间限制
            if self.avg_turn_time > 0.7:
                base_time = 0.6
                pressure = 2
            elif self.avg_turn_time > 0.5:
                base_time = 0.7
                pressure = 1
            else:
                pressure = 0

            # 游戏后期稍微保守
            if self.turn_count > 30:
                base_time *= 0.9
                pressure = max(pressure, 1)

            return base_time, pressure

    def _balanced_card_selection(self, actions, game_state, start_time, time_limit):
        """平衡的卡片选择 - 保持精度但提高效率"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']
        if not trade_actions:
            return random.choice(actions) if actions else None

        try:
            board = game_state.board.chips
            agent = game_state.agents[self.id]
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 分阶段评估 - 根据时间压力调整
            if self.time_pressure_level >= 2:
                return self._fast_card_evaluation(trade_actions, board, my_color, enemy_color)

            # 正常评估流程
            best_action = trade_actions[0]
            best_value = -999999
            evaluation_time = time_limit * (0.5 if self.time_pressure_level == 0 else 0.4)

            for action in trade_actions:
                if time.time() - start_time > evaluation_time:
                    break

                draft_card = action.get('draft_card', '').lower()

                # 智能缓存检查
                cache_key = (draft_card, self._quick_board_hash(board), my_color)
                if cache_key in self.jack_cache:
                    value = self.jack_cache[cache_key]
                else:
                    if draft_card in ['js', 'jh']:  # 单眼J
                        value = self._enhanced_one_eyed_jack(board, enemy_color, start_time, evaluation_time)
                    elif draft_card in ['jc', 'jd']:  # 双眼J
                        value = self._enhanced_two_eyed_jack(board, my_color, enemy_color, start_time, evaluation_time)
                    elif draft_card in COORDS:
                        value = self._eval_normal_card(draft_card, board, my_color, enemy_color)
                    else:
                        value = 0

                    # 缓存结果
                    if len(self.jack_cache) < 80:
                        self.jack_cache[cache_key] = value

                if value > best_value:
                    best_value = value
                    best_action = action

            return best_action

        except:
            return random.choice(trade_actions)

    def _enhanced_one_eyed_jack(self, board, enemy_color, start_time, time_limit):
        """增强的单眼J评估 - 保持精度但加速"""
        max_value = 0

        # 阶段1：优先检查HOTB和关键位置
        priority_targets = []
        for r, c in HOTB_COORDS:
            if board[r][c] == enemy_color:
                priority_targets.append((r, c))

        # 阶段2：检查战略位置
        if time.time() - start_time < time_limit * 0.3:
            for r, c in self.strategic_positions:
                if board[r][c] == enemy_color and (r, c) not in priority_targets:
                    priority_targets.append((r, c))
                    if len(priority_targets) >= 12:  # 限制检查数量
                        break

        # 阶段3：详细评估
        evaluation_budget = min(len(priority_targets), 8 if self.time_pressure_level >= 1 else 10)
        for i, target in enumerate(priority_targets[:evaluation_budget]):
            if time.time() - start_time > time_limit * 0.8:
                break

            # 使用缓存的移除价值计算
            cache_key = ('removal', target, enemy_color)
            if cache_key in self.pattern_cache:
                value = self.pattern_cache[cache_key]
            else:
                value = self._calc_removal_value(board, target, enemy_color)
                if len(self.pattern_cache) < 100:
                    self.pattern_cache[cache_key] = value

            max_value = max(max_value, value)

        return max_value if max_value > 0 else 3500

    def _enhanced_two_eyed_jack(self, board, my_color, enemy_color, start_time, time_limit):
        """增强的双眼J评估 - 分层评估"""
        max_value = 0
        available_count = 0

        # 根据时间压力调整评估范围
        if self.time_pressure_level >= 2:
            target_positions = self.priority_positions[:15]
        elif self.time_pressure_level >= 1:
            target_positions = self.priority_positions[:20]
        else:
            target_positions = self.priority_positions

        for r, c in target_positions:
            if time.time() - start_time > time_limit * 0.7:
                break

            if board[r][c] == '0':
                available_count += 1

                # 使用缓存的威胁值计算
                cache_key = ('threat', (r, c), my_color, enemy_color)
                if cache_key in self.urgency_cache:
                    threat_value = self.urgency_cache[cache_key]
                else:
                    threat_value = self._calc_position_threat_value(board, (r, c), my_color, enemy_color)
                    if len(self.urgency_cache) < 80:
                        self.urgency_cache[cache_key] = threat_value

                # 基础位置价值
                pos_value = self.position_weights.get((r, c), 0) * 100
                total_value = pos_value + threat_value
                max_value = max(max_value, total_value)

        # 灵活性奖励
        flexibility = min(available_count * 50, 1000)
        return max_value + flexibility

    def _fast_card_evaluation(self, trade_actions, board, my_color, enemy_color):
        """快速卡片评估 - 高压模式"""
        j_actions = []
        normal_actions = []

        for action in trade_actions:
            draft_card = action.get('draft_card', '').lower()
            if draft_card in ['js', 'jh', 'jc', 'jd']:
                j_actions.append((action, draft_card))
            else:
                normal_actions.append((action, draft_card))

        # 优先评估J牌
        if j_actions:
            best_j = None
            best_j_value = 0

            for action, card in j_actions:
                if card in ['js', 'jh']:
                    # 简化的单眼J评估
                    value = 4000
                    hotb_enemy = sum(1 for r, c in HOTB_COORDS if board[r][c] == enemy_color)
                    if hotb_enemy > 0:
                        value += hotb_enemy * 2000
                elif card in ['jc', 'jd']:
                    # 简化的双眼J评估
                    value = 3500
                    hotb_empty = sum(1 for r, c in HOTB_COORDS if board[r][c] == '0')
                    value += hotb_empty * 500

                if value > best_j_value:
                    best_j_value = value
                    best_j = action

            if best_j:
                return best_j

        # 评估普通卡片
        if normal_actions:
            for action, card in normal_actions[:3]:  # 只评估前3个
                if card in COORDS:
                    positions = COORDS.get(card, [])
                    for r, c in positions:
                        if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                            if (r, c) in HOTB_COORDS:
                                return action  # 立即返回HOTB位置

        return trade_actions[0]

    def _adaptive_search_strategy(self, actions, game_state, start_time, time_limit):
        """自适应搜索策略 - 根据压力等级调整搜索深度"""

        # 阶段1：快速评估 (保证有结果)
        quick_result = self._enhanced_quick_evaluation(actions, game_state)

        if self.time_pressure_level >= 2 or time.time() - start_time > time_limit * 0.2:
            return quick_result

        # 阶段2：候选筛选
        candidates = self._intelligent_candidate_selection(actions, game_state, start_time, time_limit * 0.4)
        if not candidates or time.time() - start_time > time_limit * 0.4:
            return quick_result

        # 阶段3：深度搜索 (根据时间压力调整)
        if self.time_pressure_level >= 1:
            # 中等压力：轻量级搜索
            search_result = self._lightweight_search(candidates, game_state, start_time, time_limit * 0.7)
            return search_result if search_result else quick_result
        else:
            # 低压力：完整搜索
            try:
                mcts_result = self._adaptive_mcts_search(candidates, game_state, start_time, time_limit * 0.85)
                return mcts_result if mcts_result else quick_result
            except:
                return quick_result

    def _intelligent_candidate_selection(self, actions, game_state, start_time, time_limit):
        """智能候选选择 - 保持选择质量但提高效率"""
        if not actions:
            return []

        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 紧急情况优先检查
            urgent_actions = []
            for action in actions[:15]:  # 限制检查范围
                if time.time() - start_time > time_limit * 0.3:
                    break

                coords = action.get('coords')
                if not coords:
                    continue

                # 快速紧急检查
                if self._quick_win_check(board, coords, my_color):
                    return [action]  # 立即返回获胜动作
                elif self._quick_block_check(board, coords, enemy_color):
                    urgent_actions.append(action)

            if urgent_actions:
                return urgent_actions[:3]  # 返回前3个紧急动作

            # 常规评分和排序
            scored_actions = []
            eval_limit = min(len(actions), 12 if self.time_pressure_level >= 1 else 15)

            for action in actions[:eval_limit]:
                if time.time() - start_time > time_limit * 0.8:
                    break

                coords = action.get('coords')
                if not coords:
                    continue

                # 多维度快速评估
                base_score = self._cached_action_evaluation(action, game_state)
                position_bonus = self.position_weights.get(coords, 0) * 100

                # 快速奖励
                if coords in HOTB_COORDS:
                    position_bonus += 3000

                total_score = base_score + position_bonus
                scored_actions.append((action, total_score))

            # 排序并返回候选
            scored_actions.sort(key=lambda x: x[1], reverse=True)
            max_candidates = min(6 if self.time_pressure_level >= 1 else 8, len(scored_actions))

            return [action for action, score in scored_actions[:max_candidates]]

        except:
            return actions[:5]

    def _lightweight_search(self, candidates, game_state, start_time, time_limit):
        """轻量级搜索 - 快速但有效的评估"""
        if not candidates:
            return None

        best_action = candidates[0]
        best_score = float('-inf')

        for action in candidates:
            if time.time() - start_time >= time_limit:
                break

            # 基础评分
            score = self._cached_action_evaluation(action, game_state)

            # 简单的前瞻奖励
            coords = action.get('coords')
            if coords:
                lookahead_bonus = self._quick_lookahead(action, game_state)
                score += lookahead_bonus * 0.2

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _adaptive_mcts_search(self, candidates, game_state, start_time, time_limit):
        """自适应MCTS搜索 - 根据剩余时间调整迭代次数"""
        if not candidates:
            return None

        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.05:
            return candidates[0]

        # 动态调整参数
        num_candidates = len(candidates)
        base_iterations = int(remaining_time * 30)  # 基础迭代次数
        iterations_per_candidate = max(3, base_iterations // num_candidates)

        action_stats = {}

        for action in candidates:
            if time.time() - start_time >= time_limit:
                break

            rewards = []

            for _ in range(iterations_per_candidate):
                if time.time() - start_time >= time_limit:
                    break

                # 快速模拟
                reward = self._enhanced_simulate_action(action, game_state)
                rewards.append(reward)

            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards) if len(rewards) > 1 else 0
                confidence = mean_reward + math.sqrt(variance / len(rewards)) if variance > 0 else mean_reward

                action_stats[action] = confidence

        # 选择最佳
        if action_stats:
            best_action = max(action_stats.items(), key=lambda x: x[1])[0]
            return best_action

        return candidates[0]

    def _enhanced_simulate_action(self, action, game_state):
        """增强的动作模拟 - 更准确的奖励计算"""
        try:
            coords = action.get('coords')
            if not coords:
                return 0

            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 基础位置价值
            base_value = self.position_weights.get(coords, 0)

            # 威胁价值 (降低权重避免过拟合)
            threat_value = self._calc_position_threat_value(board, coords, my_color, enemy_color)

            # HOTB特殊奖励
            hotb_bonus = 0
            if coords in HOTB_COORDS:
                hotb_bonus = self._calc_hotb_value(board, my_color, enemy_color) * 0.1

            # 随机噪声 (减少噪声影响)
            noise = random.uniform(0.95, 1.05)

            return (base_value + threat_value * 0.08 + hotb_bonus) * noise

        except:
            return random.uniform(0, 100)

    def _enhanced_quick_evaluation(self, actions, game_state):
        """增强的快速评估 - 保持快速但提高准确性"""
        if not actions:
            return None

        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'
        except:
            return actions[0]

        # 开局HOTB优先
        if self.turn_count <= 6:
            for action in actions:
                coords = action.get('coords')
                if coords and coords in HOTB_COORDS and board[coords[0]][coords[1]] == '0':
                    return action

        # 紧急情况处理
        for action in actions:
            coords = action.get('coords')
            if coords:
                if self._quick_win_check(board, coords, my_color):
                    return action
                if self._quick_block_check(board, coords, enemy_color):
                    return action

        # 快速评估 - 适应性调整评估数量
        eval_limit = min(len(actions), 8 if self.time_pressure_level >= 1 else 12)
        best_action = actions[0]
        best_score = -999999

        for action in actions[:eval_limit]:
            coords = action.get('coords')
            if not coords:
                continue

            score = self._cached_action_evaluation(action, game_state)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ======== 保持原有的核心评估函数 ========

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
                    score += 15000
                elif enemy_threat >= 3:
                    score += 4000
                elif enemy_threat >= 2:
                    score += 800

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

        bonus = my_control * 80 - enemy_control * 120
        if my_control >= 3:
            bonus += 500
        if my_control == 4:
            bonus += 1500
        return bonus

    def _quick_win_check(self, board, coords, my_color):
        """快速获胜检查"""
        for dx, dy in self.directions:
            if self._quick_threat_after_move(board, coords, dx, dy, my_color) >= 5:
                return True
        return False

    def _quick_block_check(self, board, coords, enemy_color):
        """快速阻断检查"""
        for dx, dy in self.directions:
            if self._quick_threat_after_move(board, coords, dx, dy, enemy_color) >= 4:
                return True
        return False

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

    # ======== 保持原有的其他核心函数 ========

    def _eval_normal_card(self, card, board, my_color, enemy_color):
        """评估普通卡片"""
        try:
            positions = COORDS.get(card, [])
            if not positions:
                return 0

            max_value = 0
            for r, c in positions:
                if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                    value = self._calc_position_threat_value(board, (r, c), my_color, enemy_color)
                    value += self.position_weights.get((r, c), 0) * 50
                    max_value = max(max_value, value)

            return max_value
        except:
            return 0

    def _calc_removal_value(self, board, coords, enemy_color):
        """计算移除价值"""
        r, c = coords
        total_value = self.position_weights.get(coords, 0) * 50

        # HOTB特殊价值
        if coords in HOTB_COORDS:
            total_value += 2000

        # 检查破坏的威胁线
        for line_idx in self.position_to_lines.get(coords, []):
            line = self.all_lines[line_idx]
            enemy_count = sum(1 for pos in line if board[pos[0]][pos[1]] == enemy_color)
            corner_count = sum(1 for pos in line if pos in CORNER_POSITIONS)

            threat_level = enemy_count + corner_count
            if threat_level >= 4:
                total_value += 8000
            elif threat_level >= 3:
                total_value += 3000
            elif threat_level >= 2:
                total_value += 800

        return total_value

    def _calc_position_threat_value(self, board, coords, my_color, enemy_color):
        """计算位置威胁价值"""
        r, c = coords
        total_value = 0

        # 检查所有相关的威胁线
        for line_idx in self.position_to_lines.get(coords, []):
            line = self.all_lines[line_idx]

            # 己方威胁
            my_count = sum(1 for pos in line if board[pos[0]][pos[1]] == my_color)
            enemy_count = sum(1 for pos in line if board[pos[0]][pos[1]] == enemy_color)
            empty_count = sum(1 for pos in line if board[pos[0]][pos[1]] == '0')
            corner_count = sum(1 for pos in line if pos in CORNER_POSITIONS)

            # 己方放置价值
            if enemy_count == 0:  # 无敌方阻挡
                my_threat = my_count + corner_count + 1  # +1是放置的棋子
                if my_threat >= 5:
                    total_value += 30000  # 获胜
                elif my_threat >= 4:
                    total_value += 8000  # 强威胁
                elif my_threat >= 3:
                    total_value += 2000  # 中威胁
                elif my_threat >= 2:
                    total_value += 400  # 弱威胁

            # 阻断敌方价值
            if my_count == 0 and enemy_count > 0:  # 可以阻断
                enemy_threat = enemy_count + corner_count
                if enemy_threat >= 4:
                    total_value += 15000  # 阻止获胜
                elif enemy_threat >= 3:
                    total_value += 4000  # 阻止强威胁
                elif enemy_threat >= 2:
                    total_value += 1000  # 阻止中威胁

        # HOTB特殊处理
        if coords in HOTB_COORDS:
            hotb_value = self._calc_hotb_value(board, my_color, enemy_color)
            total_value += hotb_value

        return total_value

    def _calc_hotb_value(self, board, my_color, enemy_color):
        """计算HOTB价值"""
        my_control = sum(1 for r, c in HOTB_COORDS if board[r][c] == my_color)
        enemy_control = sum(1 for r, c in HOTB_COORDS if board[r][c] == enemy_color)

        # 心脏策略
        if my_control == 3 and enemy_control == 0:
            return 25000  # 立即获胜
        elif my_control == 2 and enemy_control == 0:
            return 6000  # 强控制
        elif enemy_control == 3 and my_control == 0:
            return 20000  # 必须争夺
        elif enemy_control == 2 and my_control == 0:
            return 4000  # 重要争夺
        elif my_control >= 1 and enemy_control == 0:
            return 1500  # 发展优势
        else:
            return 800  # 基础争夺

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

    def _quick_board_hash(self, board):
        """快速棋盘哈希"""
        return hash(tuple(tuple(row) for row in board[::2]))

    def _efficient_cache_cleanup(self):
        """高效缓存清理"""
        keys_to_remove = list(self.eval_cache.keys())[:self.max_cache // 3]
        for key in keys_to_remove:
            self.eval_cache.pop(key, None)
        self.cache_size = len(self.eval_cache)

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