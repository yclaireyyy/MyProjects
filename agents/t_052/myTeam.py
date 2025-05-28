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
        self.total_thinking_time = 0

        # 预计算核心数据
        self._precompute_data()

        # 缓存系统
        self.eval_cache = {}
        self.pattern_cache = {}
        self.jack_cache = {}
        self.cache_size = 0
        self.max_cache = 100

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
        for r in range(10):
            for c in range(10):
                weight = self.position_weights.get((r, c), 0)
                if weight > 6:
                    self.priority_positions.append((r, c))

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
        """主决策入口"""
        self.turn_count += 1
        start_time = time.time()

        # 动态时间管理
        time_limit = self._get_time_limit()

        if not actions or time_limit < 0.03:
            return self._emergency_decision(actions, game_state)

        try:
            # 判断动作类型
            if self._is_card_selection(actions, game_state):
                result = self._card_selection(actions, game_state, start_time, time_limit)
            else:
                result = self._search_strategy(actions, game_state, start_time, time_limit)

            # 记录使用时间
            elapsed = time.time() - start_time
            self.total_thinking_time += elapsed

            return result

        except Exception:
            return self._emergency_decision(actions, game_state)

    def _get_time_limit(self):
        """获取时间限制"""
        if not self.startup_used:
            elapsed = time.time() - self.startup_start
            remaining = 15.0 - elapsed

            if remaining <= 0.5:
                self.startup_used = True
                return 0.7
            else:
                return min(remaining - 0.4, 0.8)
        else:
            # 根据历史表现调整
            if self.turn_count > 5:
                avg_time = self.total_thinking_time / self.turn_count
                if avg_time > 0.6:
                    return 0.6
            return 0.75

    def _card_selection(self, actions, game_state, start_time, time_limit):
        """卡片选择"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']
        if not trade_actions:
            return random.choice(actions) if actions else None

        try:
            board = game_state.board.chips
            agent = game_state.agents[self.id]
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 快速评估所有卡片
            best_action = trade_actions[0]
            best_value = -999999

            for action in trade_actions:
                if time.time() - start_time > time_limit * 0.6:
                    break

                draft_card = action.get('draft_card', '').lower()

                if draft_card in ['js', 'jh']:  # 单眼J
                    value = self._eval_one_eyed_jack(board, enemy_color)
                elif draft_card in ['jc', 'jd']:  # 双眼J
                    value = self._eval_two_eyed_jack(board, my_color, enemy_color)
                elif draft_card in COORDS:
                    value = self._eval_normal_card(draft_card, board, my_color, enemy_color)
                else:
                    value = 0

                if value > best_value:
                    best_value = value
                    best_action = action

            return best_action

        except:
            return random.choice(trade_actions)

    def _eval_one_eyed_jack(self, board, enemy_color):
        """评估单眼J"""
        # 快速找到最有价值的移除目标
        max_value = 0

        # 优先检查HOTB和高价值位置
        priority_targets = []
        for r, c in HOTB_COORDS:
            if board[r][c] == enemy_color:
                priority_targets.append((r, c))

        for r, c in self.priority_positions:
            if board[r][c] == enemy_color and (r, c) not in priority_targets:
                priority_targets.append((r, c))

        # 评估移除价值
        for target in priority_targets[:10]:  # 最多评估10个目标
            value = self._calc_removal_value(board, target, enemy_color)
            max_value = max(max_value, value)

        return max_value if max_value > 0 else 3000

    def _eval_two_eyed_jack(self, board, my_color, enemy_color):
        """评估双眼J"""
        max_value = 0
        available_count = 0

        # 只评估高价值位置
        for r, c in self.priority_positions:
            if board[r][c] == '0':
                available_count += 1

                # 基础位置价值
                pos_value = self.position_weights.get((r, c), 0) * 100

                # 威胁价值
                threat_value = self._calc_position_threat_value(board, (r, c), my_color, enemy_color)

                total_value = pos_value + threat_value
                max_value = max(max_value, total_value)

        # 灵活性奖励
        flexibility = min(available_count * 50, 1000)
        return max_value + flexibility

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

    def _search_strategy(self, actions, game_state, start_time, time_limit):
        """搜索策略"""
        # 快速预评估
        quick_result = self._quick_evaluation(actions, game_state)

        if time.time() - start_time > time_limit * 0.25:
            return quick_result

        # 候选筛选
        candidates = self._select_candidates(actions, game_state, start_time, time_limit * 0.5)
        if time.time() - start_time > time_limit * 0.5:
            return candidates[0] if candidates else quick_result

        # MCTS搜索
        try:
            mcts_result = self._mcts_search(candidates, game_state, start_time, time_limit * 0.9)
            return mcts_result if mcts_result else quick_result
        except:
            return quick_result

    def _select_candidates(self, actions, game_state, start_time, time_limit):
        """选择候选动作"""
        if not actions:
            return []

        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            scored_actions = []

            for action in actions:
                if time.time() - start_time > time_limit:
                    break

                coords = action.get('coords')
                if not coords:
                    continue

                # 多维度评估
                base_score = self._cached_action_evaluation(action, game_state)

                # 紧急情况奖励
                urgency_bonus = 0
                if self._quick_win_check(board, coords, my_color):
                    urgency_bonus += 40000
                elif self._quick_block_check(board, coords, enemy_color):
                    urgency_bonus += 20000
                elif coords in HOTB_COORDS:
                    urgency_bonus += 3000

                # 位置价值
                position_bonus = self.position_weights.get(coords, 0) * 100

                total_score = base_score + urgency_bonus + position_bonus
                scored_actions.append((action, total_score))

            # 排序并返回候选
            scored_actions.sort(key=lambda x: x[1], reverse=True)
            max_candidates = min(6, len(scored_actions))

            return [action for action, score in scored_actions[:max_candidates]]

        except:
            return actions[:5]

    def _mcts_search(self, candidates, game_state, start_time, time_limit):
        """MCTS搜索"""
        if not candidates:
            return None

        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.05:
            return candidates[0]

        # 参数调整
        num_candidates = len(candidates)
        total_iterations = max(15, int(remaining_time * 25))
        iterations_per_candidate = max(3, total_iterations // num_candidates)

        action_stats = {}

        for action in candidates:
            if time.time() - start_time >= time_limit:
                break

            rewards = []

            for _ in range(iterations_per_candidate):
                if time.time() - start_time >= time_limit:
                    break

                # 快速模拟
                reward = self._simulate_action(action, game_state)
                rewards.append(reward)

            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
                confidence = mean_reward + math.sqrt(variance / len(rewards))

                action_stats[action] = confidence

        # 选择最佳
        if action_stats:
            best_action = max(action_stats.items(), key=lambda x: x[1])[0]
            return best_action

        return candidates[0]

    def _simulate_action(self, action, game_state):
        """模拟动作"""
        try:
            coords = action.get('coords')
            if not coords:
                return 0

            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 基础评估
            base_value = self.position_weights.get(coords, 0)
            threat_value = self._calc_position_threat_value(board, coords, my_color, enemy_color)

            # 添加随机性
            noise = random.uniform(0.9, 1.1)

            return (base_value + threat_value * 0.1) * noise

        except:
            return random.uniform(0, 100)

    def _quick_evaluation(self, actions, game_state):
        """快速评估"""
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

        # 快速评估
        best_action = actions[0]
        best_score = -999999

        for action in actions[:8]:
            coords = action.get('coords')
            if not coords:
                continue

            score = self._cached_action_evaluation(action, game_state)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ======== 保持原有的核心评估函数 ========

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