from numpy.random._common import namedtuple
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
import heapq
import time
import itertools
import math

MAX_THINK_TIME = 0.8
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]

TTEntry = namedtuple('TTEntry', 'depth score flag best_move')


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.counter = itertools.count()

        # 方向向量
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 游戏阶段阈值
        self.EARLY_PHASE_THRESHOLD = 25  # 前期：0-25个棋子，攻击优先
        self.LATE_PHASE_THRESHOLD = 65  # 后期：65+个棋子，防御优先

        # 预缓存数据
        self.position_weights_cache = {}
        self.threat_patterns_cache = {}
        self.strategic_positions_cache = []
        self.opening_book = []
        self.pre_cache_completed = False

        # 阶段权重配置
        self.phase_weights = {
            'early': {'offense': 1.8, 'defense': 0.6, 'position': 1.4, 'center': 2.0},
            'mid': {'offense': 1.2, 'defense': 1.2, 'position': 1.0, 'center': 1.1},
            'late': {'offense': 0.7, 'defense': 2.0, 'position': 0.8, 'center': 0.9}
        }

    def GameStarts(self, first_player):
        """游戏开始时的15秒预缓存"""
        start_time = time.time()
        try:
            self._pre_cache_analysis(time_limit=14.5)  # 留出安全边界
            self.pre_cache_completed = True
        except Exception as e:
            # 如果预缓存失败，至少确保基本功能可用
            self.pre_cache_completed = False

        cache_time = time.time() - start_time
        # 可以记录但不打印

    def _pre_cache_analysis(self, time_limit=14.5):
        """15秒预缓存分析"""
        cache_start = time.time()

        # 预计算位置权重
        self._precompute_position_weights()

        # 预分析威胁模式
        if time.time() - cache_start < time_limit * 0.3:
            self._precompute_threat_patterns()

        # 识别战略位置
        if time.time() - cache_start < time_limit * 0.6:
            self._identify_strategic_positions()

        # 构建开局库
        if time.time() - cache_start < time_limit * 0.9:
            self._build_opening_book()

    def _precompute_position_weights(self):
        """预计算不同阶段的位置权重"""
        for phase in ['early', 'mid', 'late']:
            self.position_weights_cache[phase] = [[0] * 10 for _ in range(10)]

        center = 4.5
        for r in range(10):
            for c in range(10):
                base_value = max(1, int(15 - math.sqrt((r - center) ** 2 + (c - center) ** 2) * 2))
                hotb_bonus = 30 if (r, c) in HOTB_COORDS else 0
                edge_penalty = 5 if (r == 0 or r == 9 or c == 0 or c == 9) else 0

                self.position_weights_cache['early'][r][c] = base_value + hotb_bonus * 1.5 - edge_penalty
                self.position_weights_cache['mid'][r][c] = base_value + hotb_bonus - edge_penalty
                self.position_weights_cache['late'][r][c] = base_value + hotb_bonus * 0.8 - edge_penalty * 0.5

    def _precompute_threat_patterns(self):
        """预计算威胁模式"""
        self.threat_patterns_cache = {
            'win_5': 10000,
            'open_4': 5000,
            'half_4': 1000,
            'open_3': 500,
            'half_3': 100,
            'open_2': 50,
            'half_2': 20
        }

    def _identify_strategic_positions(self):
        """识别战略要点"""
        self.strategic_positions_cache = []
        for r in range(3, 7):  # 缩小范围以提高效率
            for c in range(3, 7):
                directions_count = 0
                for dx, dy in self.directions:
                    line_potential = 0
                    for i in range(-3, 4):  # 缩小搜索范围
                        x, y = r + dx * i, c + dy * i
                        if 0 <= x < 10 and 0 <= y < 10:
                            line_potential += 1
                    if line_potential >= 5:
                        directions_count += 1

                if directions_count >= 2:
                    self.strategic_positions_cache.append((r, c))

    def _build_opening_book(self):
        """构建开局定式"""
        self.opening_book = [
            (4, 4), (5, 5), (4, 5), (5, 4),  # HOTB优先
            (3, 3), (6, 6), (3, 6), (6, 3)  # 对角扩展
        ]

    def _get_game_phase(self, board):
        """根据棋盘棋子数量判断游戏阶段"""
        piece_count = sum(1 for row in board for cell in row if cell != '0')

        if piece_count <= self.EARLY_PHASE_THRESHOLD:
            return 'early'
        elif piece_count >= self.LATE_PHASE_THRESHOLD:
            return 'late'
        else:
            return 'mid'

    def SelectAction(self, actions, game_state):
        self.start_time = time.time()

        try:
            # 如果只有一个动作，直接返回
            if len(actions) <= 1:
                return actions[0] if actions else None

            # 获取当前游戏阶段
            current_phase = self._get_game_phase(game_state.board.chips)

            # 时间检查点1
            if time.time() - self.start_time > MAX_THINK_TIME * 0.1:
                return actions[0]

            # 开局使用预缓存的开局库
            if current_phase == 'early' and self.pre_cache_completed:
                opening_move = self._get_opening_move(actions, game_state)
                if opening_move:
                    return opening_move

            # 时间检查点2
            if time.time() - self.start_time > MAX_THINK_TIME * 0.2:
                return actions[0]

            # 检查立即获胜（快速）
            winning_move = self._find_immediate_win(actions[:20], game_state)  # 限制搜索范围
            if winning_move:
                return winning_move

            # 时间检查点3
            if time.time() - self.start_time > MAX_THINK_TIME * 0.4:
                return actions[0]

            # 检查必须防御（快速）
            blocking_move = self._find_critical_block(actions[:15], game_state)  # 限制搜索范围
            if blocking_move:
                return blocking_move

            # 时间检查点4
            if time.time() - self.start_time > MAX_THINK_TIME * 0.6:
                return actions[0]

            # 根据阶段调整搜索策略
            if current_phase == 'early':
                return self._early_phase_search(actions, game_state)
            elif current_phase == 'late':
                return self._late_phase_search(actions, game_state)
            else:
                return self._mid_phase_search(actions, game_state)

        except Exception as e:
            # 任何异常都返回第一个可用动作
            return actions[0] if actions else None

    def _get_opening_move(self, actions, game_state):
        """获取开局走法"""
        for preferred_pos in self.opening_book:
            for action in actions:
                if action.get('coords') == preferred_pos:
                    return action
        return None

    def _early_phase_search(self, actions, game_state):
        """前期搜索：攻击优先"""
        best_action = None
        best_score = float('-inf')

        # 限制搜索范围以确保时间控制
        search_actions = actions[:min(10, len(actions))]

        for action in search_actions:
            if time.time() - self.start_time > MAX_THINK_TIME * 0.9:
                break

            score = self.evaluate_state(game_state, action)
            if action.get('coords') and action['coords'] in self.strategic_positions_cache:
                score += 100

            if score > best_score:
                best_score = score
                best_action = action

        return best_action or actions[0]

    def _mid_phase_search(self, actions, game_state):
        """中期搜索：平衡策略"""
        return self._minimax_search(actions, game_state, depth=2)

    def _late_phase_search(self, actions, game_state):
        """后期搜索：防御优先，更深搜索"""
        return self._minimax_search(actions, game_state, depth=3)

    def _minimax_search(self, actions, game_state, depth):
        """Minimax搜索"""
        best_action = None
        best_score = float('-inf')

        # 快速排序，限制搜索范围
        sorted_actions = sorted(actions[:min(8, len(actions))],
                                key=lambda a: self.heuristic(game_state, a))

        for action in sorted_actions:
            if time.time() - self.start_time > MAX_THINK_TIME * 0.9:
                break

            try:
                sim_state = self.fast_simulate(game_state, action)
                score = self._minimax(sim_state, depth - 1, float('-inf'), float('inf'), False)

                if score > best_score:
                    best_score = score
                    best_action = action
            except:
                # 如果模拟失败，跳过这个动作
                continue

        return best_action or actions[0]

    def _minimax(self, state, depth, alpha, beta, maximizing):
        """Minimax递归"""
        # 严格的时间检查
        if depth <= 0 or time.time() - self.start_time > MAX_THINK_TIME * 0.85:
            return self.evaluate_state(state)

        try:
            actions = self.rule.getLegalActions(state, self.id if maximizing else 1 - self.id)
            if not actions:
                return self.evaluate_state(state)

            # 限制搜索宽度
            actions = actions[:min(4, len(actions))]

            if maximizing:
                max_eval = float('-inf')
                for action in actions:
                    if time.time() - self.start_time > MAX_THINK_TIME * 0.85:
                        break
                    sim_state = self.fast_simulate(state, action, self.id)
                    eval_score = self._minimax(sim_state, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float('inf')
                for action in actions:
                    if time.time() - self.start_time > MAX_THINK_TIME * 0.85:
                        break
                    sim_state = self.fast_simulate(state, action, 1 - self.id)
                    eval_score = self._minimax(sim_state, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                return min_eval
        except:
            return self.evaluate_state(state)

    def _find_immediate_win(self, actions, game_state):
        """寻找立即获胜走法"""
        for action in actions:
            if time.time() - self.start_time > MAX_THINK_TIME * 0.3:
                break
            if action.get('coords'):
                try:
                    sim_state = self.fast_simulate(game_state, action)
                    if self._check_win(sim_state, self.id):
                        return action
                except:
                    continue
        return None

    def _find_critical_block(self, actions, game_state):
        """寻找必须阻挡的威胁"""
        try:
            # 快速检查敌人的获胜威胁
            critical_positions = set()

            # 只检查最有可能的敌人动作
            for action in actions:
                if time.time() - self.start_time > MAX_THINK_TIME * 0.35:
                    break
                if action.get('coords'):
                    try:
                        # 检查如果敌人在这个位置会不会获胜
                        sim_state = self.fast_simulate(game_state, action, 1 - self.id)
                        if self._check_win(sim_state, 1 - self.id):
                            critical_positions.add(action['coords'])
                    except:
                        continue

            # 找到能阻挡的我方动作
            for action in actions:
                if action.get('coords') in critical_positions:
                    return action

        except:
            pass
        return None

    def _check_win(self, state, player_id):
        """检查是否获胜"""
        try:
            agent = state.agents[player_id]
            board = state.board.chips
            color = agent.colour

            # 快速检查最近放置的棋子周围
            for r in range(10):
                for c in range(10):
                    if board[r][c] == color:
                        for dx, dy in self.directions:
                            count = 1
                            # 正向检查
                            for i in range(1, 5):
                                x, y = r + dx * i, c + dy * i
                                if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == color:
                                    count += 1
                                else:
                                    break
                            # 反向检查
                            for i in range(1, 5):
                                x, y = r - dx * i, c - dy * i
                                if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == color:
                                    count += 1
                                else:
                                    break
                            if count >= 5:
                                return True
        except:
            pass
        return False

    def a_star(self, initial_state, candidate_moves):
        """简化的A*搜索算法"""
        try:
            best_move = None
            best_score = float('-inf')

            # 限制候选动作数量
            moves_to_check = candidate_moves[:min(8, len(candidate_moves))]

            for move in moves_to_check:
                if time.time() - self.start_time > MAX_THINK_TIME * 0.9:
                    break

                try:
                    sim_state = self.fast_simulate(initial_state, move)
                    score = self.evaluate_state(sim_state, move)

                    if score > best_score:
                        best_score = score
                        best_move = move
                except:
                    continue

            return best_move or candidate_moves[0]
        except:
            return candidate_moves[0] if candidate_moves else None

    def fast_simulate(self, state, action, player_id=None):
        """快速模拟动作"""
        if player_id is None:
            player_id = self.id

        try:
            new_state = state.copy() if hasattr(state, "copy") else self.custom_shallow_copy(state)

            if action.get('type') == 'place' and action.get('coords'):
                r, c = action['coords']
                agent = new_state.agents[player_id]
                new_state.board.chips[r][c] = agent.colour

            return new_state
        except:
            return state

    def custom_shallow_copy(self, state):
        from copy import deepcopy
        return deepcopy(state)

    def heuristic(self, state, action):
        """启发式函数，根据游戏阶段调整"""
        if action.get('type') != 'place' or not action.get('coords'):
            return 100

        try:
            r, c = action['coords']
            board = state.board.chips
            me = state.agents[self.id]
            color = me.colour
            enemy = 'r' if color == 'b' else 'b'

            # 获取当前阶段
            phase = self._get_game_phase(board)
            weights = self.phase_weights[phase]

            score = 0

            # 基础评分
            score += self.center_bias(r, c) * weights['center']

            # 使用预缓存的位置权重
            if self.pre_cache_completed and phase in self.position_weights_cache:
                score += self.position_weights_cache[phase][r][c] * weights['position']

            return max(1, 100 - score)
        except:
            return 50

    def center_bias(self, r, c):
        """中心偏向评分"""
        distance = abs(r - 4.5) + abs(c - 4.5)
        score = 0

        if (r, c) in HOTB_COORDS:
            score += 30
        elif 3 <= r <= 6 and 3 <= c <= 6:
            score += 15
        elif 2 <= r <= 7 and 2 <= c <= 7:
            score += 5

        score += max(0, int(6 - distance))
        return score

    def chain_score(self, board, r, c, color):
        """连子评分"""
        total_score = 0
        try:
            for dx, dy in self.directions:
                count = 1
                blocks = 0

                # 正向检查
                for i in range(1, 5):
                    x, y = r + dx * i, c + dy * i
                    if 0 <= x < 10 and 0 <= y < 10:
                        if board[x][y] == color:
                            count += 1
                        elif board[x][y] == '0':
                            break
                        else:
                            blocks += 1
                            break
                    else:
                        blocks += 1
                        break

                # 反向检查
                for i in range(1, 5):
                    x, y = r - dx * i, c - dy * i
                    if 0 <= x < 10 and 0 <= y < 10:
                        if board[x][y] == color:
                            count += 1
                        elif board[x][y] == '0':
                            break
                        else:
                            blocks += 1
                            break
                    else:
                        blocks += 1
                        break

                # 评分
                if count >= 5:
                    total_score += 1000
                elif count == 4 and blocks == 0:
                    total_score += 500
                elif count == 4 and blocks == 1:
                    total_score += 100
                elif count == 3 and blocks == 0:
                    total_score += 60
                elif count == 3 and blocks == 1:
                    total_score += 20
                elif count == 2 and blocks == 0:
                    total_score += 10
        except:
            pass

        return total_score

    def block_enemy_score(self, board, r, c, enemy_color):
        """阻挡敌人评分"""
        score = 0
        try:
            for dx, dy in self.directions:
                enemy_chain = 0
                for i in range(1, 4):  # 缩小范围
                    x, y = r + dx * i, c + dy * i
                    if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy_color:
                        enemy_chain += 1
                    else:
                        break
                if enemy_chain >= 2:
                    score += 30
        except:
            pass
        return score

    def hotb_score(self, board, color):
        """HOTB区域控制评分"""
        try:
            enemy = 'r' if color == 'b' else 'b'
            score = 0

            for r, c in HOTB_COORDS:
                if board[r][c] == color:
                    score += 25
                elif board[r][c] == enemy:
                    score -= 40

            return score
        except:
            return 0

    def evaluate_state(self, state, action=None):
        """状态评估，根据游戏阶段调整权重"""
        try:
            agent = state.agents[self.id]
            board = [row[:] for row in state.board.chips]

            if action and action.get('coords'):
                r, c = action['coords']
                board[r][c] = agent.colour

            # 获取游戏阶段
            phase = self._get_game_phase(board)
            weights = self.phase_weights[phase]

            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            score = 0
            score += self.score_friendly_chain(board, my_color) * weights['offense']
            score += self.score_enemy_threat(board, enemy_color) * weights['defense']
            score += self.score_hotb_control(board, my_color) * weights['center']

            return score
        except:
            return 0

    def score_friendly_chain(self, board, color):
        """友方连子评分"""
        try:
            max_chain = 0
            for r in range(10):
                for c in range(10):
                    if board[r][c] != color:
                        continue
                    for dx, dy in self.directions:
                        count = 1
                        for i in range(1, 5):
                            x, y = r + dx * i, c + dy * i
                            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == color:
                                count += 1
                            else:
                                break
                        max_chain = max(max_chain, count)

            if max_chain >= 5:
                return 10000
            elif max_chain == 4:
                return 1000
            elif max_chain == 3:
                return 100
            elif max_chain == 2:
                return 20
            return 0
        except:
            return 0

    def score_enemy_threat(self, board, enemy_color):
        """敌方威胁评分"""
        try:
            max_enemy_chain = 0

            for r in range(10):
                for c in range(10):
                    if board[r][c] != enemy_color:
                        continue
                    for dx, dy in self.directions:
                        count = 1
                        for i in range(1, 5):
                            x, y = r + dx * i, c + dy * i
                            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy_color:
                                count += 1
                            else:
                                break
                        max_enemy_chain = max(max_enemy_chain, count)

            if max_enemy_chain >= 4:
                return -2000
            elif max_enemy_chain == 3:
                return -200
            elif max_enemy_chain == 2:
                return -30

            return 0
        except:
            return 0

    def score_hotb_control(self, board, color):
        """HOTB控制评分"""
        try:
            control = sum(1 for r, c in HOTB_COORDS if board[r][c] == color)
            return control * 25
        except:
            return 0

    def board_hash(self, state):
        """棋盘哈希"""
        try:
            return tuple(tuple(row) for row in state.board.chips)
        except:
            return 0