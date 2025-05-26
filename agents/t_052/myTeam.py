from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import heapq
import time
import itertools
import random

MAX_THINK_TIME = 0.95
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]


class CardEvaluator:
    """卡牌评估器 """

    def __init__(self, agent):
        self.agent = agent

    def evaluate_card(self, card, state):
        """评估卡牌在当前状态下的价值"""
        if not card or not state or not hasattr(state, 'board'):
            return 0

        board = state.board.chips

        # 优先级1：双眼J - 直接最高分
        if self._is_two_eyed_jack(card):
            return 10000
        # 优先级2：单眼J - 次高分
        if self._is_one_eyed_jack(card):
            return 5000
        # 优先级3：普通卡牌 - 使用指数评分
        if card in COORDS:
            return self._exponential_card_evaluation(card, state)
        return 0

    def _exponential_card_evaluation(self, card, state):
        """基于指数的普通卡牌评估"""
        if card not in COORDS:
            return 0

        board = state.board.chips
        total_score = 0

        # 获取该卡牌对应的所有可能位置
        positions = COORDS[card] if isinstance(COORDS[card], list) else [COORDS[card]]
        for pos in positions:
            if len(pos) != 2:
                continue
            r, c = pos
            # 检查位置是否可用
            if not self._is_position_available(board, r, c):
                continue
            # 计算该位置的指数评分
            position_score = self._calculate_position_score(board, r, c)
            total_score += position_score

        # 如果有多个位置，取平均值
        return total_score / max(1, len(positions))

    def _calculate_position_score(self, board, r, c):
        """计算单个位置的指数评分"""
        my_color = self._get_my_color()
        if not my_color:
            return 0

        total_score = 0
        # 四个主要方向：水平， 垂直，主对角线，反对角线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            my_pieces = self._count_my_pieces_in_direction(board, r, c, dx, dy)
            direction_score = self._exponential_scoring(my_pieces)
            total_score += direction_score

        return total_score

    def _count_my_pieces_in_direction(self, board, r, c, dx, dy):
        """统计特定方向5个位置内的我方棋子数量"""
        my_color = self._get_my_color()
        if not my_color or not board:
            return 0

        my_pieces = 0
        # 检查该方向前后各4个位置（共8个位置）
        for i in range(-4, 5):
            # 跳过中心位置（即将放置的位置）
            if i == 0:
                continue
            x, y = r + i * dx, c + i * dy
            # 边界检查
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == my_color:
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

    def _get_my_color(self):
        """获取我方颜色"""
        if hasattr(self.agent, 'my_color') and self.agent.my_color:
            return self.agent.my_color
        elif hasattr(self.agent, 'colour'):
            return self.agent.colour
        return None

    def _is_two_eyed_jack(self, card):
        """检查是否为双眼J"""
        if not card:
            return False
        card_str = str(card).lower()
        return card_str in ['jc', 'jd']  # 双眼J

    def _is_one_eyed_jack(self, card):
        """检查是否为单眼J"""
        if not card:
            return False
        card_str = str(card).lower()
        return card_str in ['js', 'jh']  # 单眼J

    def _is_position_available(self, board, r, c):
        """检查位置是否可用"""
        if not board or not (0 <= r < 10 and 0 <= c < 10):
            return False
        return board[r][c] == 0 or board[r][c] == '0'  # 空位


class AdvancedEvaluator:
    """高级评估器 - 整合新的评估逻辑"""

    @staticmethod
    def safe_board_access(board, r, c, default=0):
        """安全的棋盘访问"""
        if board and 0 <= r < len(board) and 0 <= c < len(board[0]):
            return board[r][c]
        return default

    @staticmethod
    def get_player_colors(state, agent_id=None):
        """获取玩家颜色"""
        # 优先从状态属性获取
        if hasattr(state, 'my_color') and hasattr(state, 'opp_color'):
            return state.my_color, state.opp_color

        # 从当前玩家推断
        player_id = getattr(state, 'current_player_id', agent_id or 0)

        if hasattr(state, 'agents') and 0 <= player_id < len(state.agents):
            agent = state.agents[player_id]
            if hasattr(agent, 'colour'):
                color = agent.colour
                enemy = 'r' if color == 'b' else 'b'
                return color, enemy

        return None, None

    @staticmethod
    def count_consecutive(board, x, y, dx, dy, color):
        """计算从(x,y)出发，在方向(dx,dy)上颜色为color的最长连续序列"""
        if not board or not color:
            return 1

        count = 1  # 起始位置算一个

        # 正向检查
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if AdvancedEvaluator.safe_board_access(board, nx, ny) == color:
                count += 1
            else:
                break

        # 反向检查
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if AdvancedEvaluator.safe_board_access(board, nx, ny) == color:
                count += 1
            else:
                break

        return min(count, 5)  # 最多返回5（形成一个序列）

    @staticmethod
    def count_enemy_threat(board, r, c, dx, dy, enemy):
        """计算敌方威胁"""
        if not board or not enemy:
            return 0

        enemy_chain = 0
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if AdvancedEvaluator.safe_board_access(board, x, y) == enemy:
                enemy_chain += 1
            else:
                break
        return enemy_chain

    @staticmethod
    def advanced_heuristic(state, action, agent_id):
        """高级启发式函数 - 整合代码1的逻辑"""
        if not action or action.get('type') != 'place' or 'coords' not in action:
            return 100  # 非放置动作或无坐标

        coords = action.get('coords')
        if not coords or len(coords) != 2:
            return 100

        r, c = coords
        if (r, c) in CORNERS:
            return 100  # 角落位置

        board = state.board.chips if hasattr(state, 'board') else None
        if not board:
            return 100

        # 获取玩家颜色
        color, enemy = AdvancedEvaluator.get_player_colors(state, agent_id)
        if not color:
            return 100

        # 创建假设放置后的棋盘
        board_copy = [row[:] for row in board] if board else None
        if not board_copy:
            return 100

        if 0 <= r < len(board_copy) and 0 <= c < len(board_copy[0]):
            board_copy[r][c] = color

        score = 0

        # 中心偏好
        distance = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 5 - distance) * 2

        # 连续链评分 - 增强版
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = AdvancedEvaluator.count_consecutive(board_copy, r, c, dx, dy, color)
            if count >= 5:
                score += 200  # 形成序列
            elif count == 4:
                score += 100
            elif count == 3:
                score += 30
            elif count == 2:
                score += 10

        # 阻止对手评分 - 增强版
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            enemy_threat = AdvancedEvaluator.count_enemy_threat(board, r, c, dx, dy, enemy)
            if enemy_threat >= 3:
                score += 50  # 高优先级阻断

        # 中心控制评分
        hotb_controlled = 0
        for x, y in HOTB_COORDS:
            if AdvancedEvaluator.safe_board_access(board_copy, x, y) == color:
                hotb_controlled += 1
        score += hotb_controlled * 15

        # 转换为启发式分数（越低越好）
        return max(0, 100 - score)

    @staticmethod
    def advanced_state_evaluation(state, action, agent_id):
        """高级状态评估 - 整合代码1的逻辑"""
        if not state:
            return 0

        board = state.board.chips if hasattr(state, 'board') else None
        if not board:
            return 0

        # 获取玩家颜色
        my_color, opp_color = AdvancedEvaluator.get_player_colors(state, agent_id)
        if not my_color:
            return 0

        # 1. 位置评分
        position_score = 0
        for i in range(10):
            for j in range(10):
                cell_value = AdvancedEvaluator.safe_board_access(board, i, j)
                if cell_value == my_color:
                    # 位置权重
                    if (i, j) in HOTB_COORDS:
                        position_score += 1.5  # 中心位置
                    elif i in [0, 9] or j in [0, 9]:
                        position_score += 0.8  # 边缘位置
                    else:
                        position_score += 1.0  # 其他位置
                elif cell_value == opp_color:
                    # 对手的位置，负分
                    if (i, j) in HOTB_COORDS:
                        position_score -= 1.5
                    elif i in [0, 9] or j in [0, 9]:
                        position_score -= 0.8
                    else:
                        position_score -= 1.0

        # 2. 序列潜力评分
        sequence_score = AdvancedEvaluator._calculate_sequence_score(board, my_color)

        # 3. 防御评分 - 阻止对手的序列
        defense_score = AdvancedEvaluator._calculate_defense_score(board, opp_color)

        # 4. 中心控制评分
        hotb_score = 0
        for x, y in HOTB_COORDS:
            cell_value = AdvancedEvaluator.safe_board_access(board, x, y)
            if cell_value == my_color:
                hotb_score += 5
            elif cell_value == opp_color:
                hotb_score -= 5

        # 5. 综合评分
        total_score = position_score + sequence_score + defense_score + hotb_score

        # 归一化到合理区间
        return total_score

    @staticmethod
    def _calculate_sequence_score(board, color):
        """计算序列得分"""
        if not board or not color:
            return 0

        sequence_score = 0
        for i in range(10):
            for j in range(10):
                if AdvancedEvaluator.safe_board_access(board, i, j) == color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = AdvancedEvaluator.count_consecutive(board, i, j, dx, dy, color)
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
    def _calculate_defense_score(board, opp_color):
        """计算防御得分"""
        if not board or not opp_color:
            return 0

        defense_score = 0
        for i in range(10):
            for j in range(10):
                if AdvancedEvaluator.safe_board_access(board, i, j) == opp_color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = AdvancedEvaluator.count_consecutive(board, i, j, dx, dy, opp_color)
                        if count >= 4:
                            defense_score -= 50  # 高度威胁
                        elif count == 3:
                            defense_score -= 10
        return defense_score


class myAgent(Agent):
    """整合高级评估函数的A*代理"""

    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.counter = itertools.count()

        # 初始化评估器
        self.card_evaluator = CardEvaluator(self)

        # 玩家颜色（延迟初始化）
        self.my_color = None
        self.opp_color = None

    def _initialize_colors(self, game_state):
        """初始化颜色信息"""
        if self.my_color is None and game_state and hasattr(game_state, 'agents'):
            if 0 <= self.id < len(game_state.agents):
                agent = game_state.agents[self.id]
                if hasattr(agent, 'colour'):
                    self.my_color = agent.colour
                    self.opp_color = 'r' if self.my_color == 'b' else 'b'

    def _is_card_selection(self, actions):
        """判断是否为卡牌选择"""
        return any(a and a.get('type') == 'trade' for a in actions) if actions else False

    def _select_strategic_card(self, actions, game_state):
        """卡牌选择逻辑 - 使用高级评估"""
        trade_actions = [a for a in actions if a and a.get('type') == 'trade']

        if not trade_actions:
            return random.choice(actions) if actions else None

        if not hasattr(game_state, 'display_cards') or not game_state.display_cards:
            return random.choice(trade_actions)

        # 评估所有展示牌
        best_card = None
        best_score = float('-inf')

        for card in game_state.display_cards:
            if card:
                score = self.card_evaluator.evaluate_card(card, game_state)
                if score > best_score:
                    best_score = score
                    best_card = card

        # 找到对应动作
        if best_card:
            for action in trade_actions:
                if action.get('draft_card') == best_card:
                    return action

        return random.choice(trade_actions)

    def SelectAction(self, actions, game_state):
        """主决策函数"""
        if not actions:
            return None

        self.start_time = time.time()
        self._initialize_colors(game_state)

        # 处理卡牌选择
        if self._is_card_selection(actions):
            return self._select_strategic_card(actions, game_state)

        return self.a_star(game_state, actions)

    def a_star(self, initial_state, candidate_moves):
        """A*搜索 - 使用高级评估函数"""
        pending = []
        seen_states = set()
        best_sequence = []
        top_reward = float('-inf')

        # 初始化候选动作
        for move in candidate_moves:
            if not move:
                continue

            g = 1
            h = self.heuristic(initial_state, move)
            f = g + h
            heapq.heappush(pending, (f, next(self.counter), g, h, self.fast_simulate(initial_state, move), [move]))

        while pending and (time.time() - self.start_time < MAX_THINK_TIME):
            f, _, g, h, current_state, move_history = heapq.heappop(pending)

            if not current_state or not move_history:
                continue

            last_move = move_history[-1]

            # 状态去重
            state_signature = self.get_state_signature(current_state, last_move)
            if state_signature in seen_states:
                continue
            seen_states.add(state_signature)

            # 评估当前状态
            reward = self.evaluate_state(current_state, last_move)
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history

            # 获取下一步动作
            try:
                next_steps = self.rule.getLegalActions(current_state, self.id)
                if next_steps:
                    # 使用高级启发式排序
                    next_steps.sort(key=lambda act: self.heuristic(current_state, act))

                    # 限制搜索宽度
                    for next_move in next_steps[:5]:
                        if next_move:
                            next_g = g + 1
                            next_h = self.heuristic(current_state, next_move)
                            heapq.heappush(pending, (
                                next_g + next_h, next(self.counter),
                                next_g, next_h,
                                self.fast_simulate(current_state, next_move),
                                move_history + [next_move]
                            ))
            except:
                continue

        return best_sequence[0] if best_sequence else (candidate_moves[0] if candidate_moves else None)

    def get_state_signature(self, state, last_move):
        """获取状态签名用于去重"""
        try:
            board_hash = self.board_hash(state)
            play_card = last_move.get('play_card') if last_move else None

            # 安全获取手牌
            hand_tuple = ()
            if (hasattr(state, 'agents') and
                    0 <= self.id < len(state.agents) and
                    hasattr(state.agents[self.id], 'hand')):
                hand = state.agents[self.id].hand
                hand_tuple = tuple(hand) if hand else ()

            return (board_hash, play_card, hand_tuple)
        except:
            return (id(state), str(last_move))

    def fast_simulate(self, state, action):
        """快速模拟执行动作"""
        if not state or not action:
            return state

        try:
            new_state = state.copy() if hasattr(state, "copy") else self.custom_shallow_copy(state)
            if not new_state:
                return state

            # 处理放置动作
            if action.get('type') == 'place' and action.get('coords'):
                coords = action['coords']
                if len(coords) == 2:
                    r, c = coords
                    if (hasattr(new_state, 'agents') and
                            0 <= self.id < len(new_state.agents) and
                            hasattr(new_state.agents[self.id], 'colour')):
                        color = new_state.agents[self.id].colour
                        if (hasattr(new_state, 'board') and
                                hasattr(new_state.board, 'chips') and
                                0 <= r < len(new_state.board.chips) and
                                0 <= c < len(new_state.board.chips[0])):
                            new_state.board.chips[r][c] = color

            return new_state
        except:
            return state

    def custom_shallow_copy(self, state):
        """创建状态的深拷贝"""
        try:
            from copy import deepcopy
            return deepcopy(state)
        except:
            return state

    def heuristic(self, state, action):
        """启发式函数 - 使用高级评估"""
        return AdvancedEvaluator.advanced_heuristic(state, action, self.id)

    def evaluate_state(self, state, action):
        """状态评估函数 - 使用高级评估"""
        return AdvancedEvaluator.advanced_state_evaluation(state, action, self.id)

    def board_hash(self, state):
        """棋盘哈希"""
        try:
            if hasattr(state, 'board') and hasattr(state.board, 'chips'):
                return tuple(tuple(row) for row in state.board.chips)
            return id(state)
        except:
            return id(state)

    # ========== 保留原有的简单评估函数作为备选 ==========

    def center_bias(self, r, c):
        """中心偏好"""
        distance = abs(r - 4.5) + abs(c - 4.5)
        return max(0, 5 - distance) * 2

    def chain_score(self, board, r, c, color):
        """连锁得分"""
        score = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 5):
                x, y = r + dx * i, c + dy * i
                if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == color:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                x, y = r - dx * i, c - dy * i
                if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == color:
                    count += 1
                else:
                    break

            if count >= 4:
                score += 100
            elif count == 3:
                score += 30
            elif count == 2:
                score += 10
        return score

    def block_enemy_score(self, board, r, c, enemy_color):
        """阻挡敌人得分"""
        score = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            enemy_chain = 0
            for i in range(1, 5):
                x, y = r + dx * i, c + dy * i
                if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy_color:
                    enemy_chain += 1
                else:
                    break
            if enemy_chain >= 3:
                score += 50
        return score

    def hotb_score(self, board, color):
        """中心控制得分"""
        return 100 if all(board[x][y] == color for x, y in HOTB_COORDS) else 0