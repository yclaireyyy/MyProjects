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
SIMULATION_LIMIT = 150  # MCTS模拟的最大次数

# 新增：增强评估常量
THREAT_WEIGHTS = {
    5: 10000,  # 已完成序列 - 最高优先级
    4: 1000,  # 4连 - 极高威胁
    3: 200,  # 3连 - 高威胁
    2: 50,  # 2连 - 中等威胁
    1: 10  # 1连 - 低威胁
}

POSITION_VALUES = [
    [1, 1, 2, 3, 4, 4, 3, 2, 1, 1],
    [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
    [2, 3, 4, 5, 6, 6, 5, 4, 3, 2],
    [3, 4, 5, 6, 7, 7, 6, 5, 4, 3],
    [4, 5, 6, 7, 8, 8, 7, 6, 5, 4],
    [4, 5, 6, 7, 8, 8, 7, 6, 5, 4],
    [3, 4, 5, 6, 7, 7, 6, 5, 4, 3],
    [2, 3, 4, 5, 6, 6, 5, 4, 3, 2],
    [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
    [1, 1, 2, 3, 4, 4, 3, 2, 1, 1]
]


class CardEvaluator:
    def __init__(self, agent):
        self.agent = agent

    def _evaluate_card(self, card, state):
        """评估卡牌在当前状态下的价值 - 增强版"""
        board = state.board.chips

        # 优先级1：双眼J - 直接最高分
        if self._is_two_eyed_jack(card):
            return 15000  # 提高J牌价值
        # 优先级2：单眼J - 次高分
        if self._is_one_eyed_jack(card):
            return 8000  # 提高J牌价值
        # 优先级3：普通卡牌 - 使用增强指数评分
        if card in COORDS:
            return self._enhanced_card_evaluation(card, state)
        return 0

    def _enhanced_card_evaluation(self, card, state):
        """增强的卡牌评估 - 考虑多重因素"""
        if card not in COORDS:
            return 0

        board = state.board.chips
        total_score = 0

        # 获取该卡牌对应的所有可能位置
        positions = COORDS[card] if isinstance(COORDS[card], list) else [COORDS[card]]

        position_scores = []
        for pos in positions:
            r, c = pos
            # 检查位置是否可用
            if not self._is_position_available(board, r, c):
                continue

            # 综合位置评分
            position_score = (
                    self._calculate_position_score(board, r, c) * 0.4 +  # 原有评分
                    self._calculate_threat_potential(board, r, c) * 0.3 +  # 威胁潜力
                    self._calculate_blocking_value(board, r, c) * 0.2 +  # 阻塞价值
                    self._calculate_strategic_value(board, r, c) * 0.1  # 战略价值
            )
            position_scores.append(position_score)

        if not position_scores:
            return 0

        # 取最高分而非平均分 - 卡牌的价值由最佳位置决定
        return max(position_scores)

    def _calculate_threat_potential(self, board, r, c):
        """计算威胁潜力 - 新增方法"""
        my_color = self.agent.my_color
        if not my_color:
            return 0

        threat_score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            # 检查这个位置能形成的最长潜在序列
            potential_length = self._calculate_potential_sequence(board, r, c, dx, dy, my_color)

            if potential_length >= 5:
                threat_score += 2000  # 能完成序列
            elif potential_length == 4:
                threat_score += 800  # 能形成4连威胁
            elif potential_length == 3:
                threat_score += 300  # 能形成3连
            elif potential_length == 2:
                threat_score += 100  # 能形成2连

        return threat_score

    def _calculate_potential_sequence(self, board, r, c, dx, dy, color):
        """计算潜在序列长度 - 考虑空位"""
        sequence_length = 1  # 当前位置
        empty_spots = 0

        # 正向检查
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color:
                    sequence_length += 1
                elif board[x][y] == 0 or board[x][y] == '0':
                    empty_spots += 1
                    if empty_spots <= 1:  # 允许一个空位
                        sequence_length += 1
                else:
                    break
            else:
                break

        # 反向检查
        empty_spots = 0
        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color:
                    sequence_length += 1
                elif board[x][y] == 0 or board[x][y] == '0':
                    empty_spots += 1
                    if empty_spots <= 1:  # 允许一个空位
                        sequence_length += 1
                else:
                    break
            else:
                break

        return min(sequence_length, 5)

    def _calculate_blocking_value(self, board, r, c):
        """计算阻塞对手的价值 - 新增方法"""
        opp_color = self.agent.opp_color
        if not opp_color:
            return 0

        blocking_score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            # 检查放置这个位置是否能阻止对手形成威胁序列
            opp_threat = self._calculate_opponent_threat_at_position(board, r, c, dx, dy, opp_color)

            if opp_threat >= 4:
                blocking_score += 1500  # 阻止对手获胜
            elif opp_threat == 3:
                blocking_score += 600  # 阻止对手形成4连威胁
            elif opp_threat == 2:
                blocking_score += 200  # 阻止对手形成3连

        return blocking_score

    def _calculate_opponent_threat_at_position(self, board, r, c, dx, dy, opp_color):
        """计算对手在特定位置的威胁程度"""
        # 模拟在该位置放置对手棋子，看能形成多长的序列
        opp_sequence = 1  # 假设对手放在这里

        # 正向检查
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == opp_color:
                opp_sequence += 1
            else:
                break

        # 反向检查
        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == opp_color:
                opp_sequence += 1
            else:
                break

        return min(opp_sequence, 5)

    def _calculate_strategic_value(self, board, r, c):
        """计算战略价值 - 新增方法"""
        strategic_score = 0

        # 1. 位置价值表
        strategic_score += POSITION_VALUES[r][c] * 10

        # 2. 连接度 - 周围己方棋子的数量
        my_color = self.agent.my_color
        if my_color:
            adjacent_count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        if board[nr][nc] == my_color:
                            adjacent_count += 1
            strategic_score += adjacent_count * 30

        # 3. 中心控制加成
        if (r, c) in HOTB_COORDS:
            strategic_score += 100

        return strategic_score

    def _calculate_position_score(self, board, r, c):
        """计算单个位置的指数评分 - 原有方法保持不变"""
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
        my_pieces = 0

        # 检查该方向前后各4个位置（共8个位置）
        for i in range(-4, 5):
            # 跳过中心位置（即将放置的位置）
            if i == 0:
                continue
            x, y = r + i * dx, c + i * dy
            # 边界检查
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == self.agent.my_color:
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
    @staticmethod
    def heuristic(state, action):
        """A*启发式函数 -评估动作的潜在价值（越低越好) - 增强版"""
        if action.get('type') != 'place' or 'coords' not in action:
            return 100  # 非放置动作或无坐标

        r, c = action['coords']
        if (r, c) in CORNERS:
            return 100  # 角落位置

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

        # 创建假设放置后的棋盘
        board_copy = [row[:] for row in board]
        board_copy[r][c] = color

        # 增强的综合评分
        score = (
                ActionEvaluator._calculate_offensive_score(board_copy, r, c, color) * 0.4 +
                ActionEvaluator._calculate_defensive_score(board, r, c, enemy) * 0.3 +
                ActionEvaluator._calculate_positional_score(r, c, board_copy, color) * 0.2 +
                ActionEvaluator._calculate_tempo_score(board, r, c, color, enemy) * 0.1
        )

        # 转换为启发式分数（越低越好）
        return max(0, 150 - score)

    @staticmethod
    def _calculate_offensive_score(board_copy, r, c, color):
        """计算进攻评分 - 增强版"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        max_sequence = 0
        for dx, dy in directions:
            count = ActionEvaluator._count_consecutive(board_copy, r, c, dx, dy, color)
            max_sequence = max(max_sequence, count)

            # 使用增强的威胁权重
            if count in THREAT_WEIGHTS:
                score += THREAT_WEIGHTS[count]

            # 额外奖励：接近获胜的序列
            if count == 4:
                # 检查是否能在下一步完成序列
                if ActionEvaluator._can_complete_sequence(board_copy, r, c, dx, dy, color):
                    score += 5000

        return score

    @staticmethod
    def _can_complete_sequence(board, r, c, dx, dy, color):
        """检查是否能在下一步完成序列"""
        # 检查序列两端是否有空位可以完成5连
        sequence_positions = [(r, c)]

        # 收集当前序列的所有位置
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == color:
                sequence_positions.append((x, y))
            else:
                break

        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == color:
                sequence_positions.insert(0, (x, y))
            else:
                break

        if len(sequence_positions) >= 4:
            # 检查两端是否有空位
            first_pos = sequence_positions[0]
            last_pos = sequence_positions[-1]

            # 检查前端
            prev_x, prev_y = first_pos[0] - dx, first_pos[1] - dy
            if (0 <= prev_x < 10 and 0 <= prev_y < 10 and
                    (board[prev_x][prev_y] == 0 or board[prev_x][prev_y] == '0')):
                return True

            # 检查后端
            next_x, next_y = last_pos[0] + dx, last_pos[1] + dy
            if (0 <= next_x < 10 and 0 <= next_y < 10 and
                    (board[next_x][next_y] == 0 or board[next_x][next_y] == '0')):
                return True

        return False

    @staticmethod
    def _calculate_defensive_score(board, r, c, enemy):
        """计算防御评分 - 增强版"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            # 检查阻断对手的威胁程度
            threat_level = ActionEvaluator._assess_blocking_threat(board, r, c, dx, dy, enemy)

            if threat_level >= 4:
                score += 2000  # 阻止对手获胜
            elif threat_level == 3:
                score += 800  # 阻止对手形成获胜威胁
            elif threat_level == 2:
                score += 300  # 阻止对手形成强威胁
            elif threat_level == 1:
                score += 100  # 阻止对手扩展

        return score

    @staticmethod
    def _assess_blocking_threat(board, r, c, dx, dy, enemy):
        """评估在此位置阻断对手的威胁等级"""
        # 检查如果对手在此位置放棋子会形成多长的序列
        enemy_sequence = 1

        # 正向检查
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
                enemy_sequence += 1
            else:
                break

        # 反向检查
        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
                enemy_sequence += 1
            else:
                break

        return min(enemy_sequence, 5)

    @staticmethod
    def _calculate_positional_score(r, c, board_copy, color):
        """计算位置评分 - 增强版"""
        score = 0

        # 1. 基础位置价值
        score += POSITION_VALUES[r][c] * 15

        # 2. 中心控制
        distance = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 10 - distance) * 3

        # 3. 中心热点控制
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if board_copy[x][y] == color)
        score += hotb_controlled * 25

        # 4. 连接性评分 - 与己方棋子的连接度
        connectivity = 0
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    if board_copy[nr][nc] == color:
                        # 距离越近，连接价值越高
                        connection_value = 3 - (abs(dr) + abs(dc)) // 2
                        connectivity += max(1, connection_value)
        score += connectivity * 5

        return score

    @staticmethod
    def _calculate_tempo_score(board, r, c, color, enemy):
        """计算节奏评分 - 新增：评估紧迫性"""
        score = 0

        # 1. 己方接近获胜的紧迫性
        my_max_threat = ActionEvaluator._find_max_threat_level(board, color)
        if my_max_threat >= 3:
            score += (my_max_threat - 2) * 100  # 越接近获胜越重要

        # 2. 对手威胁的紧迫性
        enemy_max_threat = ActionEvaluator._find_max_threat_level(board, enemy)
        if enemy_max_threat >= 3:
            score += (enemy_max_threat - 2) * 150  # 防守比进攻更紧迫

        # 3. 双向威胁奖励
        if my_max_threat >= 3 and enemy_max_threat >= 3:
            score += 200  # 关键对决位置

        return score

    @staticmethod
    def _find_max_threat_level(board, color):
        """找到棋盘上某颜色的最大威胁等级"""
        max_threat = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for i in range(10):
            for j in range(10):
                if board[i][j] == color:
                    for dx, dy in directions:
                        threat = ActionEvaluator._count_consecutive(board, i, j, dx, dy, color)
                        max_threat = max(max_threat, threat)

        return max_threat

    @staticmethod
    def _calculate_action_score(board, r, c, color, enemy):
        """计算动作分数 - 保持原有接口"""
        score = 0

        # 创建假设棋盘
        board_copy = [row[:] for row in board]
        board_copy[r][c] = color

        # 中心偏好
        distance = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 5 - distance) * 2

        # 连续链评分
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = ActionEvaluator._count_consecutive(board_copy, r, c, dx, dy, color)
            if count >= 5:
                score += 200
            elif count == 4:
                score += 100
            elif count == 3:
                score += 30
            elif count == 2:
                score += 10

        # 防守评分
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            enemy_threat = ActionEvaluator._count_enemy_threat(board, r, c, dx, dy, enemy)
            if enemy_threat >= 3:
                score += 50

        # 中心控制
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if board_copy[x][y] == color)
        score += hotb_controlled * 15

        return score

    @staticmethod
    def _count_consecutive(board, x, y, dx, dy, color):
        """计算从(x,y)出发，在方向(dx,dy)上颜色为color的最长连续序列"""
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
    def _count_enemy_threat(board, r, c, dx, dy, enemy):
        """计算敌方威胁"""
        enemy_chain = 0
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
                enemy_chain += 1
            else:
                break
        return enemy_chain


class StateEvaluator:
    @staticmethod
    def evaluate(state, last_action=None):
        """评估游戏状态的价值 - 增强版"""
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

        # 增强的多维度评分
        position_score = StateEvaluator._enhanced_position_score(board, my_color, opp_color)
        sequence_score = StateEvaluator._enhanced_sequence_score(board, my_color)
        defense_score = StateEvaluator._enhanced_defense_score(board, opp_color)
        control_score = StateEvaluator._enhanced_control_score(board, my_color, opp_color)
        tempo_score = StateEvaluator._calculate_tempo_advantage(board, my_color, opp_color)

        # 动态权重 - 根据游戏阶段调整
        game_phase = StateEvaluator._determine_game_phase(board)
        if game_phase == "opening":
            weights = [0.3, 0.2, 0.1, 0.3, 0.1]  # 重视位置和控制
        elif game_phase == "middle":
            weights = [0.2, 0.3, 0.3, 0.1, 0.1]  # 重视序列和防御
        else:  # endgame
            weights = [0.1, 0.4, 0.4, 0.05, 0.05]  # 重视序列和防御

        # 5. 综合评分
        total_score = (
                position_score * weights[0] +
                sequence_score * weights[1] +
                defense_score * weights[2] +
                control_score * weights[3] +
                tempo_score * weights[4]
        )

        # 归一化到[-1, 1]区间
        return max(-1, min(1, total_score / 300))

    @staticmethod
    def _determine_game_phase(board):
        """判断游戏阶段"""
        total_pieces = sum(1 for i in range(10) for j in range(10)
                           if board[i][j] not in [0, '0'])

        if total_pieces < 20:
            return "opening"
        elif total_pieces < 60:
            return "middle"
        else:
            return "endgame"

    @staticmethod
    def _enhanced_position_score(board, my_color, opp_color):
        """增强的位置评分"""
        position_score = 0

        for i in range(10):
            for j in range(10):
                cell_value = board[i][j]
                base_value = POSITION_VALUES[i][j]

                if cell_value == my_color:
                    # 己方棋子的价值
                    multiplier = 1.0

                    # 连接性奖励
                    connections = StateEvaluator._count_connections(board, i, j, my_color)
                    multiplier += connections * 0.1

                    # 中心位置额外奖励
                    if (i, j) in HOTB_COORDS:
                        multiplier += 0.5

                    position_score += base_value * multiplier

                elif cell_value == opp_color:
                    # 对手棋子的负面影响
                    multiplier = 1.0
                    connections = StateEvaluator._count_connections(board, i, j, opp_color)
                    multiplier += connections * 0.1

                    if (i, j) in HOTB_COORDS:
                        multiplier += 0.5

                    position_score -= base_value * multiplier

        return position_score

    @staticmethod
    def _count_connections(board, r, c, color):
        """计算位置的连接数"""
        connections = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    if board[nr][nc] == color:
                        connections += 1
        return connections

    @staticmethod
    def _enhanced_sequence_score(board, my_color):
        """增强的序列评分"""
        sequence_score = 0
        threat_levels = {}

        # 收集所有威胁等级
        for i in range(10):
            for j in range(10):
                if board[i][j] == my_color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = ActionEvaluator._count_consecutive(board, i, j, dx, dy, my_color)
                        if count > 1:
                            threat_levels[count] = threat_levels.get(count, 0) + 1

        # 根据威胁等级计算分数
        for level, count in threat_levels.items():
            if level >= 5:
                sequence_score += 10000  # 已获胜
            elif level == 4:
                sequence_score += 1000 * count  # 4连威胁
            elif level == 3:
                sequence_score += 100 * count  # 3连威胁
            elif level == 2:
                sequence_score += 20 * count  # 2连威胁

        return sequence_score

    @staticmethod
    def _enhanced_defense_score(board, opp_color):
        """增强的防御评分"""
        defense_score = 0
        opp_threats = {}

        # 收集对手威胁
        for i in range(10):
            for j in range(10):
                if board[i][j] == opp_color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = ActionEvaluator._count_consecutive(board, i, j, dx, dy, opp_color)
                        if count > 1:
                            opp_threats[count] = opp_threats.get(count, 0) + 1

        # 根据对手威胁计算防御分数
        for level, count in opp_threats.items():
            if level >= 4:
                defense_score -= 2000 * count  # 对手4连 - 极度危险
            elif level == 3:
                defense_score -= 400 * count  # 对手3连 - 高度危险
            elif level == 2:
                defense_score -= 50 * count  # 对手2连 - 中度威胁

        return defense_score

    @staticmethod
    def _enhanced_control_score(board, my_color, opp_color):
        """增强的控制评分"""
        control_score = 0

        # 1. 中心热点控制
        my_hotb = sum(1 for x, y in HOTB_COORDS if board[x][y] == my_color)
        opp_hotb = sum(1 for x, y in HOTB_COORDS if board[x][y] == opp_color)
        control_score += (my_hotb - opp_hotb) * 20

        # 2. 关键空位控制
        key_positions = StateEvaluator._identify_key_positions(board, my_color, opp_color)
        control_score += len(key_positions) * 10

        # 3. 边缘控制
        my_edge = StateEvaluator._count_edge_control(board, my_color)
        opp_edge = StateEvaluator._count_edge_control(board, opp_color)
        control_score += (my_edge - opp_edge) * 2

        return control_score

    @staticmethod
    def _identify_key_positions(board, my_color, opp_color):
        """识别关键空位"""
        key_positions = set()
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for i in range(10):
            for j in range(10):
                if board[i][j] in [0, '0']:
                    # 检查这个空位是否对任一方有战略价值
                    importance = 0

                    for dx, dy in directions:
                        # 检查能否帮助己方形成威胁
                        my_potential = StateEvaluator._calc_position_potential(
                            board, i, j, dx, dy, my_color)
                        opp_potential = StateEvaluator._calc_position_potential(
                            board, i, j, dx, dy, opp_color)

                        if my_potential >= 3 or opp_potential >= 3:
                            importance += 1

                    if importance >= 2:  # 多个方向都有价值
                        key_positions.add((i, j))

        return key_positions

    @staticmethod
    def _calc_position_potential(board, r, c, dx, dy, color):
        """计算位置在特定方向的潜力"""
        potential = 1  # 当前位置

        # 正向检查
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color:
                    potential += 1
                elif board[x][y] not in [0, '0']:
                    break
            else:
                break

        # 反向检查
        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color:
                    potential += 1
                elif board[x][y] not in [0, '0']:
                    break
            else:
                break

        return min(potential, 5)

    @staticmethod
    def _count_edge_control(board, color):
        """计算边缘控制"""
        edge_count = 0
        for i in range(10):
            if board[i][0] == color:
                edge_count += 1
            if board[i][9] == color:
                edge_count += 1
        for j in range(1, 9):  # 避免重复计算角落
            if board[0][j] == color:
                edge_count += 1
            if board[9][j] == color:
                edge_count += 1
        return edge_count

    @staticmethod
    def _calculate_tempo_advantage(board, my_color, opp_color):
        """计算节奏优势"""
        my_max_threat = ActionEvaluator._find_max_threat_level(board, my_color)
        opp_max_threat = ActionEvaluator._find_max_threat_level(board, opp_color)

        tempo_score = 0

        # 威胁等级差异
        threat_diff = my_max_threat - opp_max_threat
        tempo_score += threat_diff * 50

        # 主动权评估
        if my_max_threat >= 4:
            tempo_score += 200  # 我方有获胜威胁
        if opp_max_threat >= 4:
            tempo_score -= 250  # 对手有获胜威胁，更危险

        return tempo_score

    @staticmethod
    def _calculate_sequence_score(board, color):
        """计算序列得分 - 保持原有接口"""
        sequence_score = 0
        for i in range(10):
            for j in range(10):
                if board[i][j] == color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = ActionEvaluator._count_consecutive(board, i, j, dx, dy, color)
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
        """计算防御得分 - 保持原有接口"""
        defense_score = 0
        for i in range(10):
            for j in range(10):
                if board[i][j] == opp_color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = ActionEvaluator._count_consecutive(board, i, j, dx, dy, opp_color)
                        if count >= 4:
                            defense_score -= 50
                        elif count == 3:
                            defense_score -= 10
        return defense_score


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
    The search tree node integrating MCTS and A*
    """

    def __init__(self, state, parent=None, action=None):
        # 状态表示
        try:
            self.state = state.clone()
        except:
            self.state = copy.deepcopy(state)
        # 节点关系
        self.parent = parent
        self.children = []
        self.action = action
        # MCTS统计数据
        self.visits = 0
        self.value = 0.0
        # 动作管理（延迟初始化）
        self.untried_actions = None

    def _clone_state(self, state):
        try:
            return state.clone()
        except:
            return copy.deepcopy(state)

    def get_untried_actions(self):
        """获取未尝试的动作，使用启发式排序"""
        if self.untried_actions is None:
            # 初始化未尝试动作列表
            if hasattr(self.state, 'available_actions'):
                self.untried_actions = list(self.state.available_actions)
            else:
                self.untried_actions = []
            # 使用A*启发式进行排序（越小越优先）
            self.untried_actions.sort(key=lambda a: ActionEvaluator.heuristic(self.state, a))
        return self.untried_actions

    def is_fully_expanded(self):
        """检查节点是否已完全展开"""
        return len(self.get_untried_actions()) == 0

    def select_child(self):
        """使用UCB公式选择最有希望的子节点"""
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            # UCB计算
            if child.visits == 0:
                score = float('inf')
            else:
                # 结合A*启发式的UCB计算
                exploitation = child.value / child.visits
                exploration = EXPLORATION_WEIGHT * math.sqrt(2 * math.log(self.visits) / child.visits)
                # 启发式调整
                if child.action:
                    heuristic_factor = 1.0 / (1.0 + ActionEvaluator.heuristic(self.state, child.action) / 100)
                else:
                    heuristic_factor = 1.0

                score = exploitation + exploration * heuristic_factor

            # 更新最佳节点
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, agent):
        """扩展一个新子节点，使用A*启发式选择最有前途的动作"""
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
        return self.get_remaining_time() < 0.3


class myAgent(Agent):
    """智能体 myAgent - 增强对抗性版本"""

    def __init__(self, _id):
        """初始化Agent"""
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)  # 2人游戏
        self.counter = itertools.count()  # 用于A*搜索的唯一标识符

        self.card_evaluator = CardEvaluator(self)
        self.time_manager = TimeManager()

        # 玩家颜色初始化
        self.my_color = None
        self.opp_color = None

        # 搜索参数
        self.simulation_depth = 5  # 模拟深度
        self.candidate_limit = 10  # A*筛选的候选动作数

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
        """主决策函数 - 融合A*和MCTS"""
        self.time_manager.start_timing()
        self._initialize_colors(game_state)

        if self._is_card_selection(actions):
            return self._select_strategic_card(actions, game_state)

        # A*筛选候选动作
        candidates = self._a_star_filter(actions, game_state)

        # 时间检查
        if self.time_manager.should_use_quick_mode():
            return candidates[0] if candidates else random.choice(actions)
        # MCTS深度搜索
        try:
            return self._mcts_search(candidates, game_state)
        except:
            return candidates[0] if candidates else random.choice(actions)

    def _a_star_filter(self, actions, game_state):
        """使用A*算法筛选最有前途的动作"""
        # 排除角落位置
        valid_actions = [a for a in actions if 'coords' not in a or a['coords'] not in CORNERS]
        if not valid_actions:
            return actions[:1]  # 如果没有有效动作，返回第一个动作

        # 评估每个动作
        scored_actions = []
        for action in valid_actions:
            score = ActionEvaluator.heuristic(game_state, action)
            scored_actions.append((action, score))

        # 按评分排序（升序，越小越好）
        scored_actions.sort(key=lambda x: x[1])

        # 返回前N个候选动作
        scored_actions.sort(key=lambda x: x[1])
        return [a for a, _ in scored_actions[:self.candidate_limit]]

    def _mcts_search(self, candidate_actions, game_state):
        """使用MCTS分析候选动作"""
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
            value = self._a_star_guided_simulate(node.state)

            # 4. 回溯阶段
            while node:
                node.update(value)
                node = node.parent

        # 选择最佳动作（访问次数最多的子节点）
        if not root.children:
            return candidate_actions[0] if candidate_actions else None

        # 选择访问次数最多的子节点
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _a_star_guided_simulate(self, state):
        """A*引导的MCTS模拟"""
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

            # 80%概率使用启发式，20%随机选择
            if random.random() < 0.8:
                # 使用启发式选择动作
                scored_actions = [(a, ActionEvaluator.heuristic(state_copy, a)) for a in actions]
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
        """快速模拟执行动作"""
        new_state = state.copy() if hasattr(state, "copy") else self.custom_shallow_copy(state)

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
        """创建游戏状态的深拷贝"""
        from copy import deepcopy
        return deepcopy(state)