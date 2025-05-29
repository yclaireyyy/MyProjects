from numpy import inf
from numpy.random._common import namedtuple

from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
import heapq
import time
import itertools

MAX_THINK_TIME = 0.95
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]


TTEntry = namedtuple('TTEntry', 'depth score flag best_move')
# flag: 'EXACT', 'LOWER', 'UPPER'


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)
        self.counter = itertools.count()

    def SelectAction(self, actions, game_state):
        self.start_time = time.time()
        return self.a_star(game_state, actions)

    def a_star(self, initial_state, candidate_moves):
        pending = []
        seen_states = set()
        best_sequence = []
        top_reward = float('-inf')

        for move in candidate_moves:
            g = 1
            h = self.heuristic(initial_state, move)
            f = g + h
            heapq.heappush(pending, (f, next(self.counter), g, h, self.fast_simulate(initial_state, move), [move]))

        while pending and (time.time() - self.start_time < MAX_THINK_TIME):
            f, _, g, h, current_state, move_history = heapq.heappop(pending)
            last_move = move_history[-1]

            state_signature = (
                self.board_hash(current_state),
                last_move['play_card'],
                tuple(current_state.agents[self.id].hand)
            )
            if state_signature in seen_states:
                continue
            seen_states.add(state_signature)

            reward = self.evaluate_state(current_state, last_move)
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history

            next_steps = self.rule.getLegalActions(current_state, self.id)
            next_steps.sort(key=lambda act: self.heuristic(current_state, act))

            for next_move in next_steps[:5]:
                next_g = g + 1
                next_h = self.heuristic(current_state, next_move)
                heapq.heappush(pending, (
                    next_g + next_h, next(self.counter),
                    next_g, next_h,
                    self.fast_simulate(current_state, next_move), move_history + [next_move]
                ))

        return best_sequence[0] if best_sequence else candidate_moves[0]

    def fast_simulate(self, state, action):
        new_state = state.copy() if hasattr(state, "copy") else self.custom_shallow_copy(state)
        agent = new_state.agents[self.id]

        if action['type'] == 'place' and 'coords' in action:
            r, c = action['coords']
            new_state.board.chips[r][c] = agent.colour

        return new_state

    def custom_shallow_copy(self, state):
        from copy import deepcopy
        return deepcopy(state)

    def heuristic(self, state, action):
        if action.get('type') != 'place' or not action.get('coords'):
            return 100

        r, c = action['coords']
        board = [row[:] for row in state.board.chips]
        me = state.agents[self.id]
        color = me.colour
        enemy = 'r' if color == 'b' else 'b'

        board[r][c] = color
        score = 0
        score += self.center_bias(r, c)
        score += self.chain_score(board, r, c, color)
        score += self.block_enemy_score(board, r, c, enemy)
        score += self.hotb_score(board, color)

        return 100 - score

    def center_bias(self, r, c):
        # 使用曼哈顿距离计算，更平滑的衰减
        center_r, center_c = 4.5, 4.5
        distance = abs(r - center_r) + abs(c - center_c)

        # 根据距离给出分数，中心最高
        if distance <= 1:  # 最中心的4个格子
            return 20
        elif distance <= 2.5:  # 次中心区域
            return 15
        elif distance <= 4:  # 中等区域
            return 10
        else:  # 边缘区域
            return max(0, 8 - distance)

    def chain_score(self, board, r, c, color):
        total_score = 0
        threat_count = 0  # 记录形成威胁的数量

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            blocks = 0
            spaces = []  # 记录空位位置，用于判断是否能延伸

            # 正向检查
            for i in range(1, 5):
                x, y = r + dx * i, c + dy * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if board[x][y] == color:
                        count += 1
                    elif board[x][y] == '0':
                        spaces.append((x, y))
                        if len(spaces) > 1:  # 超过1个空位就停止
                            break
                    else:
                        blocks += 1
                        break
                else:
                    blocks += 1
                    break

            # 反向检查（类似逻辑）
            # ...

            # 更细致的评分
            if count >= 5:
                return 10000  # 直接返回最高分
            elif count == 4:
                if blocks == 0:
                    total_score += 1000  # 活四
                    threat_count += 1
                elif blocks == 1:
                    total_score += 200  # 冲四
            elif count == 3:
                if blocks == 0:
                    total_score += 100  # 活三
                    threat_count += 1
                elif blocks == 1 and len(spaces) == 1:
                    total_score += 50  # 跳三（有潜力的三）
            elif count == 2:
                if blocks == 0:
                    total_score += 20  # 活二
                elif blocks == 1:
                    total_score += 5  # 死二

        # 多重威胁加成
        if threat_count >= 2:
            total_score *= 1.5

        return int(total_score)

    def block_enemy_score(self, board, r, c, enemy_color):
        max_threat = 0

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # 检查两个方向
            forward_count = 0
            backward_count = 0
            forward_space = 0
            backward_space = 0

            # 正向
            for i in range(1, 5):
                x, y = r + dx * i, c + dy * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if board[x][y] == enemy_color:
                        forward_count += 1
                    elif board[x][y] == '0':
                        forward_space += 1
                        if forward_space > 1:
                            break
                    else:
                        break
                else:
                    break

            # 反向（类似逻辑）
            # ...

            total_count = forward_count + backward_count + 1  # +1是假设敌人下在(r,c)

            # 根据威胁程度评分
            if total_count >= 4:
                max_threat = max(max_threat, 500)  # 必须阻挡
            elif total_count == 3 and (forward_space > 0 or backward_space > 0):
                max_threat = max(max_threat, 200)  # 高威胁
            elif total_count == 3:
                max_threat = max(max_threat, 100)  # 中等威胁
            elif total_count == 2:
                max_threat = max(max_threat, 30)  # 低威胁000

        return max_threat


    def hotb_score(self, board, color):
        occupied = sum(1 for x, y in HOTB_COORDS if board[x][y] == color)
        enemy_occupied = sum(1 for x, y in HOTB_COORDS if board[x][y] != '0' and board[x][y] != color)

        if occupied == len(HOTB_COORDS):
            return 200  # 完全占领
        elif occupied > enemy_occupied:
            return 50 + occupied * 20  # 部分占领加成
        else:
            return occupied * 10  # 基础分数

    def evaluate_position(self, board, r, c, color, enemy_color):
        # 基础位置分
        score = self.center_bias(r, c)

        # 进攻分数
        attack_score = self.chain_score(board, r, c, color)

        # 防守分数
        defense_score = self.block_enemy_score(board, r, c, enemy_color)

        # HOTB分数
        hotb = self.hotb_score(board, color)

        # 综合评分，防守优先级略高
        return score + attack_score + defense_score * 1.2 + hotb

    def evaluate_state(self, state, action):
        agent = state.agents[self.id]
        board = [row[:] for row in state.board.chips]
        if action.get('coords'):
            r, c = action['coords']
            board[r][c] = agent.colour
        return self.evaluate_board(board, agent)

    def evaluate_board(self, board, agent):
        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        score = 0
        score += self.score_friendly_chain(board, my_color)
        score += self.score_enemy_threat(board, enemy_color)
        score += self.score_hotb_control(board, my_color)
        # score += self.score_board_mobility(board, my_color)  # 可选

        return score

    def score_friendly_chain(self, board, color):
        max_chain = 0
        for r in range(10):
            for c in range(10):
                if board[r][c] != color:
                    continue
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 1
                    for i in range(1, 5):
                        x, y = r + dx * i, c + dy * i
                        if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == color:
                            count += 1
                        else:
                            break
                    max_chain = max(max_chain, count)
        if max_chain >= 4:
            return 100
        elif max_chain == 3:
            return 30
        elif max_chain == 2:
            return 10
        return 0

    def score_enemy_threat(self, board, enemy_color):
        threat_score = 0
        for r in range(10):
            for c in range(10):
                if board[r][c] != enemy_color:
                    continue
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 1
                    for i in range(1, 5):
                        x, y = r + dx * i, c + dy * i
                        if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy_color:
                            count += 1
                        else:
                            break
                    if count >= 3:
                        threat_score -= 50  # 威胁越大，分越低（惩罚）
        return threat_score

    def score_hotb_control(self, board, color):
        control = sum(1 for r, c in HOTB_COORDS if board[r][c] == color)
        return control * 25  # 满分 100

    # 可选项：根据当前颜色控制的空位数量给加分
    def score_board_mobility(self, board, color):
        count = 0
        for r in range(10):
            for c in range(10):
                if board[r][c] == '0':  # 假设空位是 0 字符
                    count += 1
        return count // 5

    def board_hash(self, state):
        return tuple(tuple(row) for row in state.board.chips)
