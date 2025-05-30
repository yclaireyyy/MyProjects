from numpy.random._common import namedtuple

from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
import heapq
import time
import itertools

MAX_THINK_TIME = 0.95
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]


TTEntry = namedtuple('TTEntry', 'depth score flag best_move')


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
        center = [(4, 4), (4, 5), (5, 4), (5, 5)]
        if (r, c) in center:
            return 20
        elif 3 <= r <= 6 and 3 <= c <= 6:
            return 10
        else:
            return max(0, 6 - (abs(r - 4.5) + abs(c - 4.5)))

    def chain_score(self, board, r, c, color):
        total_score = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            blocks = 0

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

            # 评分逻辑：考虑连子长度和封堵情况
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
            elif count == 2 and blocks == 1:
                total_score += 3

        return total_score

    def block_enemy_score(self, board, r, c, enemy_color):
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
        return 100 if all(board[x][y] == color for x, y in HOTB_COORDS) else 0

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
                        threat_score -= 50
        return threat_score

    def score_hotb_control(self, board, color):
        control = sum(1 for r, c in HOTB_COORDS if board[r][c] == color)
        return control * 25

    def score_board_mobility(self, board, color):
        count = 0
        for r in range(10):
            for c in range(10):
                if board[r][c] == '0':
                    count += 1
        return count // 5

    def board_hash(self, state):
        return tuple(tuple(row) for row in state.board.chips)
