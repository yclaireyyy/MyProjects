import random
from copy import deepcopy

EMPTY = '_'
JOKER = 'X'  # wildcard at corners

def simulate_action_on_board(chips, action, player_colour):
    new_chips = deepcopy(chips)
    r, c = action['coords']
    act_type = action['type']
    if act_type == 'place':
        new_chips[r][c] = player_colour
    elif act_type == 'remove':
        new_chips[r][c] = EMPTY
    else:
        raise ValueError(f"Unsupported action type: {act_type}")
    return new_chips

class myAgent:
    def __init__(self, _id):
        self.id = _id

    def SelectAction(self, actions, game_state):
        best_score = float('-inf')
        best_action = None
        chips = game_state.board.chips
        player = game_state.agents[self.id]
        clr = player.colour
        sclr = player.seq_colour
        opp = player.opp_colour
        opp_s = player.opp_seq_colour
        try:
            for action in actions:
                next_chips = simulate_action_on_board(chips, action, clr)
                my_score = self.evaluate(next_chips, clr, sclr, opp, opp_s)
                opp_score = self.evaluate(next_chips, opp, opp_s, clr, sclr)
                score = my_score - opp_score
                if score > best_score:
                    best_score = score
                    best_action = action
        except Exception as e:
            print(e)
            return random.choice(actions)
        return best_action

    def evaluate(self, chips, clr, sclr, opp, opp_s):
        # 权重矩阵
        position_weights = [
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 3],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 3, 3, 3, 3, 3, 3, 2, 1],
            [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
            [1, 2, 3, 3, 3, 3, 3, 3, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 3]
        ]

        jokers = {(0, 0), (0, 9), (9, 0), (9, 9)}
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        def is_ours(r, c):
            return (r, c) in jokers or chips[r][c] in [clr, sclr]

        def is_opp(r, c):
            return chips[r][c] in [opp, opp_s]

        def valid(x, y):
            return 0 <= x < 10 and 0 <= y < 10

        # 1. Heart 判断
        heart = [(4, 4), (4, 5), (5, 4), (5, 5)]
        if all(is_ours(r, c) for r, c in heart):
            return 9999

        heart_score = sum(1 for r, c in heart if is_ours(r, c))

        # 2. 已完成 sequence 数量
        seq_count = 0
        for r in range(10):
            for c in range(10):
                for dr, dc in directions:
                    cells = [(r + i * dr, c + i * dc) for i in range(5)]
                    if all(valid(x, y) and is_ours(x, y) for x, y in cells):
                        seq_count += 1

        if seq_count >= 2:
            return 9999
        elif seq_count == 1:
            seq_score = 5000
        else:
            seq_score = 0

        # 3. 潜在 sequence（开放式）打分
        potential_seq = 0
        for r in range(10):
            for c in range(10):
                for dr, dc in directions:
                    cells = [(r + i * dr, c + i * dc) for i in range(5)]
                    if not all(valid(x, y) for x, y in cells):
                        continue
                    values = [chips[x][y] if (x, y) not in jokers else clr for x, y in cells]
                    if any(v in [opp, opp_s] for v in values):
                        continue
                    count = sum(1 for v in values if v in [clr, sclr])
                    if count >= 2:
                        # 检查两端是否开放
                        before = (r - dr, c - dc)
                        after = (r + 5 * dr, c + 5 * dc)
                        blocked_before = not valid(*before) or is_opp(*before)
                        blocked_after = not valid(*after) or is_opp(*after)
                        if blocked_before and blocked_after:
                            continue
                        potential_seq += count ** 2

        # 4. 位置权重
        pos_score = 0
        for r in range(10):
            for c in range(10):
                if is_ours(r, c):
                    pos_score += position_weights[r][c]

        return seq_score + potential_seq + heart_score * 30 + pos_score

