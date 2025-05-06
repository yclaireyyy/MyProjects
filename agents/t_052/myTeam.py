import time
from copy import deepcopy

from agents.t_052.example_bfs import THINKTIME

EMPTY = '_'
JOKER = 'X'
THINKTIME = 0.98

def simulate_action_on_board(chips, action, player_colour):
    new_chips = deepcopy(chips)
    r, c = action['coords']
    act_type = action['type']
    if act_type == 'place':
        new_chips[r][c] = player_colour
    elif act_type == 'remove':
        new_chips[r][c] = EMPTY
    return new_chips

def deduplicate_actions(actions):
    seen = set()
    unique_actions = []
    for action in actions:
        # 将关键字段组成元组用于哈希
        key = (
            action.get("type"),
            tuple(action.get("coords")) if action.get("coords") else None,
            action.get("play_card"),
            action.get("draft_card")
        )
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)
    return unique_actions


class myAgent:
    def __init__(self, _id):
        self.id = _id
        self.op_last = 0
        self.guesses = []

        self.card_mapping = {
            '2c': [(1, 4), (3, 6)], '2d': [(2, 2), (5, 9)], '2h': [(5, 4), (8, 7)], '2s': [(0, 1), (8, 6)],
            '3c': [(1, 3), (3, 5)], '3d': [(2, 3), (6, 9)], '3h': [(5, 5), (8, 8)], '3s': [(0, 2), (8, 5)],
            '4c': [(1, 2), (3, 4)], '4d': [(2, 4), (7, 9)], '4h': [(4, 5), (7, 8)], '4s': [(0, 3), (8, 4)],
            '5c': [(1, 1), (3, 3)], '5d': [(2, 5), (8, 9)], '5h': [(4, 4), (6, 8)], '5s': [(0, 4), (8, 3)],
            '6c': [(1, 0), (3, 2)], '6d': [(2, 6), (9, 8)], '6h': [(4, 3), (5, 8)], '6s': [(0, 5), (8, 2)],
            '7c': [(2, 0), (4, 2)], '7d': [(2, 7), (9, 7)], '7h': [(4, 8), (5, 3)], '7s': [(0, 6), (8, 1)],
            '8c': [(3, 0), (5, 2)], '8d': [(3, 7), (9, 6)], '8h': [(3, 8), (6, 3)], '8s': [(0, 7), (7, 1)],
            '9c': [(4, 0), (6, 2)], '9d': [(4, 7), (9, 5)], '9h': [(2, 8), (6, 4)], '9s': [(0, 8), (6, 1)],
            'ac': [(7, 5), (8, 0)], 'ad': [(7, 6), (9, 1)], 'ah': [(1, 5), (4, 6)], 'as': [(2, 1), (4, 9)],
            'kc': [(7, 0), (7, 4)], 'kd': [(7, 7), (9, 2)], 'kh': [(1, 6), (5, 6)], 'ks': [(3, 1), (3, 9)],
            'qc': [(6, 0), (7, 3)], 'qd': [(6, 7), (9, 3)], 'qh': [(1, 7), (6, 6)], 'qs': [(2, 9), (4, 1)],
            'tc': [(5, 0), (7, 2)], 'td': [(5, 7), (9, 4)], 'th': [(1, 8), (6, 5)], 'ts': [(1, 9), (5, 1)]
        }

    def SelectAction(self, actions, game_state):
        start_time = time.time()
        # actions = deduplicate_actions(actions)
        chips = game_state.board.chips
        player = game_state.agents[self.id]
        clr, sclr = player.colour, player.seq_colour
        opp, opp_s = player.opp_colour, player.opp_seq_colour

        best_score = float('-inf')
        best_action = None
        for action in actions:
            if time.time()-start_time > THINKTIME:
                break
            score = self.evaluate_action_value(action, chips, clr, sclr, opp, opp_s)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def evaluate_action_value(self, action, chips, clr, sclr, opp, opp_s):
        draft_card = action.get("draft_card")

        if action.get("type") == "trade":
            # 不模拟落子，直接评估未来潜力
            next_chips = chips
            delta_score = 0
        else:
            # 模拟落子
            next_chips = simulate_action_on_board(chips, action, clr)
            my_score = self.evaluate(next_chips, clr, sclr, opp, opp_s)
            opp_score = self.evaluate(next_chips, opp, opp_s, clr, sclr)
            delta_score = my_score - opp_score
            if my_score >= 9999:
                return float('inf')

        # 评估未来可能
        future_potential = 0
        if draft_card:
            valid = self.get_valid_positions(draft_card, next_chips, opp)
            future_potential = len(valid)

        return delta_score + 1.5 * future_potential

    def evaluate(self, chips, clr, sclr, opp, opp_s):
        jokers = {(0, 0), (0, 9), (9, 0), (9, 9)}

        def is_ours(r, c):
            return (r, c) in jokers or chips[r][c] in [clr, sclr]

        def is_opp(r, c):
            return chips[r][c] in [opp, opp_s]

        heart = [(4, 4), (4, 5), (5, 4), (5, 5)]
        if all(is_ours(r, c) for r, c in heart):
            return 9999
        heart_score = sum(1 for r, c in heart if is_ours(r, c))

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        seq_score = 0
        for r in range(10):
            for c in range(10):
                for dr, dc in directions:
                    cells = [(r + i * dr, c + i * dc) for i in range(5)]
                    if not all(0 <= x < 10 and 0 <= y < 10 for x, y in cells):
                        continue
                    values = [chips[x][y] if (x, y) not in jokers else clr for x, y in cells]
                    if any(v in [opp, opp_s] for v in values):
                        continue
                    count = sum(1 for v in values if v in [clr, sclr])
                    if count >= 2:
                        before = (r - dr, c - dc)
                        after = (r + 5 * dr, c + 5 * dc)
                        blocked_before = not (0 <= before[0] < 10 and 0 <= before[1] < 10) or is_opp(*before)
                        blocked_after = not (0 <= after[0] < 10 and 0 <= after[1] < 10) or is_opp(*after)
                        if blocked_before and blocked_after:
                            continue
                        seq_score += count ** 2
        return heart_score * 30 + seq_score

    def get_valid_positions(self, card, chips, opp_colour=None):
        if card in ['jc', 'jd']:
            return [(r, c) for r in range(10) for c in range(10) if chips[r][c] == EMPTY]
        elif card in ['js', 'jh']:
            return [(r, c) for r in range(10) for c in range(10) if chips[r][c] == opp_colour]
        else:
            return [pos for pos in self.card_mapping.get(card, []) if chips[pos[0]][pos[1]] == EMPTY]
