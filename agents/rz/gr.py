from Sequence.sequence_model import SequenceGameRule
import random
from copy import deepcopy

EMPTY = '_'  # 确保与你的项目中定义一致
JOKER = 'X'  # 四个角落万能格

def simulate_action_on_board(chips, action, player_colour):
    """
    模拟某个动作（place 或 remove）在棋盘上的影响，仅对 chips 做出副本修改。
    """
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
                # 模拟这个动作后的状态
                next_chips = simulate_action_on_board(chips, action, clr)
                # 评估局面
                score = self.evaluate(next_chips,clr, sclr,opp,opp_s)
                if score > best_score:
                    best_score = score
                    best_action = action
        except Exception as e:
            print(e)
            return random.choice(actions)
        return best_action

    def evaluate(self, chips, clr, sclr, opp, opp_s):

        # 四角视为己方通配符
        jokers = {(0, 0), (0, 9), (9, 0), (9, 9)}

        def is_ours(r, c):
            return (r, c) in jokers or chips[r][c] in [clr, sclr]

        def is_opp(r, c):
            return chips[r][c] in [opp, opp_s]

        # 1. Heart 四格是否占满
        heart = [(4, 4), (4, 5), (5, 4), (5, 5)]
        if all(is_ours(r, c) for r, c in heart):
            return 9999  # 视为胜利

        heart_score = sum(1 for r, c in heart if is_ours(r, c))

        # 2. Sequence 潜力评估（考虑两端开放）
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
                        continue  # 被敌方阻挡

                    count = sum(1 for v in values if v in [clr, sclr])
                    if count >= 2:
                        # 检查两端是否开放
                        before = (r - dr, c - dc)
                        after = (r + 5 * dr, c + 5 * dc)
                        blocked_before = not (0 <= before[0] < 10 and 0 <= before[1] < 10) or is_opp(*before)
                        blocked_after = not (0 <= after[0] < 10 and 0 <= after[1] < 10) or is_opp(*after)

                        if blocked_before and blocked_after:
                            continue  # 死棋，不加分

                        seq_score += count ** 2

        return heart_score * 20 + seq_score
