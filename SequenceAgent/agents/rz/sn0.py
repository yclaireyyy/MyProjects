from typing import List, Dict, Optional, Tuple
from Sequence.sequence_utils import *
from Sequence.sequence_model import BOARD, COORDS


class GameState:
    def __init__(
            self,
            my_hand: List[str],
            op_hand: List[str],
            draft: List[str],
            deck: List[str],
            chips: List[List[int]],
            turn: int,
            trade: bool,
            player_id: int,
            last_action: Optional[dict] = None
    ):
        self.my_hand = my_hand[:]  # 当前玩家手牌
        self.op_hand = op_hand[:]  # 对手手牌（模拟用）
        self.draft = draft[:]  # 公共牌池（5 张）
        self.deck = deck[:]  # 剩余未抽牌堆（不可见，但构造后固定）
        self.chips = [row[:] for row in chips]  # 棋盘，二维 list
        self.turn = turn  # 当前轮到的玩家（0 或 1）
        self.trade = trade  # 当前玩家是否 trade 过
        self.player_id = player_id  # 当前 perspective 中我是谁
        self.last_action = last_action  # 上一个动作记录（可选）

    def clone(self) -> "GameState":
        return GameState(
            my_hand=self.my_hand[:],
            op_hand=self.op_hand[:],
            draft=self.draft[:],
            deck=self.deck[:],
            chips=[row[:] for row in self.chips],
            turn=self.turn,
            trade=self.trade,
            player_id=self.player_id,
            last_action=self.last_action.copy() if self.last_action else None
        )

    def getLegalActions(self):
        actions = []
        hand = self.my_hand if self.turn == self.player_id else self.op_hand
        opp_colour = RED if self.player_id % 2 else BLU

        # First, give the agent the option to trade a dead card, if they haven't just done so.
        if not self.trade:
            for card in hand:
                if card[0] != 'j':
                    free_spaces = 0
                    for r, c in COORDS[card]:
                        if self.chips[r][c] == EMPTY:
                            free_spaces += 1
                    if not free_spaces:  # No option to place, so card is considered dead and can be traded.
                        for draft in self.draft:
                            actions.append({'play_card': card, 'draft_card': draft, 'type': 'trade', 'coords': None})

            if len(actions):  # If trade actions available, return those, along with the option to forego the trade.
                actions.append({'play_card': None, 'draft_card': None, 'type': 'trade', 'coords': None})
                return actions

        # If trade is prohibited, or no trades available, add action/s for each card in player's hand.
        # For each action, add copies corresponding to the various draft cards that could be selected at end of turn.
        for card in hand:
            if card in ['jd', 'jc']:  # two-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if self.chips[r][c] == EMPTY:
                            for draft in self.draft:
                                actions.append(
                                    {'play_card': card, 'draft_card': draft, 'type': 'place', 'coords': (r, c)})

            elif card in ['jh', 'js']:  # one-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if self.chips[r][c] == opp_colour:
                            for draft in self.draft:
                                actions.append(
                                    {'play_card': card, 'draft_card': draft, 'type': 'remove', 'coords': (r, c)})

            else:  # regular cards
                for r, c in COORDS[card]:
                    if self.chips[r][c] == EMPTY:
                        for draft in self.draft:
                            actions.append({'play_card': card, 'draft_card': draft, 'type': 'place', 'coords': (r, c)})

        return actions
