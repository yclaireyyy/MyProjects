# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Ruifan Zhang
# Date:    05/07/2021
# Purpose: Implements "Sequence" game agent

# IMPORTS ------------------------------------------------------------------------------------------------------------#

import time
import random
from copy import deepcopy

from Sequence.sequence_utils import *
from Sequence.sequence_model import COORDS
from Sequence.sequence_model import *
from agents.rz.gp4 import THINKTIME

# CONSTANTS ----------------------------------------------------------------------------------------------------------#
THINKTIME = 0.95


# CLASS DEF ----------------------------------------------------------------------------------------------------------#
class Node:
    gr = SequenceGameRule(2)
    card_mapping = {
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
    def __init__(self,
                 game_state,
                 player_id,
                 actions,
                 action,
                 layer,
                 parent=None):
        self.game_state = game_state
        self.player_id = player_id
        self.actions = actions
        self.parent = parent
        self.action = action
        self.children = []
        self.layer = layer
        self.alpha = 0
        self.beta = 0
        self.score = float('-inf')

    def expand(self):
        if self.actions is None:
            self.actions = Node.gr.getLegalActions(self.game_state, self.player_id)
        for action in self.actions:
            new_state = deepcopy(self.game_state)
            Node.gr.generateSuccessor(new_state,action,self.player_id)
            if action.get("type") == "trade":
                new_id = self.player_id
                new_layer = self.layer
            else:
                new_id = 1 - self.player_id
                new_layer = self.layer + 1
            new_Node = Node(new_state,new_id,None,action,new_layer,self)
            self.children.append(new_Node)

    def back_propagate(self):
        node = self
        while node.parent is not None:
            if node.parent.layer %2 == 0:
                node.parent.score = max(node.score, node.parent.score)
            else:
                node.parent.score = min(node.parent.score, node.parent.score)
            node = node.parent

    def get_score(self):
        my_list = []
        op_list = []
        to_score = {
            0:0,
            1:1,
            2:4,
            3:9,
            4:16,
            5:25,
        }
        vr_start = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9)]
        hz_start = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9)]
        d1_start = [(5,0),(4,0),(3,0),(2,0),(1,0),(0,0),(0,1),(0,2),(0,3),(0,4),(0,5)]
        d2_start = [(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(1,9),(2,9),(3,9),(4,9),(5,9)]
        starts_lists = [vr_start,hz_start,d1_start,d2_start]
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        heart_pos = [(4,4),(4,5),(5,4),(5,5)]

        target_id = self.player_id if self.layer %2 == 0 else 1-self.player_id
        agent = self.game_state.agent[target_id]
        clr, sclr = agent.colour, agent.seq_colour
        oc, os = agent.opp_colour, agent.opp_seq_colour

        chips = self.game_state.board.chips

        heart_score = 0
        heart_window = [chips[r][c] for r,c in heart_pos]
        a = heart_window.count(clr) + heart_window.count(sclr)
        b = heart_window.count(oc) + heart_window.count(os)
        if a > 0 and b > 0:
            heart_score = 0
        elif b == 0:
            heart_score = a**3
        elif a == 0:
            heart_score = -b**3


        for i, starts in enumerate(starts_lists):
            (dx,dy) = directions[i]
            for r, c in starts:
                window = []
                step = 0
                mv = 0
                ov = 0
                while 0 <= r+step*dx < 10 and 0 <= c+step*dy < 10:
                    window.append(chips[r+step*dx][c+step*dy])
                    step += 1
                for s in range(len(window)-4):
                    w = window[s:s+5]
                    mv = max(mv,self.place_value(w,(clr,sclr,oc,os)))
                    ov = max(ov,self.place_value(w,(oc,os,clr,sclr)))
                my_list.append(mv)
                op_list.append(ov)
        my_list.sort(reverse=True)
        op_list.sort(reverse=True)
        my_score = sum([to_score[each] for each in my_list])
        op_score = sum([to_score[each] for each in op_list])
        self.score = heart_score + my_score - 0.8 * op_score


    def place_value(self, window:list, clrs):
        debug = False
        # 窗口长度必须为5
        assert len(window) == 5
        clr, sclr, oc, os = clrs
        # 该窗口无法作为潜在得分窗口
        # 5格窗口内存在对手棋子，无放置必要
        # 5格窗口内存在2枚及以上自己的Sequence Chips，无放置必要
        if oc in window or os in window or window.count(sclr) > 1:
            return 0
        # 在窗口内只有自己棋子的情况，返回窗口内自己棋子的数量
        return window.count(clr) + window.count(sclr) + window.count(JOKER)

class Minimax:
    def __init__(self, player_id, max_depth):
        self.player_id = player_id
        self.max_depth = max_depth
        self.root = None

    def set_root(self, actions, game_state):
        self.root = Node(
            game_state,
            self.player_id,
            actions,
            None,
            0,
            None
        )

    def search(self):
        start = time.time()
        # Step 1: 构建所有节点（BFS展开到 max_depth）
        queue = [self.root]
        all_nodes = [self.root]
        leafs = []

        while queue:
            node = queue.pop(0)
            # 展开的节点
            if node.layer < self.max_depth and node.game_state.agents[1-self.player_id].hand != []:
                node.expand()
                queue.extend(node.children)
                all_nodes.extend(node.children)
            # 叶子结点
            else:
                node.get_score()
                # if node.score != 0:
                #     print("find node with score", node.score)
                node.back_propagate()
                leafs.append(node)
            if time.time() - start > THINKTIME:
                break

        # Step 3: 从 root 中选得分最高的动作
        best_score = float("-inf")
        best_action = random.choice(self.root.actions)
        for child in self.root.children:
            if child.score > best_score:
                best_score = child.score
                best_action = child.action
        return best_score, best_action


class myAgent:
    def __init__(self, _id):
        self.id = _id
        self.op_last = 0
        self.guesses = []
        self.deck = [(r+s) for r in ['2','3','4','5','6','7','8','9','t','j','q','k','a'] for s in ['d','c','h','s']]
        self.deck = self.deck * 2
        self.last_draft = []
        random.shuffle(self.deck)
        self.seq_score = 0
        self.op_score = 0
        self.search_tree = Minimax(self.id, 1)

    def SelectAction(self, actions, game_state):
        self.calc_rest(game_state)
        root_state = deepcopy(game_state)
        root_state.agents[1 - self.id].hand = deepcopy(self.guesses)
        root_state.deck.cards = deepcopy(self.deck)
        self.search_tree.set_root(actions, root_state)
        try:
            score, action = self.search_tree.search()
            print(score, action)
        except Exception as e:
            print(e)
            return random.choice(actions)
        return action

    def calc_rest(self, game_state):
        # 初始化
        # 去除自己的手牌
        if self.op_last == 0:
            for each in game_state.agents[self.id].hand:
                self.deck.remove(each)
        # 去除未见过的公共牌
        for each in game_state.board.draft:
            if each not in self.last_draft:
                self.deck.remove(each)
        # 保存当前公共牌
        self.last_draft = []
        for each in game_state.board.draft:
            self.last_draft.append(each)

        # 对手手牌
        trace = game_state.agents[1 - self.id].agent_trace.action_reward[self.op_last:]
        self.op_last = len(game_state.agents[1 - self.id].agent_trace.action_reward)
        picked = []
        discarded = []
        for action, r in trace:
            if action["draft_card"] is not None:
                picked.append(action["draft_card"])
            if action["play_card"] is not None:
                discarded.append(action["play_card"])
        for each in picked:
            self.guesses.append(each)

        # 要么从对方手中去除
        # 要么从牌堆中去除
        for each in discarded:
            if each in self.guesses:
                self.guesses.remove(each)
            else:
                self.deck.remove(each)
