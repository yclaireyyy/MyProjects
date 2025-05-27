from collections import namedtuple
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
from Sequence.sequence_model import COORDS
import heapq
import time
import itertools
import random
import math
from collections import defaultdict

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
        self.CORNER_POSITIONS = [(0, 0), (0, 9), (9, 0), (9, 9)]

        # 启动时间和回合追踪
        self.startup_start_time = time.time()
        self.startup_time_limit = 15.0
        self.startup_time_used = False
        self.turn_count = 0

        # 新增：游戏阶段权重配置
        self.phase_weights = {
            'opening': {'center': 2.0, 'chain': 0.8, 'block': 0.5, 'corner': 1.5},
            'middle': {'center': 1.0, 'chain': 1.5, 'block': 1.2, 'corner': 1.8},
            'endgame': {'center': 0.3, 'chain': 2.0, 'block': 1.8, 'corner': 2.5}
        }

        # 蒙特卡罗模拟参数
        self.mc_simulations = 50  # 平衡计算时间和准确性

        # 选牌策略相关配置
        self.card_selection_weights = {
            'immediate_play': 3.0,  # 立即可用的价值
            'strategic_position': 2.0,  # 战略位置价值
            'blocking_value': 2.5,  # 阻断敌方的价值
            'flexibility': 1.0,  # 未来灵活性
            'opponent_denial': 2.8  # 拒绝对手获得的价值
        }

        self.COORDS = COORDS

        # 启动时预计算优化
        if not self.startup_time_used:
            self._precompute_startup_data()

    def _precompute_startup_data(self):
        """利用15秒启动时间进行预计算优化"""
        if self.startup_time_used:
            return

        startup_elapsed = time.time() - self.startup_start_time
        remaining_startup_time = self.startup_time_limit - startup_elapsed

        if remaining_startup_time <= 2.0:  # 保留1秒安全缓冲
            self.startup_time_used = True
            return

        # 核心预计算1：位置价值映射表
        # 为每个棋盘位置预先计算好"基础地租"
        self.position_values = {}
        for r in range(10):
            for c in range(10):
                self.position_values[(r, c)] = self._calculate_base_position_value(r, c)

        # 核心预计算2：角落影响区域映射（修复原有的数据结构问题）
        self.corner_influence_zones = {}  # 角落 -> 影响位置列表
        self.position_to_corners = {}  # 位置 -> 影响它的角落列表（反向映射）

        for corner in self.CORNER_POSITIONS:
            influence_positions = self._compute_corner_influence_zone(corner)
            self.corner_influence_zones[corner] = influence_positions

            # 建立反向映射，用于O(1)时间查询位置是否受角落影响
            for pos in influence_positions:
                if pos not in self.position_to_corners:
                    self.position_to_corners[pos] = []
                self.position_to_corners[pos].append(corner)

        # 核心预计算3：方向向量和常用集合
        self.direction_vectors = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 预计算HOTB邻接区域，用于快速检查HOTB影响
        self.hotb_adjacent = set()
        for hr, hc in HOTB_COORDS:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = hr + dr, hc + dc
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        self.hotb_adjacent.add((nr, nc))

        # 预计算常用的威胁评估模板
        self._precompute_threat_patterns()

        self.startup_time_used = True

    def _precompute_position_value(self, r, c):
        """预计算位置的基础战略价值"""
        value = 0

        # HOTB区域高价值
        if (r, c) in HOTB_COORDS:
            value += 100

        # 中心区域价值
        distance_to_center = abs(r - 4.5) + abs(c - 4.5)
        value += max(0, 20 - distance_to_center * 2)

        # 角落邻近价值
        for corner_r, corner_c in self.CORNER_POSITIONS:
            corner_distance = max(abs(r - corner_r), abs(c - corner_c))
            if corner_distance <= 3:
                value += max(0, 15 - corner_distance * 5)

        return value

    def _compute_corner_influence(self, corner):
        """计算角落的影响区域"""
        corner_r, corner_c = corner
        influence_positions = []

        # 计算从角落出发的各个方向
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                # 沿着这个方向找出连线位置
                for i in range(1, 6):  # 最多5个位置形成胜利连线
                    r, c = corner_r + dr * i, corner_c + dc * i
                    if 0 <= r < 10 and 0 <= c < 10:
                        influence_positions.append((r, c))
                    else:
                        break

        return influence_positions

    def is_corner_position(self, r, c):
        """检查位置是否为角落"""
        return (r, c) in self.CORNER_POSITIONS

    def SelectAction(self, actions, game_state):
        self.turn_count += 1

        # 如果仍在启动时间内，使用剩余启动时间
        if not self.startup_time_used:
            startup_elapsed = time.time() - self.startup_start_time
            if startup_elapsed < self.startup_time_limit:
                time_limit = self.startup_time_limit - startup_elapsed
            else:
                time_limit = 0.98  # 标准时间限制
                self.startup_time_used = True
        else:
            time_limit = 0.98  # 为1秒限制预留安全缓冲

        # 初始化时间管理器
        self.time_manager = TimeManager(time_limit)
        self.time_manager.start_turn()

        action_type = self.identify_action_type(actions, game_state)

        if action_type == 'card_selection':
            # 选牌阶段：从5张明牌中选择最优的一张
            return self.select_card_strategy(actions, game_state)
        elif action_type == 'place':
            # 放牌阶段：选择最优的放置位置
            return self.place_strategy(actions, game_state)
        else:
            # 默认策略：如果无法识别类型，使用原始放牌策略
            return self.original_place_strategy(actions, game_state)

    def identify_action_type(self, actions, game_state):
        """
        判断是选牌还是放牌
        """
        if not actions:
            return None

        # 检查游戏状态来确定当前应该是什么类型的动作
        if game_state:
            try:
                agent = game_state.agents[self.id]

                # 如果有hand属性且长度小于预期，可能是选牌阶段
                if hasattr(agent, 'hand') and len(agent.hand) < 7:
                    # 检查动作是否包含选牌相关字段
                    for action in actions:
                        if isinstance(action, dict):
                            if ('draft_card' in action or 'card' in action or
                                    action.get('type') == 'draw' or action.get('type') == 'select_card'):
                                return 'card_selection'
            except:
                pass

        # 检查是否为放置动作
        for action in actions:
            if isinstance(action, dict):
                if 'coords' in action and action.get('type') == 'place':
                    return 'place'
                elif hasattr(action, 'coords') or 'coords' in str(action):
                    return 'place'

        return 'place'  # 默认假设为放置动作

    def safe_get_game_state_info(self, game_state):
        """
        返回 (agent, board, is_valid) 三元组
        """
        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            return agent, board, True
        except (AttributeError, KeyError, IndexError) as e:
            # 如果无法获取状态信息，返回默认值
            return None, None, False

    def select_card_strategy(self, card_actions, game_state):
        """
        选牌策略核心方法 - 从5张牌中选择最优的一张
        设计思路：快速评估每张牌的战略价值，选择最高分的
        """
        if not card_actions:
            return None

        try:
            # 获取当前游戏状态信息
            agent, board, is_valid = self.safe_get_game_state_info(game_state)
            if not is_valid:
                return card_actions[0]

            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            best_card_action = card_actions[0]
            best_score = float('-inf')

            # 评估每张可选的牌
            for card_action in card_actions:
                try:
                    card_score = self.evaluate_card_value(
                        card_action, board, my_color, enemy_color, game_state
                    )

                    if card_score > best_score:
                        best_score = card_score
                        best_card_action = card_action

                except Exception as e:
                    continue

            return best_card_action

        except Exception as e:
            return card_actions[0]

    def evaluate_card_value(self, card_action, board, my_color, enemy_color, game_state):
        """
        评估单张牌的战略价值,考虑对手阻断
        """
        total_score = 0

        try:
            # 获取这张牌对应的棋盘位置
            card_positions = self.get_card_board_positions(card_action, game_state)

            if not card_positions:
                return 0  # 如果牌无效，返回0分

            # 维度1：立即可用价值 - 现在就能下的位置有多好
            immediate_value = self.calculate_immediate_play_value(
                card_positions, board, my_color
            )
            total_score += immediate_value * self.card_selection_weights['immediate_play']

            # 维度2：战略位置价值 - 这些位置的长期战略意义
            strategic_value = self.calculate_strategic_position_value(
                card_positions, board, my_color
            )
            total_score += strategic_value * self.card_selection_weights['strategic_position']

            # 维度3：阻断价值 - 阻止敌方使用这张牌的价值
            blocking_value = self.calculate_blocking_value(
                card_positions, board, enemy_color
            )
            total_score += blocking_value * self.card_selection_weights['blocking_value']

            # 维度4：灵活性价值 - 这张牌提供的选择多样性
            flexibility_value = self.calculate_flexibility_value(card_positions, board)
            total_score += flexibility_value * self.card_selection_weights['flexibility']

            # 维度5：对手拒绝价值 - 阻止对手获得这张牌
            opponent_denial_value = self.calculate_opponent_denial_value(
                card_positions, board, enemy_color
            )
            total_score += opponent_denial_value * self.card_selection_weights['opponent_denial']


        except Exception as e:
            return 0

        return total_score

    def calculate_opponent_denial_value(self, positions, board, enemy_color):
        """
        计算阻止对手获得这张牌的价值
        """
        denial_value = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 评估对手在这个位置的潜在价值
                opponent_potential = 0

                # 检查对手在此位置能形成的连接
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    opponent_chain_value = self.evaluate_opponent_chain_at_position(
                        board, r, c, dx, dy, enemy_color
                    )
                    opponent_potential += opponent_chain_value

                # 检查是否是关键战略位置
                if (r, c) in HOTB_COORDS:
                    opponent_potential += 80  # HOTB对对手也很重要

                # 检查角落支持价值
                if self.position_near_corner(r, c):
                    opponent_potential += 40

                denial_value += opponent_potential

        return denial_value

    def evaluate_opponent_chain_at_position(self, board, r, c, dx, dy, enemy_color):
        """评估对手在特定位置和方向的连子价值"""
        # 临时模拟对手在此位置放置棋子
        temp_board = [row[:] for row in board]
        temp_board[r][c] = enemy_color

        # 修复6：使用角落感知的连子分析
        count, openings = self.analyze_chain_pattern(
            temp_board, r, c, dx, dy, enemy_color
        )

        # 根据连子威胁程度评分
        if count >= 4:
            return 1000  # 阻止对手即将获胜
        elif count >= 3:
            return 300 if openings >= 1 else 100
        elif count >= 2:
            return 80 if openings >= 1 else 20
        else:
            return 10

    def position_near_corner(self, r, c):
        """检查位置是否靠近角落"""
        for corner_r, corner_c in self.CORNER_POSITIONS:
            if max(abs(r - corner_r), abs(c - corner_c)) <= 2:
                return True
        return False

    def get_card_board_positions(self, card_action, game_state):
        """
        获取一张牌对应的棋盘位置列表,真正关心的是draft_card
        """
        try:
            # 尝试多种可能的字段名称
            card = None
            # 处理5张明牌选择的特殊格式
            if isinstance(card_action, dict):
                # 优先检查draft_card
                card = (card_action.get('draft_card') or
                        card_action.get('card') or
                        card_action.get('play_card') or
                        card_action.get('selected_card'))

            # 如果还是找不到牌面信息，尝试其他方式
            if not card:
                if hasattr(card_action, 'draft_card'):
                    card = card_action.draft_card
                elif hasattr(card_action, 'card'):
                    card = card_action.card
                elif hasattr(card_action, 'selected_card'):
                    card = card_action.selected_card

            # 确保获得的是有效的牌面信息
            if card:
                # 处理字符串格式的牌面
                if isinstance(card, str) and card in self.COORDS:
                    return self.COORDS[card]

                # 处理可能的元组格式 (suit, rank) 或 (rank, suit)
                if isinstance(card, (tuple, list)) and len(card) == 2:
                    # 尝试两种可能的组合方式
                    card_str1 = f"{card[1]}{card[0]}"  # rank + suit
                    card_str2 = f"{card[0]}{card[1]}"  # suit + rank

                    if card_str1 in self.COORDS:
                        return self.COORDS[card_str1]
                    elif card_str2 in self.COORDS:
                        return self.COORDS[card_str2]

            # 如果没有找到有效的牌面信息，返回空列表
            return []

        except Exception as e:
            return []

    def place_strategy(self, actions, game_state):
        # 第一层防护：快速启发式决策（保底方案）
        quick_decision = self.get_quick_decision(actions, game_state)

        if not self.time_manager.should_continue_phase('heuristic_search'):
            return quick_decision

        # 第二层：标准A*搜索（主要决策）
        try:
            better_decision = self.decision_making(game_state, actions, quick_decision)
            return better_decision
        except Exception as e:
            # 如果搜索过程中出现任何问题，立即返回保底决策
            return quick_decision

    def card_to_positions_mapping(self, card, game_state):
        """
        牌面到棋盘位置的映射方法
        """
        try:
            if card and card in self.COORDS:
                return self.COORDS[card]

            # 处理特殊情况：如果是None或未知牌面，返回空列表
            return []

        except Exception as e:
            return []

    def _calculate_base_position_value(self, r, c):
        """基础位置价值计算"""
        value = 0

        # HOTB区域高价值
        if (r, c) in HOTB_COORDS:
            value += 100

        # 中心区域价值
        distance_to_center = abs(r - 4.5) + abs(c - 4.5)
        value += max(0, 20 - distance_to_center * 2)

        # 角落邻近价值
        for corner_r, corner_c in self.CORNER_POSITIONS:
            corner_distance = max(abs(r - corner_r), abs(c - corner_c))
            if corner_distance <= 3:
                value += max(0, 15 - corner_distance * 5)

        return value

    def _compute_corner_influence_zone(self, corner):
        """角落影响区域计算"""
        corner_r, corner_c = corner
        influence_positions = []

        # 计算从角落出发的各个方向
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                # 沿着这个方向找出连线位置
                for i in range(1, 6):
                    r, c = corner_r + dr * i, corner_c + dc * i
                    if 0 <= r < 10 and 0 <= c < 10:
                        influence_positions.append((r, c))
                    else:
                        break

        return influence_positions

    def _precompute_threat_patterns(self):
        """实现缺失的威胁模式预计算"""
        # 预计算常用的威胁评估模板
        self.threat_patterns = {
            'win_pattern': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
            'threat_4': [(0, 0), (0, 1), (0, 2), (0, 3)],
            'threat_3': [(0, 0), (0, 1), (0, 2)],
            'threat_2': [(0, 0), (0, 1)]
        }

    def calculate_immediate_play_value(self, positions, board, my_color):
        """
        计算立即可下价值 - 评估现在就能利用的位置质量
        """
        value = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 使用预计算的位置价值
                position_value = self.position_values.get((r, c), 0)

                # 简化的连子潜力评估
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    chain_potential = self.quick_chain_evaluation(
                        board, r, c, dx, dy, my_color
                    )
                    position_value += chain_potential

                value += position_value

        return value

    def quick_chain_evaluation(self, board, r, c, dx, dy, color):
        """
        快速连子评估 - 简化版的连子分析，用于选牌阶段的快速决策
        """
        # 临时放置棋子
        temp_board = [row[:] for row in board]
        temp_board[r][c] = color

        # 使用简化的连子计数
        count = 1
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if (temp_board[x][y] == color or
                            self.is_corner_position(x, y)):
                        count += 1
                    else:
                        break
                else:
                    break

        # 根据连子数给分
        if count >= 4:
            return 1000
        elif count >= 3:
            return 300
        elif count >= 2:
            return 100
        else:
            return 10

    def calculate_strategic_position_value(self, positions, board, my_color):
        """
        计算战略位置价值
        """
        value = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10:
                # 使用预计算的基础价值
                strategic_importance = self.position_values.get((r, c), 0)

                # 额外的动态战略价值
                strategic_importance += self.corner_strategic_value(r, c, board, my_color)

                value += strategic_importance

        return value

    def calculate_blocking_value(self, positions, board, enemy_color):
        """
        计算阻断价值 - 评估阻止敌方使用这张牌的价值
        """
        value = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 评估敌方在这个位置的威胁程度
                enemy_threat = 0

                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    threat_level = self.calculate_position_threat_for_enemy(
                        board, r, c, dx, dy, enemy_color
                    )
                    enemy_threat += threat_level

                # 威胁越大，阻断价值越高
                value += enemy_threat * 0.8

        return value

    def calculate_flexibility_value(self, positions, board):
        """
        计算灵活性价值 - 这张牌提供多少可用选择
        """
        available_positions = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                available_positions += 1

        # 可用位置越多，灵活性越高
        return available_positions * 10

    def calculate_position_threat_for_enemy(self, board, r, c, dx, dy, enemy_color):
        """
        计算敌方在特定位置的威胁程度
        """
        threat_level = 0

        # 临时模拟敌方在此位置放置棋子
        temp_board = [row[:] for row in board]
        temp_board[r][c] = enemy_color

        # 分析敌方在此方向的连子情况
        for direction in [1, -1]:  # 两个方向
            count = 0
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if temp_board[x][y] == enemy_color or self.is_corner_position(x, y):
                        count += 1
                    elif temp_board[x][y] == '0':
                        break  # 遇到空位，停止计数
                    else:
                        break  # 遇到我方棋子，停止计数
                else:
                    break  # 超出边界
            threat_level = max(threat_level, count)

        return threat_level

    def original_place_strategy(self, actions, game_state):
        # 第一层防护：快速启发式决策（保底方案）
        quick_decision = self.get_quick_decision(actions, game_state)

        if not self.time_manager.should_continue_phase('heuristic_search'):
            return quick_decision

        # 第二层：标准A*搜索（主要决策）
        try:
            better_decision = self.decision_making(game_state, actions, quick_decision)
            return better_decision
        except Exception as e:
            # 如果搜索过程中出现任何问题，立即返回保底决策
            return quick_decision

    def get_quick_decision(self, actions, game_state):
        """
        快速决策：150毫秒内必须完成的保底方案
        """
        if not actions:
            return None

        # 使用简化的评估方法，只考虑最重要的因素
        best_action = actions[0]
        best_score = float('-inf')

        agent, board, is_valid = self.safe_get_game_state_info(game_state)
        if not is_valid:
            return actions[0]

        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        for action in actions[:min(len(actions), 5)]:  # 限制候选数量，确保速度
            if not action.get('coords'):
                continue

            r, c = action['coords']

            # 简化的评分：只考虑最关键的因素
            score = 0

            # 中心控制价值（快速计算）
            if (r, c) in HOTB_COORDS:
                score += 100
            elif 3 <= r <= 6 and 3 <= c <= 6:
                score += 50

            # 使用角落感知的连子检查
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                adjacent_same = 0
                for direction in [1, -1]:
                    x, y = r + dx * direction, c + dy * direction
                    if (0 <= x < 10 and 0 <= y < 10 and
                            (board[x][y] == my_color or self.is_corner_position(x, y))):
                        adjacent_same += 1
                score += adjacent_same * 20

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # 增强的决策制定：融合A*、蒙特卡罗和深度评估以及时间限制，寻找最优解
    def decision_making(self, initial_state, candidate_moves, fallback_decision):
        current_best = fallback_decision
        # 第一阶段：改进的启发式搜索
        if self.time_manager.should_continue_phase('heuristic_search'):
            try:
                heuristic_result = self.a_star(initial_state, candidate_moves)
                if heuristic_result:
                    current_best = heuristic_result
            except TimeoutError:
                pass  # 时间不够，使用当前最佳决策

        # 第二阶段：蒙特卡罗验证（如果时间充裕）
        if self.time_manager.should_continue_phase('monte_carlo'):
            try:
                mc_result = self.monte_carlo_validation(initial_state, [current_best] + candidate_moves[:3])
                if mc_result:
                    current_best = mc_result
            except TimeoutError:
                pass  # 时间不够，使用当前最佳决策

        return current_best

    def a_star(self, initial_state, candidate_moves):
        pending = []
        seen_states = set()
        best_sequence = []
        top_reward = float('-inf')
        expansions_count = 0
        remaining_time = self.time_manager.get_remaining_for_phase('heuristic_search')

        # 根据剩余时间动态调整搜索参数
        if remaining_time > 0.3:
            max_candidates = 8
            max_expansions = 50
        elif remaining_time > 0.15:
            max_candidates = 5
            max_expansions = 20
        else:
            max_candidates = 3
            max_expansions = 10

        # 改进1：使用新的启发式函数进行初始排序
        candidate_moves.sort(key=lambda move: self.heuristic(initial_state, move))

        for move in candidate_moves[:max_candidates]:  # 限制初始候选数量，提高效率
            g = 1
            h = self.heuristic(initial_state, move)
            f = g + h
            heapq.heappush(pending, (f, next(self.counter), g, h,
                                     self.fast_simulate(initial_state, move), [move]))

        while (pending and
               expansions_count < max_expansions and
               self.time_manager.should_continue_phase('heuristic_search')):  # 为后续算法预留时间

            f, _, g, h, current_state, move_history = heapq.heappop(pending)
            last_move = move_history[-1]
            expansions_count += 1

            # 改进2：更精确的状态签名
            state_signature = self.create_enhanced_state_signature(current_state, last_move)
            if state_signature in seen_states:
                continue
            seen_states.add(state_signature)

            # 改进3：使用增强的状态评估
            reward = self.evaluate_state(current_state, last_move)
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history

            # 改进4：智能剪枝 - 根据当前最佳分数决定扩展深度 - 只在时间和深度都允许的情况下扩展
            if (g < 2 and
                    self.time_manager.get_remaining_for_phase('heuristic_search') > 0.1):

                next_steps = self.rule.getLegalActions(current_state, self.id)
                next_steps.sort(key=lambda act: self.heuristic(current_state, act))

                for next_move in next_steps[:3]:  # 进一步限制分支
                    next_g = g + 1
                    next_h = self.heuristic(current_state, next_move)
                    heapq.heappush(pending, (
                        next_g + next_h, next(self.counter),
                        next_g, next_h,
                        self.fast_simulate(current_state, next_move),
                        move_history + [next_move]
                    ))

        return best_sequence[0] if best_sequence else None

    def monte_carlo_validation(self, initial_state, top_candidates):
        if not top_candidates:
            return None
        action_scores = {}
        simulations_per_action = max(1, self.mc_simulations // len(top_candidates))

        for action in top_candidates:
            if not self.time_manager.should_continue_phase('monte_carlo'):
                break  # 时间控制

            total_score = 0
            for _ in range(simulations_per_action):
                # 模拟这个行动的多种可能结果
                simulated_reward = self.simulate_action_outcome(initial_state, action)
                total_score += simulated_reward

            action_scores[action] = total_score / simulations_per_action

        return max(action_scores.items(), key=lambda x: x[1])[0] if action_scores else top_candidates[0]

    # 模拟单个行动的结果：考虑随机性
    def simulate_action_outcome(self, state, action, depth=2):
        current_state = self.fast_simulate(state, action)
        total_reward = self.evaluate_state(current_state, action)

        # 模拟后续几步的随机发展
        for step in range(depth):
            possible_actions = self.rule.getLegalActions(current_state, self.id)
            if not possible_actions:
                break
            # 使用启发式选择而非完全随机，提高模拟质量
            weights = [1.0 / (1 + self.heuristic(current_state, act)) for act in possible_actions]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            chosen_action = random.choices(possible_actions, weights=weights)[0]
            current_state = self.fast_simulate(current_state, chosen_action)

        return total_reward + self.evaluate_state(current_state, None) * 0.3

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
            return 1000

        r, c = action['coords']
        board = [row[:] for row in state.board.chips]
        me = state.agents[self.id]
        color = me.colour
        enemy = 'r' if color == 'b' else 'b'
        # 临时放置棋子进行评估
        board[r][c] = color
        # 获取当前游戏阶段的权重
        phase_weights = self.get_current_phase_weights(board)
        score = 0
        # 改进：所有评分函数都考虑角落万能位置
        score += self.center_bias(r, c) * phase_weights['center']
        score += self.chain_score(board, r, c, color) * phase_weights['chain']
        score += self.block_enemy_score(board, r, c, enemy) * phase_weights['block']
        score += self.hotb_score(board, color)
        score += self.corner_strategic_value(r, c, board, color) * phase_weights['corner']

        # 新增：检查获胜潜力并给予额外优先级
        win_potential_bonus = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            if self.check_win_potential(board, r, c, dx, dy, color):
                # 发现获胜潜力，大幅提升这个走法的优先级
                win_potential_bonus += 200  # 这会显著降低A*的成本值
                break  # 只要发现一个方向有获胜潜力就足够了

        # 转换为A*成本（越小越好），获胜潜力会大幅降低成本
        return max(1, 1000 - score - win_potential_bonus)

    def center_bias(self, r, c, board=None, color=None):
        # 使用预计算的基础价值
        base_score = self.position_values.get((r, c), 0)

        # 动态邻近HOTB加分
        if board is not None and color is not None:
            if (r, c) in self.hotb_adjacent:
                # 检查邻近的HOTB位置是否可推进
                hotb_potential = 0
                for hr, hc in HOTB_COORDS:
                    if abs(r - hr) <= 1 and abs(c - hc) <= 1 and board[hr][hc] == '0':
                        hotb_potential += 5
                base_score += hotb_potential

        return base_score

    # 指数级评分系统
    def chain_score(self, board, r, c, color):
        total_score = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count, openings = self.analyze_chain_pattern(board, r, c, dx, dy, color)

            corner_support = self.check_corner_support(board, r, c, dx, dy, color)

            # 指数级评分逻辑 - 关键差异被放大
            if count >= 5:
                total_score += 50000  # 胜利局面：压倒性优势
            elif count == 4:
                if openings == 2:  # 双端开放的活四
                    total_score += 20000  # 必胜局面
                elif corner_support and openings >= 1:
                    total_score += 18000 # 角落支持的活四：非常强的威胁
                elif openings == 1:  # 单端开放的冲四,强威胁，但可防守
                    total_score += 12000
                elif corner_support:
                    total_score += 10000
                else:  # 完全被堵的死四
                    total_score += 2000  # 几乎无用
            elif count == 3:
                if openings == 2:  # 活三
                    total_score += 1500  # 重要进攻点
                elif corner_support and openings >= 1:
                    total_score += 1200   # 角落支持的活三：优质发展
                elif openings == 1:
                    # 单端开放的三连：一般发展
                    total_score += 400
                elif corner_support:
                    # 纯角落支持的三连
                    total_score += 600
                else:
                    # 死三：价值很低
                    total_score += 50
            elif count == 2:
                if openings >= 2:
                    # 双端开放的活二：有发展空间
                    total_score += 150
                elif corner_support and openings >= 1:
                    # 角落支持的活二
                    total_score += 120
                elif openings >= 1:
                    # 单端开放的二连
                    total_score += 60
                elif corner_support:
                    # 角落支持的二连
                    total_score += 80
                else:
                    # 死二：基本无价值
                    total_score += 10
            else:
                if corner_support:
                    total_score += 20  # 角落附近的单子有潜在价值
                else:
                    total_score += 5  # 普通单子的基础分

        return total_score

    def check_corner_support(self, board, r, c, dx, dy, color):
        """
        检查连子方向上是否包含角落位置
        """
        # 检查正向和负向是否经过角落
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if not (0 <= x < 10 and 0 <= y < 10):
                    break
                if self.is_corner_position(x, y):
                    return True  # 找到角落支持
                if board[x][y] != color and board[x][y] != '0' and not self.is_corner_position(x, y):
                    break  # 被敌方阻断
        return False

    def analyze_chain_pattern(self, board, r, c, dx, dy, color):
        """角落感知的连子模式分析"""
        count = 1  # 包含当前位置
        openings = 0  # 开放端计数
        # 正向计数
        pos_open = False
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color or self.is_corner_position(x, y):
                    count += 1
                elif board[x][y] == '0':
                    pos_open = True
                    break
                else:  # 敌方棋子
                    break
            else:  # 边界
                break
        # 负向计数
        neg_open = False
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == color or self.is_corner_position(x, y):
                    count += 1
                elif board[x][y] == '0':
                    neg_open = True
                    break
                else:  # 敌方棋子
                    break
            else:  # 边界
                break

        # 计算开口总数
        openings = (1 if pos_open else 0) + (1 if neg_open else 0)
        return count, openings

    def corner_strategic_value(self, r, c, board, color):
        """评估位置的角落战略价值"""
        corner_value = 0

        # 检查这个位置是否在任何角落的影响区域内
        for corner, positions in self.corner_influence_zones.items():
            if (r, c) in positions:
                corner_r, corner_c = corner
                distance = max(abs(r - corner_r), abs(c - corner_c))
                line_potential = self.evaluate_corner_line_potential(
                    board, r, c, corner_r, corner_c, color
                )
                corner_value += line_potential * max(0, 60 - distance * 10)

        return corner_value

    def can_form_strategic_line(self, r1, c1, r2, c2):
        """检查两点是否能形成战略直线"""
        return (r1 == r2 or c1 == c2 or  # 横线或竖线
                abs(r1 - r2) == abs(c1 - c2))  # 对角线

    def evaluate_corner_line_potential(self, board, r, c, corner_r, corner_c, color):
        """评估与角落形成直线的潜力"""
        # 计算这条线上已有的同色棋子数
        dr = 0 if corner_r == r else (1 if corner_r > r else -1)
        dc = 0 if corner_c == c else (1 if corner_c > c else -1)

        friendly_count = 0
        enemy_count = 0

        current_r, current_c = r, c
        while (current_r != corner_r or current_c != corner_c):
            if 0 <= current_r < 10 and 0 <= current_c < 10:
                if board[current_r][current_c] == color:
                    friendly_count += 1
                elif board[current_r][current_c] != '0':
                    enemy_count += 1
            current_r += dr
            current_c += dc

        # 如果没有敌方阻挡且有发展空间，则有价值
        return max(0, friendly_count * 20 - enemy_count * 50) if enemy_count < 2 else 0

    def check_win_potential(self, board, r, c, dx, dy, color):
        """
        评估在当前位置放子后，是否能在后续几步内形成胜利威胁
        """
        # 使用角落感知的分析
        # 获取基础分析结果（元组格式）
        count, openings = self.analyze_chain_pattern(board, r, c, dx, dy, color)
        corner_support = self.check_corner_support(board, r, c, dx, dy, color)

        # 如果已经有4连且有开口或角落支持，就有直接获胜潜力
        if (count >= 4 and
                (openings >= 1 or corner_support)):
            return True

        # 如果有3连且有多个开口，也有很强的获胜潜力
        if (count >= 3 and
                openings >= 2):
            return True

        return False

    def block_enemy_score(self, board, r, c, enemy_color):
        score = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            enemy_threat = self.calculate_enemy_threat_in_direction(
                board, r, c, dx, dy, enemy_color)
            if enemy_threat >= 4:  # 阻止即将获胜
                score += 8000
            elif enemy_threat >= 3:  # 阻止强威胁
                score += 1500
            elif enemy_threat >= 2:  # 阻止发展
                score += 300

        return score

    def calculate_enemy_threat_in_direction(self, board, r, c, dx, dy, enemy_color):
        """计算特定方向的敌方威胁等级（考虑角落）"""
        original = board[r][c]

        # 分析敌方在此方向的连子情况
        threat_level = 0
        for direction in [1, -1]:  # 两个方向
            count = 0
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    # 修复：通过逻辑判断来模拟敌方棋子
                    cell_value = board[x][y]
                    # 如果是目标位置(r,c)，假设它是敌方棋子
                    if x == r and y == c:
                        cell_value = enemy_color
                    if cell_value == enemy_color or self.is_corner_position(x, y):
                        count += 1
                    elif cell_value == '0':
                        break
                    else:
                        break
                else:
                    break
            threat_level = max(threat_level, count)

        return threat_level

    def hotb_score(self, board, color):
        enemy = 'r' if color == 'b' else 'b'
        score = 0
        controlled = 0
        enemy_controlled = 0

        for r, c in HOTB_COORDS:
            if board[r][c] == color:
                controlled += 1
                score += 40  # 提高单格价值
            elif board[r][c] == enemy:
                enemy_controlled += 1
                score -= 60  # 提高失控惩罚

        # 控制优势奖励
        if controlled >= 3:
            score += 150  # 接近完全控制
        if controlled == 4:
            score += 300  # 完全控制额外奖励

        # 阻止敌方控制的紧迫性
        if enemy_controlled >= 2:
            score -= 100  # 敌方威胁惩罚

        return score

    def get_current_phase_weights(self, board):
        """根据当前局面判断游戏阶段并返回相应权重"""
        total_pieces = sum(1 for row in board for cell in row if cell != '0')

        if total_pieces < 12:
            return self.phase_weights['opening']
        elif total_pieces < 35:
            return self.phase_weights['middle']
        else:
            return self.phase_weights['endgame']

    def evaluate_state(self, state, action):
        agent = state.agents[self.id]
        board = [row[:] for row in state.board.chips]
        if action and action.get('coords'):
            r, c = action['coords']
            board[r][c] = agent.colour

        return self.evaluate_board(board, agent)

    def evaluate_board(self, board, agent):
        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        score = 0
        # 加权评估各个方面
        score += self.score_all_chains(board, my_color) * 2.0
        score -= self.score_all_chains(board, enemy_color) * 1.5  # 敌方威胁权重稍低
        score += self.hotb_score(board, my_color)
        score += self.corner_control_bonus(board, my_color)

        # 敌方威胁评估
        enemy_threat_level = self.score_enemy_threat(board, enemy_color)
        score += enemy_threat_level * 0.8  # 给敌方威胁适当的权重

        return score

    def simple_chain_analysis_for_scoring(self, board, r, c, dx, dy, color):
        # 使用与chain_score完全相同的分析方法
        count, openings = self.analyze_chain_pattern(board, r, c, dx, dy, color)
        corner_support = self.check_corner_support(board, r, c, dx, dy, color)

        # 简单的包装：将元组结果转换为字典格式
        return {
            'total_count': count,  # 基础连子数
            'openings': openings,
            'corner_support': corner_support
        }

    def score_all_chains(self, board, color):
        """评估棋盘上所有的连子情况"""
        total_score = 0
        processed_positions = set()

        for r in range(10):
            for c in range(10):
                if (r, c) in processed_positions:
                    continue

                if board[r][c] == color:
                    # 为每个棋子计算其参与的最佳连子
                    max_chain_score = 0
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        chain_info = self.simple_chain_analysis_for_scoring(board, r, c, dx, dy, color)
                        chain_score = self.convert_chain_to_score(chain_info)
                        max_chain_score = max(max_chain_score, chain_score)

                    total_score += max_chain_score
                    processed_positions.add((r, c))

        return total_score

    def convert_chain_to_score(self, chain_info):
        """将连子信息转换为分数"""
        count = chain_info['total_count']
        openings = chain_info['openings']
        corner_support = chain_info['corner_support']

        if count >= 5:
            return 10000
        elif count == 4:
            return 2000 if (openings > 0 or corner_support) else 400
        elif count == 3:
            return 300 if (openings > 0 or corner_support) else 60
        elif count == 2:
            return 40 if openings > 0 else 10
        else:
            return 0

    def corner_control_bonus(self, board, color):
        """角落战略控制奖励"""
        bonus = 0
        for corner_r, corner_c in self.CORNER_POSITIONS:
            # 检查角落周围的控制情况
            adjacent_control = 0
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                nr, nc = corner_r + dr, corner_c + dc
                if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == color:
                    adjacent_control += 1
            bonus += adjacent_control * 15  # 控制角落周围有额外价值

        return bonus

    def create_enhanced_state_signature(self, state: object, last_move: object) -> object:
        """增强的状态签名以避免重复搜索"""
        board_hash = self.get_critical_positions_hash(state.board.chips)

        return (
            board_hash,  # 棋盘关键位置哈希
            last_move.get('play_card') if last_move else None,
            tuple(sorted(state.agents[self.id].hand)) if hasattr(state.agents[self.id], 'hand') else (),
            len([cell for row in state.board.chips for cell in row if cell != '0'])  # 棋盘上的总棋子数
        )

    def score_enemy_threat(self, board, enemy_color):
        """这个函数的作用是扫描整个棋盘，识别敌方已经形成的威胁连子
            威胁越大，返回的分数越低（因为这是惩罚分）"""
        threat_score = 0
        for r in range(10):
            for c in range(10):
                if board[r][c] != enemy_color:
                    continue
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 1

                    for i in range(1, 5):
                        x, y = r + dx * i, c + dy * i
                        if 0 <= x < 10 and 0 <= y < 10:
                            if board[x][y] == enemy_color or self.is_corner_position(x, y):
                                count += 1
                            else:
                                break
                        else:
                            break

                    if count >= 3:
                        threat_score -= 50  # 威胁越大，分越低（惩罚）
        return threat_score

    def get_critical_positions_hash(self, board):
        """获取关键位置的哈希（HOTB和角落附近）"""
        critical_positions = []
        for r, c in HOTB_COORDS + self.CORNER_POSITIONS:
            if 0 <= r < 10 and 0 <= c < 10:
                critical_positions.append(board[r][c])
        return tuple(critical_positions)



class TimeManager:

    def __init__(self, total_time_limit=0.95):
        self.total_limit = total_time_limit
        self.start_time = None
        self.safety_buffer = 0.05  # 预留50毫秒作为安全缓冲

        # 定义不同决策阶段的时间分配
        self.phase_limits = {
            'quick_decision': 0.15,  # 150毫秒：快速决策作为保底
            'heuristic_search': 0.50,  # 500毫秒：主要的A*搜索
            'monte_carlo': 0.80  # 800毫秒：蒙特卡罗（如果时间允许）
        }

    def start_turn(self):
        """开始新的回合计时"""
        self.start_time = time.time()

    def get_elapsed_time(self):
        """获取已经消耗的时间"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    def get_remaining_time(self):
        """获取剩余可用时间"""
        return max(0, self.total_limit - self.safety_buffer - self.get_elapsed_time())

    def should_continue_phase(self, phase_name):
        """
        判断是否应该继续当前阶段
        """
        elapsed = self.get_elapsed_time()
        phase_limit = self.phase_limits.get(phase_name, 0.8)
        return elapsed < phase_limit

    def get_remaining_for_phase(self, phase_name):
        """获取某个阶段的剩余时间"""
        phase_limit = self.phase_limits.get(phase_name, 0.8)
        elapsed = self.get_elapsed_time()
        return max(0, phase_limit - elapsed)

    def is_emergency_mode(self):
        """检查是否进入紧急模式（时间即将耗尽）"""
        return self.get_remaining_time() < 0.1