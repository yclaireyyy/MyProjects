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

        self.mcts_config = {
            'max_iterations': 100,  # 动态调整
            'exploration_weight': 1.414,
            'simulation_depth': 10,
            'early_termination_threshold': 0.8
        }

        # 新增：对手建模
        self.opponent_model = OpponentModel()

        # 新增：开局库
        self.opening_book = OpeningBook()

        # 游戏阶段权重配置
        self.phase_weights = {
            'opening': {'center': 2.0, 'chain': 0.8, 'block': 0.5, 'corner': 1.5},
            'middle': {'center': 1.0, 'chain': 1.5, 'block': 1.2, 'corner': 1.8},
            'endgame': {'center': 0.3, 'chain': 2.0, 'block': 1.8, 'corner': 2.5}
        }

        # 蒙特卡罗模拟参数
        self.mc_simulations = 80  # 平衡计算时间和准确性

        # 选牌策略相关配置
        self.card_selection_weights = {
            'immediate_play': 3.0,  # 立即可用的价值
            'strategic_position': 2.0,  # 战略位置价值
            'blocking_value': 2.5,  # 阻断敌方的价值
            'flexibility': 1.0,  # 未来灵活性
            'opponent_denial': 2.8, # 拒绝对手获得的价值
            'opponent_expectation': 2.2
        }

        self.evaluation_weights = {
            'chain_win': 1.0,
            'chain_threat_4': 1.2,  # 优化：提高4连威胁权重
            'chain_threat_3': 1.0,
            'chain_threat_2': 1.0,
            'compound_bonus': 1.0,
            'fork_attack': 1.5,  # 优化：提高叉攻权重
            'chain_threat': 1.0,
            'block_enemy_win': 1.0,
            'block_enemy_threat': 1.0,
            'block_compound': 1.3,  # 优化：提高复合阻断权重
            'hotb_control': 1.1,  # 优化：略微提高HOTB权重
            'center_bias': 1.0,
            'corner_strategic': 1.0,
            'tempo': 0.9,  # 优化：略微降低节奏权重
            'space_control': 1.0,
            'flexibility': 1.0
        }

        self.base_weights = {
            'phase_weights': self.phase_weights,
            'card_selection_weights': self.card_selection_weights,
            'evaluation_weights': self.evaluation_weights
        }

        self.advanced_card_weights = {
            'opponent_prediction': 2.5,  # 提高对手预测权重
            'multi_threat_creation': 2.2,  # 提高多重威胁价值
            'threat_disruption': 2.8,  # 提高威胁破坏价值
            'position_synergy': 1.8,  # 提高位置协同权重
            'future_flexibility': 1.5,  # 提高未来灵活性权重
            'card_scarcity_value': 2.0  # 新增：稀缺卡片价值
        }

        # 复合威胁检测配置
        self.compound_threat_config = {
            'fork_attack_bonus': 800,  # 提高叉攻奖励
            'double_threat_bonus': 500,  # 提高双威胁奖励
            'chain_reaction_bonus': 600,  # 提高连环威胁奖励
            'cross_pattern_bonus': 400,  # 提高交叉模式奖励
            'tempo_advantage_bonus': 300  # 新增：节奏优势奖励
        }

        # 当前使用的权重（可以在游戏过程中调整）
        self.current_weights = self.deep_copy_weights(self.base_weights)

        # 权重学习系统
        self.weight_learning = WeightLearningSystem()
        self.game_statistics = GameStatistics()

        # 如果存在历史学习数据，加载优化后的权重
        self.load_optimized_weights()

        self.COORDS = COORDS

        # 启动时预计算优化
        if not self.startup_time_used:
            self._precompute_startup_data()

        # 添加缓存机制
        self.evaluation_cache = {}  # 简单的评估缓存
        self.cache_max_size = 2000  # 限制缓存大小
        self.cache_hit_count = 0
        self.cache_total_count = 0

        self.position_value_cache = {}

        self.move_history = []

    def get_cached_evaluation(self, board, action, cache_type='heuristic'):
        """获取缓存的评估结果"""
        try:
            if not action.get('coords'):
                return None

            r, c = action['coords']
            # 使用更稳定的棋盘哈希方法
            board_hash = self._get_board_hash(board)
            cache_key = (board_hash, r, c, cache_type)

            self.cache_total_count += 1
            if cache_key in self.evaluation_cache:
                self.cache_hit_count += 1
                return self.evaluation_cache[cache_key]
            return None
        except:
            return None

    def cache_evaluation(self, board, action, result, cache_type='heuristic'):
        """缓存评估结果"""
        try:
            if not action.get('coords') or result is None:
                return

            r, c = action['coords']
            board_hash = self._get_board_hash(board)
            cache_key = (board_hash, r, c, cache_type)

            # 缓存大小控制
            if len(self.evaluation_cache) >= self.cache_max_size:
                # 简单的LRU：删除一半缓存
                keys_to_remove = list(self.evaluation_cache.keys())[:self.cache_max_size // 3]
                for key in keys_to_remove:
                    del self.evaluation_cache[key]

            self.evaluation_cache[cache_key] = result
        except:
            pass

    def _get_board_hash(self, board):
        """统一的棋盘哈希方法"""
        return hash(''.join(''.join(row) for row in board))

    def load_optimized_weights(self):
        """
        加载经过优化的权重配置
        这里可以从文件读取，或从数据库加载经过训练的权重
        """
        try:
            # 模拟从文件或数据库加载优化权重的过程
            # 在实际应用中，这些权重来自于大量对局的统计分析
            optimized_adjustments = {
                'evaluation_weights': {
                    'chain_threat_4': 1.5,
                    'fork_attack': 1.8,
                    'block_compound': 1.5,
                    'hotb_control': 1.3,
                    'tempo': 0.8,
                    'opponent_disruption': 1.4
                }
            }

            # 应用优化调整
            self.apply_weight_adjustments(optimized_adjustments)

        except Exception as e:
            pass  # 如果加载失败，使用默认权重

    def apply_weight_adjustments(self, adjustments):
        """应用权重调整"""
        for category, weights in adjustments.items():
            if category in self.current_weights:
                for weight_name, adjustment in weights.items():
                    if weight_name in self.current_weights[category]:
                        self.current_weights[category][weight_name] *= adjustment

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

        self._precompute_opening_positions()

        self.startup_time_used = True

    def _precompute_opening_positions(self):
        """预计算开局最佳位置"""
        # 开局优先位置：HOTB中心 + 角落附近战略位置
        self.opening_priority_positions = [
            (4, 4), (4, 5), (5, 4), (5, 5),  # HOTB
            (3, 3), (3, 6), (6, 3), (6, 6),  # HOTB扩展
            (1, 1), (1, 8), (8, 1), (8, 8),  # 角落附近
            (2, 2), (2, 7), (7, 2), (7, 7)  # 次级战略位置
        ]

    def _calculate_base_position_value(self, r, c):
        """预计算位置的基础战略价值"""
        value = 0

        # HOTB区域高价值
        if (r, c) in HOTB_COORDS:
            value += 120  # 提高HOTB价值

        # 中心区域价值（优化）
        distance_to_center = abs(r - 4.5) + abs(c - 4.5)
        value += max(0, 25 - distance_to_center * 2.5)

        # 角落邻近价值（优化）
        for corner_r, corner_c in self.CORNER_POSITIONS:
            corner_distance = max(abs(r - corner_r), abs(c - corner_c))
            if corner_distance <= 3:
                value += max(0, 20 - corner_distance * 5)

        return value

    def _precompute_threat_patterns(self):
        """预计算威胁模式"""
        self.threat_patterns = {
            'win_pattern': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
            'threat_4': [(0, 0), (0, 1), (0, 2), (0, 3)],
            'threat_3': [(0, 0), (0, 1), (0, 2)],
            'threat_2': [(0, 0), (0, 1)]
        }

    def classify_threat(self, count, openings, corner_support):
        """威胁分类"""
        if count >= 5:
            return 'win'
        elif count >= 4:
            if openings >= 2 or (openings >= 1 and corner_support):
                return 'critical_win'  # 必胜威胁
            elif openings >= 1 or corner_support:
                return 'major_threat'  # 主要威胁
            else:
                return 'blocked_threat'  # 被阻威胁
        elif count >= 3:
            if openings >= 2:
                return 'double_threat'  # 双端威胁
            elif openings >= 1 or corner_support:
                return 'active_threat'  # 活跃威胁
            else:
                return 'potential_threat'  # 潜在威胁
        elif count >= 2:
            if openings >= 2:
                return 'development'  # 发展性
            else:
                return 'weak_connection'  # 弱连接
        else:
            return 'none'

    def calculate_threat_urgency(self, threat_type, count, openings):
        """计算威胁紧迫性"""
        urgency_map = {
            'win': 10000,
            'critical_win': 9000,
            'major_threat': 7000,
            'double_threat': 5000,
            'active_threat': 3000,
            'blocked_threat': 1500,
            'potential_threat': 800,
            'development': 400,
            'weak_connection': 100,
            'none': 0
        }
        return urgency_map.get(threat_type, 0)

    def is_corner_position(self, r, c):
        """检查位置是否为角落"""
        return (r, c) in self.CORNER_POSITIONS

    def SelectAction(self, actions, game_state):
        self.turn_count += 1

        # 如果仍在启动时间内，使用剩余启动时间
        if not self.startup_time_used:
            startup_elapsed = time.time() - self.startup_start_time
            remaining_startup = self.startup_time_limit - startup_elapsed
            if remaining_startup > 0.2:  # 只有足够时间才使用
                time_limit = min(remaining_startup - 0.1, 0.98)  # 预留安全缓冲
            else:
                time_limit = 0.98
                self.startup_time_used = True
        else:
            time_limit = 0.98

        # 初始化时间管理器
        self.time_manager = TimeManager(time_limit)
        self.time_manager.start_turn()

        if time_limit < 0.05:  # 时间极少时
            return self.emergency_quick_decision(actions, game_state)

        action_type = self.identify_action_type(actions, game_state)
        if action_type == 'card_selection':
            # 选牌阶段：从5张明牌中选择最优的一张
            return self.select_card_strategy(actions, game_state)
        elif action_type == 'place':
            # 放牌阶段：选择最优的放置位置
            return self.place_strategy(actions, game_state)
        else:
            # 默认策略：如果无法识别类型，使用原始放牌策略
            return self.place_strategy(actions, game_state)

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

        agent, board, is_valid = self.safe_get_game_state_info(game_state)
        if not is_valid:
            return card_actions[0]

        my_color = agent.colour
        enemy_color = 'r' if my_color == 'b' else 'b'

        # 分析当前局面
        game_phase = self.determine_game_phase(board)
        my_threats = self.analyze_all_threats(board, my_color)
        enemy_threats = self.analyze_all_threats(board, enemy_color)

        best_card = None
        best_score = float('-inf')

        for card_action in card_actions:
            score = 0
            card_positions = self.get_card_board_positions(card_action, game_state)

            if not card_positions:
                continue

            # 1. 胜利检查 - 最高优先级
            win_potential = self.check_immediate_win_potential(
                card_positions, board, my_color)
            if win_potential:
                return card_action  # 直接返回能获胜的牌

            # 2. 阻止对手获胜 - 次高优先级
            block_enemy_win = self.check_block_enemy_win_potential(
                card_positions, board, enemy_color)
            score += block_enemy_win * 10000

            # 3. 创造关键威胁
            threat_creation = self.evaluate_threat_creation_potential(
                card_positions, board, my_color, game_phase)
            score += threat_creation

            # 4. 阻断对手威胁
            threat_blocking = self.evaluate_threat_blocking_potential(
                card_positions, board, enemy_color)
            score += threat_blocking

            # 5. 对手拒绝价值 - 阻止对手获得这张牌
            denial_value = self.calculate_opponent_card_denial_value(
                card_positions, board, enemy_color, enemy_threats)
            score += denial_value

            # 6. 位置灵活性和发展潜力
            flexibility_value = self.calculate_strategic_flexibility(
                card_positions, board, my_color, game_phase)
            score += flexibility_value

            if score > best_score:
                best_score = score
                best_card = card_action

        return best_card or card_actions[0]

    def evaluate_card_value(self, card_action, board, my_color, enemy_color, game_state):
        """
        评估单张牌的战略价值,考虑对手阻断
        """
        total_score = 0
        start_time = time.time()
        time_budget = 0.12

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

            # 高级分析（如果时间允许才执行）
            if time.time() - start_time < time_budget * 0.6:  # 60%时间用完前
                # 对手选牌预测分析
                opponent_prediction_value = self.analyze_opponent_card_preference(
                    card_action, card_positions, board, my_color, enemy_color, game_state
                )
                total_score += opponent_prediction_value * self.advanced_card_weights['opponent_prediction']

            if time.time() - start_time < time_budget * 0.8:  # 80%时间用完前
                # 多重威胁创造分析
                multi_threat_value = self.analyze_multi_threat_creation(card_positions, board, my_color)
                total_score += multi_threat_value * self.advanced_card_weights['multi_threat_creation']

            if time.time() - start_time < time_budget * 0.9:  # 90%时间用完前
                # 威胁链破坏分析
                threat_disruption_value = self.analyze_threat_disruption(card_positions, board, enemy_color)
                total_score += threat_disruption_value * self.advanced_card_weights['threat_disruption']

            if time.time() - start_time < time_budget:  # 时间允许的话
                # 位置协同效应分析
                synergy_value = self.analyze_position_synergy(card_positions, board, my_color)
                total_score += synergy_value * self.advanced_card_weights['position_synergy']

        except Exception:
            return 0

        return total_score

    def analyze_opponent_card_preference(self, card_action, positions, board, my_color, enemy_color, game_state):
        """分析对手对这张牌的渴望程度"""
        opponent_desire = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 评估对手在此位置的复合收益

                # 检查对手能否形成叉攻
                fork_potential = self.check_opponent_fork_potential(board, r, c, enemy_color)
                opponent_desire += fork_potential

                # 检查对手能否破坏我方计划
                disruption_potential = self.check_opponent_disruption_potential(board, r, c, my_color, enemy_color)
                opponent_desire += disruption_potential

                # 检查对手的战略布局价值
                strategic_layout_value = self.check_opponent_strategic_layout(board, r, c, enemy_color)
                opponent_desire += strategic_layout_value

        return opponent_desire

    def check_opponent_fork_potential(self, board, r, c, enemy_color):
        """检查对手在此位置能否形成叉攻"""
        temp_board = [row[:] for row in board]
        temp_board[r][c] = enemy_color

        # 计算对手在此位置能同时推进的威胁方向数
        threat_directions = 0
        threat_levels = []

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count, openings = self.analyze_chain_pattern(temp_board, r, c, dx, dy, enemy_color)
            corner_support = self.check_corner_support(temp_board, r, c, dx, dy, enemy_color)

            # 评估威胁等级
            if count >= 3 and (openings >= 1 or corner_support):
                threat_directions += 1
                threat_levels.append(count)
            elif count >= 2 and openings >= 2:  # 活二也有威胁潜力
                threat_directions += 0.5
                threat_levels.append(count)

        # 如果对手能形成多重威胁，这张牌对他们价值很高
        if threat_directions >= 2:
            fork_value = threat_directions * 400
            # 如果包含高等级威胁，进一步加分
            if any(level >= 3 for level in threat_levels):
                fork_value += 600
            return fork_value

        return 0

    def check_opponent_disruption_potential(self, board, r, c, my_color, enemy_color):
        """检查对手在此位置能否有效破坏我方威胁"""
        disruption_value = 0

        # 临时模拟对手在此位置放子
        temp_board = [row[:] for row in board]
        temp_board[r][c] = enemy_color

        # 检查这是否阻断了我方的威胁线
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # 检查正负两个方向上我方的连子情况
            my_threat_blocked = self.calculate_threat_blocking_impact(
                board, temp_board, r, c, dx, dy, my_color
            )
            disruption_value += my_threat_blocked

        return disruption_value

    def calculate_threat_blocking_impact(self, original_board, blocked_board, r, c, dx, dy, my_color):
        """计算阻断对我方威胁的影响"""
        impact = 0

        # 检查沿此方向的我方威胁被阻断程度
        for direction in [1, -1]:
            pieces_affected = 0
            potential_extension = 0

            # 沿方向查看我方棋子和潜在发展
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if original_board[x][y] == my_color:
                        pieces_affected += 1
                    elif original_board[x][y] == '0':
                        potential_extension += 1
                        break
                    else:
                        break
                else:
                    break

            # 根据被阻断的威胁程度评分
            if pieces_affected >= 2:
                impact += pieces_affected * 150 + potential_extension * 50

        return impact

    def check_opponent_strategic_layout(self, board, r, c, enemy_color):
        """检查对手的战略布局价值"""
        strategic_value = 0

        # 检查是否能连接对手已有的棋子群
        connection_value = 0
        connected_groups = 0

        # 搜索周围的对手棋子群
        for dr, dc in [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == enemy_color:
                # 检查是否能与这个棋子形成有效连线
                if self.can_form_strategic_connection(r, c, nr, nc, board, enemy_color):
                    connected_groups += 1
                    connection_value += 80

        strategic_value += connection_value

        # 连接多个棋子群有额外奖励
        if connected_groups >= 2:
            strategic_value += connected_groups * 100

        return strategic_value

    def can_form_strategic_connection(self, r1, c1, r2, c2, board, color):
        """检查两点是否能形成有效的战略连接"""
        # 检查是否在同一条线上（水平、垂直或对角线）
        dr = r2 - r1
        dc = c2 - c1

        # 标准化方向
        if dr != 0:
            dr = dr // abs(dr)
        if dc != 0:
            dc = dc // abs(dc)

        # 检查连线上是否有阻挡
        current_r, current_c = r1 + dr, c1 + dc
        while (current_r, current_c) != (r2, c2):
            if 0 <= current_r < 10 and 0 <= current_c < 10:
                if board[current_r][current_c] != '0' and board[current_r][current_c] != color:
                    return False  # 被敌方棋子阻断
            current_r += dr
            current_c += dc

        return True

    def analyze_multi_threat_creation(self, positions, board, my_color):
        """分析选择这张牌能否为我方创造多重威胁"""
        multi_threat_value = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 模拟在此位置放子
                temp_board = [row[:] for row in board]
                temp_board[r][c] = my_color

                # 检查能同时推进的威胁方向
                active_threats = []
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count, openings = self.analyze_chain_pattern(temp_board, r, c, dx, dy, my_color)
                    corner_support = self.check_corner_support(temp_board, r, c, dx, dy, my_color)

                    # 记录有效威胁
                    if count >= 2 and (openings >= 1 or corner_support):
                        threat_level = self.calculate_threat_level(count, openings, corner_support)
                        active_threats.append({
                            'direction': (dx, dy),
                            'level': threat_level,
                            'count': count
                        })

                # 评估多重威胁的价值
                if len(active_threats) >= 2:
                    # 基础多重威胁奖励
                    multi_threat_value += len(active_threats) * self.compound_threat_config['double_threat_bonus']

                    # 检查是否为叉攻（包含强威胁）
                    strong_threats = [t for t in active_threats if t['count'] >= 3]
                    if len(strong_threats) >= 2:
                        multi_threat_value += self.compound_threat_config['fork_attack_bonus']

                    # 检查威胁方向的互补性
                    complementary_bonus = self.calculate_threat_complementarity(active_threats)
                    multi_threat_value += complementary_bonus

        return multi_threat_value

    def calculate_threat_level(self, count, openings, corner_support):
        """计算威胁等级"""
        if count >= 4:
            return 4
        elif count >= 3:
            return 3 if (openings >= 1 or corner_support) else 2
        elif count >= 2:
            return 2 if openings >= 1 else 1
        else:
            return 0

    def calculate_threat_complementarity(self, threats):
        """计算威胁方向间的互补性"""
        bonus = 0

        # 检查是否存在互补的威胁方向组合
        directions = [t['direction'] for t in threats]

        # 十字交叉威胁（水平+垂直）
        if (0, 1) in directions and (1, 0) in directions:
            bonus += self.compound_threat_config['cross_pattern_bonus']

        # X型交叉威胁（两条对角线）
        if (1, 1) in directions and (1, -1) in directions:
            bonus += self.compound_threat_config['cross_pattern_bonus']

        # 三重或四重威胁额外奖励
        if len(threats) >= 3:
            bonus += (len(threats) - 2) * 200

        return bonus

    # 新增方法3：威胁链破坏分析
    def analyze_threat_disruption(self, positions, board, enemy_color):
        """分析选择这张牌能否破坏对手的威胁链"""
        disruption_value = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 评估在此位置放子对对手威胁网络的破坏程度

                # 检查直接威胁阻断
                direct_blocking = self.calculate_enemy_threat_in_direction(board, r, c, enemy_color)
                disruption_value += direct_blocking

                # 检查威胁网络分割
                network_disruption = self.calculate_threat_network_disruption(board, r, c, enemy_color)
                disruption_value += network_disruption

                # 检查关键节点占领
                key_position_value = self.calculate_key_position_occupation(board, r, c, enemy_color)
                disruption_value += key_position_value

        return disruption_value

    def calculate_threat_network_disruption(self, board, r, c, enemy_color):
        """计算对手威胁网络分割的价值"""
        # 检查这个位置是否是多条威胁线的交汇点
        intersection_value = 0
        threat_lines_affected = 0

        # 检查通过此位置的威胁线数量
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            line_threat_level = self.calculate_line_threat_level(board, r, c, dx, dy, enemy_color)
            if line_threat_level > 0:
                threat_lines_affected += 1
                intersection_value += line_threat_level * 150

        # 如果影响多条威胁线，有额外的网络分割奖励
        if threat_lines_affected >= 2:
            intersection_value += threat_lines_affected * 300

        return intersection_value

    def calculate_line_threat_level(self, board, r, c, dx, dy, enemy_color):
        """计算穿过指定位置的威胁线等级"""
        total_enemy_pieces = 0

        # 检查正负两个方向
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if board[x][y] == enemy_color:
                        total_enemy_pieces += 1
                    elif board[x][y] != '0':
                        break
                else:
                    break

        return total_enemy_pieces

    def calculate_key_position_occupation(self, board, r, c, enemy_color):
        """计算占领关键位置的价值"""
        key_value = 0

        # HOTB区域的战略价值
        if (r, c) in HOTB_COORDS:
            # 检查对手在HOTB的布局
            enemy_hotb_count = sum(1 for hr, hc in HOTB_COORDS if board[hr][hc] == enemy_color)
            if enemy_hotb_count >= 2:  # 对手已有HOTB布局
                key_value += 400

        # 角落附近的战略价值
        for corner_r, corner_c in self.CORNER_POSITIONS:
            distance = max(abs(r - corner_r), abs(c - corner_c))
            if distance <= 2:
                # 检查对手在此角落的威胁
                corner_threat = self.evaluate_corner_threat_level(board, corner_r, corner_c, enemy_color)
                key_value += corner_threat * max(0, 200 - distance * 50)

        return key_value

    def evaluate_corner_threat_level(self, board, corner_r, corner_c, enemy_color):
        """评估对手在角落的威胁等级"""
        threat_level = 0

        # 检查角落周围的对手布局
        for dr, dc in [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = corner_r + dr, corner_c + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == enemy_color:
                threat_level += 1

        return threat_level

    # 新增方法4：位置协同效应分析
    def analyze_position_synergy(self, positions, board, my_color):
        """分析位置间的协同效应"""
        synergy_value = 0

        if len(positions) < 2:
            return 0

        # 检查位置间的相互支撑
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1:]:
                if board[pos1[0]][pos1[1]] == '0' and board[pos2[0]][pos2[1]] == '0':
                    # 计算两个位置的协同价值
                    pair_synergy = self.calculate_position_pair_synergy(pos1, pos2, board, my_color)
                    synergy_value += pair_synergy

        return synergy_value

    def calculate_position_pair_synergy(self, pos1, pos2, board, my_color):
        """计算两个位置间的协同价值"""
        r1, c1 = pos1
        r2, c2 = pos2

        synergy = 0

        # 检查是否在同一条潜在连线上
        if self.positions_on_same_line(r1, c1, r2, c2):
            distance = max(abs(r1 - r2), abs(c1 - c2))
            if distance <= 4:  # 在5连范围内
                synergy += max(0, 100 - distance * 20)

        # 检查相互支援的威胁创造
        support_value = self.calculate_mutual_threat_support(pos1, pos2, board, my_color)
        synergy += support_value

        return synergy

    def positions_on_same_line(self, r1, c1, r2, c2):
        """检查两个位置是否在同一条直线上"""
        return (r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2))

    def calculate_mutual_threat_support(self, pos1, pos2, board, my_color):
        """计算相互威胁支援价值"""
        r1, c1 = pos1
        r2, c2 = pos2

        # 模拟两个位置都放子
        temp_board = [row[:] for row in board]
        temp_board[r1][c1] = my_color
        temp_board[r2][c2] = my_color

        # 计算组合威胁效果
        combined_threats = 0

        # 检查每个位置的威胁在另一个位置支援下的增强
        for r, c in [pos1, pos2]:
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count, openings = self.analyze_chain_pattern(temp_board, r, c, dx, dy, my_color)
                if count >= 3:
                    combined_threats += count * 50

        return combined_threats

    def calculate_opponent_expectation_value(self, card_action, positions, board, my_color, enemy_color, game_state):
        """计算对手对这张牌的期望价值（轻量级）"""
        opponent_benefit = 0

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 快速评估对手在此位置的收益
                opponent_threat = 0

                # 检查对手的连子潜力
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    temp_board = [row[:] for row in board]
                    temp_board[r][c] = enemy_color
                    count, openings = self.analyze_chain_pattern(temp_board, r, c, dx, dy, enemy_color)

                    if count >= 4:
                        opponent_threat += 1000
                    elif count >= 3:
                        opponent_threat += 300 if openings >= 1 else 100
                    elif count >= 2:
                        opponent_threat += 50 if openings >= 1 else 10

                # 检查战略位置价值
                if (r, c) in HOTB_COORDS:
                    opponent_threat += 80

                opponent_benefit += opponent_threat

        return opponent_benefit

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

    def evaluate_card_scarcity(self, card_action, game_state):
        """
        评估卡片稀缺性价值 - 新增方法
        基于已使用卡片数量评估剩余卡片的稀缺性
        """
        try:
            card_name = self._extract_card_name_from_action(card_action)
            if not card_name:
                return 0

            # 简单的稀缺性评估：检查该卡片在棋盘上的使用情况
            board = game_state.board.chips if game_state else None
            if not board:
                return 0

            # 计算该卡片对应位置的占用情况
            card_positions = self.COORDS.get(card_name, [])
            if not card_positions:
                return 0

            occupied_count = 0
            total_positions = len(card_positions)

            for r, c in card_positions:
                if 0 <= r < 10 and 0 <= c < 10 and board[r][c] != '0':
                    occupied_count += 1

            # 剩余位置越少，稀缺性越高
            remaining_positions = total_positions - occupied_count
            if remaining_positions <= 0:
                return 0

            # 稀缺性评分：位置越少分数越高
            scarcity_score = max(0, (total_positions - remaining_positions) * 20)

            # 特殊卡片（如Jack）有额外稀缺性价值
            if card_name == 'jk':
                scarcity_score += 30
            elif card_name in ['5h', '4h', '2h', '3h']:  # HOTB卡片
                scarcity_score += 15

            return scarcity_score

        except Exception:
            return 0

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
                    chain_potential = self.analyze_chain_for_evaluation(
                        board, r, c, dx, dy, my_color)
                    position_value += chain_potential

                value += position_value

        return value

    def analyze_chain_for_evaluation(self, board, r, c, dx, dy, color, quick_mode=False):
        """
        连子评估
        """
        if quick_mode:
            # 原 quick_chain_evaluation 的逻辑
            temp_board = [row[:] for row in board]
            temp_board[r][c] = color
            count = 1
            for direction in [1, -1]:
                for i in range(1, 5):
                    x, y = r + dx * direction * i, c + dy * direction * i
                    if 0 <= x < 10 and 0 <= y < 10:
                        if (temp_board[x][y] == color or self.is_corner_position(x, y)):
                            count += 1
                        else:
                            break
                    else:
                        break
            # 返回简化评分
            if count >= 4:
                return 1000
            elif count >= 3:
                return 300
            elif count >= 2:
                return 100
            else:
                return 10
        else:
            # 原 simple_chain_analysis_for_scoring 的逻辑
            count, openings = self.analyze_chain_pattern(board, r, c, dx, dy, color)
            corner_support = self.check_corner_support(board, r, c, dx, dy, color)
            return {
                'total_count': count,
                'openings': openings,
                'corner_support': corner_support
            }

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
                    threat_level = self.evaluate_opponent_chain_at_position(
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

    def get_quick_decision(self, actions, game_state):
        """
        快速决策
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

        # 开局检查
        if self.turn_count <= 6:
            for action in actions[:min(len(actions), 5)]:
                if not action.get('coords'):
                    continue
                r, c = action['coords']
                if (r, c) in self.opening_priority_positions:
                    return action

        for action in actions[:min(len(actions), 6)]:  # 增加候选数量
            if not action.get('coords'):
                continue

            r, c = action['coords']
            score = 0

            # 快速评估
            if (r, c) in HOTB_COORDS:
                score += 150  # 提高HOTB价值
            elif 3 <= r <= 6 and 3 <= c <= 6:
                score += 80  # 提高中心价值

            # 快速连子检查
            temp_board = [row[:] for row in board]
            temp_board[r][c] = my_color

            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count = 1
                for direction in [1, -1]:
                    for i in range(1, 5):
                        x, y = r + dx * direction * i, c + dy * direction * i
                        if (0 <= x < 10 and 0 <= y < 10 and
                                (temp_board[x][y] == my_color or self.is_corner_position(x, y))):
                            count += 1
                        else:
                            break

                if count >= 5:
                    return action  # 直接获胜
                elif count >= 4:
                    score += 2000
                elif count >= 3:
                    score += 500

            # 快速阻断检查
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                enemy_temp = [row[:] for row in board]
                enemy_temp[r][c] = enemy_color

                enemy_count = 1
                for direction in [1, -1]:
                    for i in range(1, 5):
                        x, y = r + dx * direction * i, c + dy * direction * i
                        if (0 <= x < 10 and 0 <= y < 10 and
                                (enemy_temp[x][y] == enemy_color or self.is_corner_position(x, y))):
                            enemy_count += 1
                        else:
                            break

                if enemy_count >= 4:
                    score += 1500  # 阻止敌方获胜

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

        # 启发式函数进行初始排序
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

            #更精确的状态签名
            state_signature = self.create_state_signature(current_state, last_move)
            if state_signature in seen_states:
                continue
            seen_states.add(state_signature)

            # 使用增强的状态评估
            reward = self.evaluate_state(current_state, last_move)
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history

            # 智能剪枝 - 根据当前最佳分数决定扩展深度 - 只在时间和深度都允许的情况下扩展
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

    def monte_carlo_validation(self, initial_state, candidate_moves):
        if not candidate_moves:
            return None

        # 动态调整模拟次数
        remaining_time = self.time_manager.get_remaining_for_phase('monte_carlo')
        max_iterations = min(self.mcts_config['max_iterations'], int(remaining_time * 200))  # 动态调整

        # 为每个候选动作创建根节点
        action_scores = {}

        for action in candidate_moves:
            if not self.time_manager.should_continue_phase('monte_carlo'):
                break

            # 为这个动作创建MCTS树
            simulated_state = self.fast_simulate(initial_state, action)
            root = MCTSNode(simulated_state, action)

            root.player_id = self.id
            legal_actions = self.rule.getLegalActions(simulated_state, self.id)
            root.untried_actions = legal_actions.copy() if legal_actions else []

            # 执行MCTS迭代
            iterations_per_action = max_iterations // len(candidate_moves)
            for _ in range(iterations_per_action):
                if not self.time_manager.should_continue_phase('monte_carlo'):
                    break

                self.mcts_iteration(root)

            # 计算最终分数
            if root.visits > 0:
                action_scores[action] = root.wins / root.visits
            else:
                action_scores[action] = 0

        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]

        return candidate_moves[0]

    def mcts_iteration(self, root):
        """单次MCTS迭代：选择、扩展、模拟、回传"""
        # 1. 选择
        node = self.mcts_select(root)

        # 2. 扩展
        if not node.is_terminal() and node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state = self.fast_simulate(node.state, action)
            node = node.add_child(action, new_state)

            other_player = 1 - node.parent.player_id if node.parent.player_id in [0, 1] else (
                1 if node.parent.player_id == 0 else 0)
            legal_actions = self.rule.getLegalActions(new_state, other_player)
            node.untried_actions = legal_actions.copy() if legal_actions else []

        # 3. 模拟
        result = self.mcts_simulate(node)

        # 4. 回传
        self.mcts_backpropagate(node, result)

    def mcts_select(self, root):
        """MCTS选择阶段"""
        node = root
        while node.children and node.is_fully_expanded():
            node = node.select_child()
        return node

    def mcts_simulate(self, node):
        """MCTS模拟阶段"""
        current_state = node.state
        current_player = node.player_id
        depth = 0
        max_depth = self.mcts_config['simulation_depth']

        while depth < max_depth:
            if self.is_game_over(current_state):
                break

            actions = self.rule.getLegalActions(current_state, current_player)
            if not actions:
                break

            # 使用启发式选择而非完全随机
            if len(actions) <= 3:
                action = actions[0]
            else:
                # 选择启发式评分较高的动作
                action = max(actions[:min(5, len(actions))],
                             key=lambda a: self.quick_heuristic(current_state, a))

            current_state = self.fast_simulate(current_state, action)
            current_player = 1 - current_player if current_player in [0, 1] else (1 if current_player == 0 else 0)
            depth += 1

        # 评估最终状态
        final_score = self.evaluate_state(current_state, None)

        # 归一化结果到[0,1]
        if final_score > 1000:
            return 1.0
        elif final_score < -1000:
            return 0.0
        else:
            return (final_score + 1000) / 2000.0

    def mcts_backpropagate(self, node, result):
        """MCTS回传阶段"""
        while node is not None:
            node.update(result)
            node = node.parent

    def quick_heuristic(self, state, action):
        """快速启发式评估（用于MCTS模拟）"""
        if not action.get('coords'):
            return 0

        r, c = action['coords']

        # 简单的位置价值评估
        score = 0
        if (r, c) in HOTB_COORDS:
            score += 50
        elif 3 <= r <= 6 and 3 <= c <= 6:
            score += 20

        return score

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
            if len(possible_actions) <= 3:
                chosen_action = possible_actions[0]  # 选择第一个
            else:
                # 从前几个动作中随机选择
                top_actions = possible_actions[:min(3, len(possible_actions))]
                chosen_action = random.choice(top_actions)

            current_state = self.fast_simulate(current_state, chosen_action)

        final_reward = self.evaluate_state(current_state, None)

        return (total_reward + final_reward * 0.3) / 1000.0  # 归一化到[0,1]

    def calculate_tempo_value(self, board, r, c, color, enemy_color):
        """计算节奏价值"""
        tempo_value = 0

        # 检查是否能迫使对手防守
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count, openings = self.analyze_chain_pattern(board, r, c, dx, dy, color)
            if count >= 3 and openings >= 1:
                tempo_value += 200
                break
        return tempo_value

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

    def deep_copy_weights(self, weights):
        """深拷贝权重配置"""
        import copy
        return copy.deepcopy(weights)

    def heuristic(self, state, action):
        if action.get('type') != 'place' or not action.get('coords'):
            return 1000

        # 检查缓存
        board = state.board.chips
        cached_result = self.get_cached_evaluation(board, action, 'heuristic')
        if cached_result is not None:
            return cached_result

        r, c = action['coords']
        board = [row[:] for row in state.board.chips]
        me = state.agents[self.id]
        color = me.colour
        enemy = 'r' if color == 'b' else 'b'
        # 临时放置棋子进行评估
        board[r][c] = color
        # 获取当前游戏阶段的权重
        phase_weights = self.get_current_phase_weights(board)
        eval_weights = self.current_weights['evaluation_weights']
        score = 0
        # 应用动态权重的评分
        center_score = self.center_bias(r, c, board, color)
        score += center_score * phase_weights['center'] * eval_weights['center_bias']

        chain_score = self.chain_score(board, r, c, color)
        score += chain_score * phase_weights['chain'] * eval_weights.get('chain_threat_4', 1.0)

        block_score = self.block_enemy_score(board, r, c, enemy)
        score += block_score * phase_weights['block'] * eval_weights['block_enemy_threat']

        hotb_score = self.hotb_score(board, color)
        score += hotb_score * eval_weights['hotb_control']

        corner_score = self.corner_strategic_value(r, c, board, color)
        score += corner_score * phase_weights['corner'] * eval_weights['corner_strategic']

        # 节奏和空间控制评估
        tempo_score = self.calculate_tempo_value(board, r, c, color, enemy)
        score += tempo_score * eval_weights['tempo']
        space_control_score = self.calculate_space_control_value(board, r, c, color)
        score += space_control_score * eval_weights['space_control']

        # 对手破坏评估
        opponent_disruption_score = self.calculate_opponent_disruption_value(board, r, c, color, enemy)
        score += opponent_disruption_score * eval_weights.get('opponent_disruption', 1.0)

        result = max(1, 1000 - score)

        # 缓存结果
        self.cache_evaluation(state.board.chips, action, result, 'heuristic')

        return result

    def calculate_space_control_value(self, board, r, c, color):
        """
        计算空间控制价值：这个位置能控制多少周围空间
        """
        control_value = 0
        controlled_spaces = 0

        # 计算能影响的空白位置数量
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == '0':
                    # 距离越近，控制力越强
                    distance = max(abs(dr), abs(dc))
                    control_strength = max(0, 3 - distance)
                    controlled_spaces += control_strength

        control_value = controlled_spaces * 15

        # 检查是否控制了关键通道
        if self.controls_key_passages(board, r, c, color):
            control_value += 150  # 提高关键通道价值

        return control_value

    def controls_key_passages(self, board, r, c, color):
        """检查是否控制了关键通道（连接重要区域的路径）"""
        # 检查是否在HOTB区域之间的连接路径上
        hotb_connections = 0
        for hr, hc in HOTB_COORDS:
            if abs(r - hr) <= 2 and abs(c - hc) <= 2:
                hotb_connections += 1
        # 检查角落连接
        corner_connections = 0
        for corner_r, corner_c in self.CORNER_POSITIONS:
            if max(abs(r - corner_r), abs(c - corner_c)) <= 3:
                corner_connections += 1

        return hotb_connections >= 2 or corner_connections >= 1

    def center_bias(self, r, c, board=None, color=None):
        # 使用预计算的基础价值
        base_score = self.position_values.get((r, c), 0)

        # 动态邻近HOTB加分
        if board is not None and color is not None:
            # 动态HOTB价值评估
            if (r, c) in self.hotb_adjacent:
                hotb_control = sum(1 for hr, hc in HOTB_COORDS if board[hr][hc] == color)
                hotb_potential = sum(1 for hr, hc in HOTB_COORDS if board[hr][hc] == '0')

                # 根据HOTB控制情况调整价值
                if hotb_control >= 2:
                    base_score += 30  # 已有优势，继续扩大
                elif hotb_potential >= 3:
                    base_score += 20  # 有扩展潜力
                else:
                    base_score += 10  # 基础价值

        return base_score

    # 指数级评分系统
    def chain_score(self, board, r, c, color):
        total_score = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count, openings = self.analyze_chain_pattern(board, r, c, dx, dy, color)

            corner_support = self.check_corner_support(board, r, c, dx, dy, color)

            # 指数级评分逻辑 - 关键差异被放大
            if count >= 5:
                total_score += 100000  # 胜利局面：压倒性优势
            elif count == 4:
                if openings == 2:  # 双端开放的活四
                    total_score += 50000  # 必胜局面
                elif corner_support and openings >= 1:
                    total_score += 30000 # 角落支持的活四：非常强的威胁
                elif openings == 1:  # 单端开放的冲四,强威胁，但可防守
                    total_score += 20000
                elif corner_support:
                    total_score += 15000
                else:  # 完全被堵的死四
                    total_score += 3000  # 几乎无用
            elif count == 3:
                if openings == 2:  # 活三
                    total_score += 2000  # 重要进攻点
                elif corner_support and openings >= 1:
                    total_score += 1500   # 角落支持的活三：优质发展
                elif openings == 1:
                    # 单端开放的三连：一般发展
                    total_score += 600
                elif corner_support:
                    # 纯角落支持的三连
                    total_score += 800
                else:
                    # 死三：价值很低
                    total_score += 100
            elif count == 2:
                if openings >= 2:
                    # 双端开放的活二：有发展空间
                    total_score += 200
                elif corner_support and openings >= 1:
                    # 角落支持的活二
                    total_score += 150
                elif openings >= 1:
                    # 单端开放的二连
                    total_score += 80
                elif corner_support:
                    # 角落支持的二连
                    total_score += 100
                else:
                    # 死二：基本无价值
                    total_score += 20
            else:
                if corner_support:
                    total_score += 30  # 角落附近的单子有潜在价值
                else:
                    total_score += 10  # 普通单子的基础分

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
        if not hasattr(self, 'corner_influence_zones'):
            return 0
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

    def check_immediate_win_potential(self, positions, board, color):
        """检查是否能立即获胜"""
        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                temp_board = [row[:] for row in board]
                temp_board[r][c] = color

                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count, _ = self.analyze_chain_pattern(temp_board, r, c, dx, dy, color)
                    if count >= 5:
                        return True
        return False

    def block_enemy_score(self, board, r, c, enemy_color):
        score = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            enemy_threat = self.calculate_enemy_threat_in_direction(
                board, r, c, dx, dy, enemy_color)

            if enemy_threat >= 5:
                score += 15000  # 阻止敌方获胜
            elif enemy_threat >= 4:
                score += 8000  # 阻止敌方活四
            elif enemy_threat >= 3:
                score += 2000  # 阻止敌方活三
            elif enemy_threat >= 2:
                score += 500  # 阻止敌方发展

        return score

    def check_block_enemy_win_potential(self, positions, board, enemy_color):
        """检查是否能阻止对手获胜"""
        block_value = 0
        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                # 模拟对手在此位置放子
                temp_board = [row[:] for row in board]
                temp_board[r][c] = enemy_color

                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count, _ = self.analyze_chain_pattern(temp_board, r, c, dx, dy, enemy_color)
                    if count >= 5:
                        block_value += 15000  # 阻止获胜
                    elif count >= 4:
                        block_value += 8000  # 阻止关键威胁
        return block_value

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

    def calculate_enhanced_enemy_threat(self, board, r, c, dx, dy, enemy_color):
        """计算增强的敌方威胁等级"""
        # 模拟敌方在此位置放子
        temp_board = [row[:] for row in board]
        temp_board[r][c] = enemy_color

        count, openings = self.analyze_chain_pattern(temp_board, r, c, dx, dy, enemy_color)
        corner_support = self.check_corner_support(temp_board, r, c, dx, dy, enemy_color)

        # 威胁等级评估
        threat_level = count
        if openings >= 2:
            threat_level += 1  # 活连更危险
        if corner_support:
            threat_level += 0.5  # 角落支持增加威胁

        return threat_level

    def determine_game_phase(self, board):
        """确定游戏阶段"""
        total_pieces = sum(1 for row in board for cell in row if cell != '0')
        if total_pieces < 12:
            return 'opening'
        elif total_pieces < 35:
            return 'middle'
        else:
            return 'endgame'

    def analyze_all_threats(self, board, color):
        """分析棋盘上所有威胁"""
        threats = []
        for r in range(10):
            for c in range(10):
                if board[r][c] == color:
                    position_threats = self.threat_analysis(board, r, c, color,
                                                            'r' if color == 'b' else 'b')
                    threats.extend(position_threats)
        return threats

    def evaluate_threat_creation_potential(self, positions, board, color, game_phase):
        """评估威胁创造潜力"""
        total_value = 0
        phase_multiplier = {'opening': 0.8, 'middle': 1.2, 'endgame': 1.5}.get(game_phase, 1.0)

        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                threats = self.threat_analysis(board, r, c, color,
                                               'r' if color == 'b' else 'b')
                for threat in threats:
                    total_value += threat['urgency'] * phase_multiplier
        return total_value

    def evaluate_threat_blocking_potential(self, positions, board, enemy_color):
        """评估威胁阻断潜力"""
        total_value = 0
        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                blocking_value = self.calculate_direct_threat_blocking(board, r, c, enemy_color)
                total_value += blocking_value
        return total_value

    def calculate_opponent_card_denial_value(self, positions, board, enemy_color, enemy_threats):
        """计算对手拒绝价值"""
        denial_value = 0
        for r, c in positions:
            if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0':
                potential_value = self.evaluate_opponent_chain_at_position(
                    board, r, c, (0, 1), enemy_color)  # 简化评估
                denial_value += potential_value
        return denial_value

    def calculate_strategic_flexibility(self, positions, board, color, game_phase):
        """计算战略灵活性价值"""
        flexibility_value = 0
        available_count = sum(1 for r, c in positions
                              if 0 <= r < 10 and 0 <= c < 10 and board[r][c] == '0')

        flexibility_value = available_count * 15
        phase_multiplier = {'opening': 1.2, 'middle': 1.0, 'endgame': 0.8}.get(game_phase, 1.0)
        return flexibility_value * phase_multiplier

    def _extract_card_name_from_action(self, card_action):
        """提取卡片名称"""
        try:
            if isinstance(card_action, dict):
                card = (card_action.get('draft_card') or
                        card_action.get('card') or
                        card_action.get('play_card'))

                if isinstance(card, str):
                    return card
                elif isinstance(card, (tuple, list)) and len(card) == 2:
                    rank, suit = card
                    return f"{rank}{suit}"
            return None
        except:
            return None

    def has_critical_threats(self, game_state):
        """检查是否存在关键威胁"""
        if not game_state:
            return False

        try:
            agent = game_state.agents[self.id]
            board = game_state.board.chips
            my_color = agent.colour
            enemy_color = 'r' if my_color == 'b' else 'b'

            # 检查是否存在4连或以上的威胁
            for r in range(10):
                for c in range(10):
                    if board[r][c] in [my_color, enemy_color]:
                        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                            count, _ = self.analyze_chain_pattern(board, r, c, dx, dy, board[r][c])
                            if count >= 4:
                                return True
            return False
        except:
            return False

    def can_form_line(self, r1, c1, r2, c2):
        """检查两点是否能形成直线"""
        return (r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2))

    def calculate_line_threat_potential(self, board, r, c, dx, dy, color):
        """计算线性威胁潜力"""
        threat_count = 0
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + dx * direction * i, c + dy * direction * i
                if 0 <= x < 10 and 0 <= y < 10:
                    if board[x][y] == color:
                        threat_count += 1
                    elif board[x][y] != '0':
                        break
                else:
                    break
        return threat_count

    def evaluate_position_connectivity(self, board, color):
        """评估位置连通性"""
        connectivity_score = 0
        pieces = [(r, c) for r in range(10) for c in range(10) if board[r][c] == color]

        for i, (r1, c1) in enumerate(pieces):
            for r2, c2 in pieces[i + 1:]:
                distance = max(abs(r1 - r2), abs(c1 - c2))
                if distance <= 3:
                    connectivity_score += max(0, 30 - distance * 8)

        return connectivity_score

    def evaluate_threat_density(self, board, my_color, enemy_color):
        """评估威胁密度"""
        my_threats = len(self.analyze_all_threats(board, my_color))
        enemy_threats = len(self.analyze_all_threats(board, enemy_color))
        return my_threats * 50 - enemy_threats * 40

    def calculate_opponent_disruption_value(self, board, r, c, color, enemy_color):
        """计算对手破坏价值"""
        disruption_value = 0

        # 检查这个位置是否破坏了敌方的计划
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # 检查敌方在此方向的威胁
            enemy_pieces_blocked = 0

            for direction in [1, -1]:
                for i in range(1, 5):
                    x, y = r + dx * direction * i, c + dy * direction * i
                    if 0 <= x < 10 and 0 <= y < 10:
                        if board[x][y] == enemy_color:
                            enemy_pieces_blocked += 1
                        elif board[x][y] != '0':
                            break
                    else:
                        break

            # 根据阻断的敌方棋子数量评分
            if enemy_pieces_blocked >= 2:
                disruption_value += enemy_pieces_blocked * 100

        return disruption_value

    def position_evaluation(self, board, r, c, my_color, enemy_color):
        """增强的位置评估"""
        score = 0

        # 基础位置价值
        base_value = self.position_values.get((r, c), 0)
        score += base_value

        # 威胁创建价值
        my_threats = self.threat_analysis(board, r, c, my_color, enemy_color)
        for threat in my_threats:
            score += threat['urgency']

        # 阻断价值
        blocking_value = 0
        temp_board = [row[:] for row in board]
        temp_board[r][c] = my_color  # 占据此位置

        # 检查这样做阻断了对手多少威胁
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            original_threat = self.calculate_line_threat_potential(
                board, r, c, dx, dy, enemy_color)
            blocked_threat = self.calculate_line_threat_potential(
                temp_board, r, c, dx, dy, enemy_color)
            blocking_value += max(0, (original_threat - blocked_threat) * 200)

        score += blocking_value

        # 战略协同价值
        synergy_value = self.calculate_position_synergy_with_existing(
            board, r, c, my_color)
        score += synergy_value

        return score

    def calculate_position_synergy_with_existing(self, board, r, c, color):
        """计算与已有棋子的协同价值"""
        synergy = 0

        # 寻找附近的同色棋子
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < 10 and 0 <= nc < 10 and
                        board[nr][nc] == color):

                    # 计算连接价值
                    distance = max(abs(dr), abs(dc))
                    if distance <= 3:
                        # 检查是否能形成有效连线
                        if self.can_form_line(r, c, nr, nc):
                            synergy += max(0, 80 - distance * 20)

        return synergy

    def threat_analysis(self, board, r, c, color, enemy_color):
        """增强威胁分析 - 更准确的威胁等级判断"""
        threats = []

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            temp_board = [row[:] for row in board]
            temp_board[r][c] = color

            count, openings = self.analyze_chain_pattern(temp_board, r, c, dx, dy, color)
            corner_support = self.check_corner_support(temp_board, r, c, dx, dy, color)

            # 威胁分类更精细
            threat_type = self.classify_threat(count, openings, corner_support)
            if threat_type != 'none':
                threats.append({
                    'direction': (dx, dy),
                    'type': threat_type,
                    'count': count,
                    'openings': openings,
                    'corner_support': corner_support,
                    'urgency': self.calculate_threat_urgency(threat_type, count, openings)
                })

        return threats

    def hotb_score(self, board, color):
        enemy = 'r' if color == 'b' else 'b'
        score = 0
        controlled = 0
        enemy_controlled = 0

        for r, c in HOTB_COORDS:
            if board[r][c] == color:
                controlled += 1
                score += 50  # 提高单格价值
            elif board[r][c] == enemy:
                enemy_controlled += 1
                score -= 80  # 提高失控惩罚

        # 控制优势奖励
        if controlled >= 3:
            score += 200  # 接近完全控制
        if controlled == 4:
            score += 500  # 完全控制额外奖励

        # 阻止敌方控制的紧迫性
        if enemy_controlled >= 2:
            score -= 150  # 敌方威胁惩罚

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
        # 基础评估（权重调整）
        my_chain_score = self.score_all_chains(board, my_color)
        enemy_chain_score = self.score_all_chains(board, enemy_color)
        score += my_chain_score * 2.5 - enemy_chain_score * 2.0

        score += self.hotb_score(board, my_color)
        score += self.corner_control_bonus(board, my_color)

        # 新增评估项
        score += self.evaluate_position_connectivity(board, my_color)
        score += self.evaluate_threat_density(board, my_color, enemy_color)

        return score

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
                        chain_info = self.analyze_chain_for_evaluation(board, r, c, dx, dy, color)
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

    def create_state_signature(self, state: object, last_move: object) -> object:
        """状态签名以避免重复搜索"""
        try:
            board = state.board.chips
            board_str = ''.join(''.join(row) for row in board)
            board_hash = hash(board_str)

            move_info = None
            if last_move and 'coords' in last_move:
                move_info = last_move['coords']

            # 简化状态签名，避免复杂对象序列化
            return (board_hash, move_info, self.turn_count)
        except:
            return (hash(str(state)), str(last_move), self.turn_count)

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

    # 添加性能监控方法
    def get_performance_stats(self):
        """获取性能统计信息"""
        stats = {}

        # 缓存性能
        if self.cache_total_count > 0:
            stats['cache_hit_rate'] = self.cache_hit_count / self.cache_total_count
            stats['cache_size'] = len(self.evaluation_cache)

        # 时间统计
        if hasattr(self, 'time_manager') and self.time_manager.start_time:
            stats['current_turn_time'] = self.time_manager.get_elapsed_time()

        # 游戏统计
        stats['turn_count'] = self.turn_count

        return stats

    # 添加紧急模式处理
    def emergency_quick_decision(self, actions, game_state):
        """紧急模式：超快速决策（50ms内完成）"""
        if not actions:
            return None

        agent, board, is_valid = self.safe_get_game_state_info(game_state)
        if not is_valid:
            return actions[0]

        my_color = agent.colour
        best_action = actions[0]
        best_score = float('-inf')

        # 超快速检查：只考虑HOTB和直接获胜
        for action in actions[:min(len(actions), 3)]:
            if not action.get('coords'):
                continue

            r, c = action['coords']
            score = 0

            # HOTB最高优先级
            if (r, c) in HOTB_COORDS:
                score += 300

            # 检查直接获胜
            temp_board = [row[:] for row in board]
            temp_board[r][c] = my_color
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count = 1
                for direction in [1, -1]:
                    for i in range(1, 4):  # 减少检查范围
                        x, y = r + dx * direction * i, c + dy * direction * i
                        if (0 <= x < 10 and 0 <= y < 10 and
                                (temp_board[x][y] == my_color or self.is_corner_position(x, y))):
                            count += 1
                        else:
                            break
                if count >= 5:
                    return action  # 直接获胜

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def a_star(self, initial_state, candidate_moves):
        """优化的A*搜索，使用更激进的剪枝"""
        pending = []
        seen_states = set()
        best_sequence = []
        top_reward = float('-inf')

        # 自适应参数
        remaining_time = self.time_manager.get_remaining_for_phase('heuristic_search')
        if remaining_time > 0.4:
            max_candidates = 6
            max_expansions = 40
            depth_limit = 3
        elif remaining_time > 0.2:
            max_candidates = 4
            max_expansions = 20
            depth_limit = 2
        else:
            max_candidates = 3
            max_expansions = 10
            depth_limit = 1

        # Alpha-Beta式剪枝阈值
        alpha = float('-inf')

        # 预排序候选动作
        candidate_moves = self.smart_sort_candidates_optimized(initial_state, candidate_moves)

        for move in candidate_moves[:max_candidates]:
            quick_eval = self.quick_evaluate_move(initial_state, move)
            if quick_eval < alpha - 500:  # 激进剪枝
                continue

            g = 1
            h = self.heuristic(initial_state, move)
            f = g + h

            heapq.heappush(pending, (f, next(self.counter), g, h,
                                     self.fast_simulate(initial_state, move), [move]))

        expansions_count = 0
        while (pending and expansions_count < max_expansions and
               self.time_manager.should_continue_phase('heuristic_search')):

            f, _, g, h, current_state, move_history = heapq.heappop(pending)

            # 深度限制剪枝
            if g > depth_limit:
                continue

            # 状态重复检查
            state_sig = self.create_state_signature(current_state, move_history[-1])
            if state_sig in seen_states:
                continue
            seen_states.add(state_sig)

            expansions_count += 1

            # 评估当前状态
            reward = self.evaluate_state(current_state, move_history[-1])
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history
                alpha = max(alpha, reward)  # 更新下界

            # 只在有希望的分支上继续扩展
            if (g < depth_limit and
                    reward > alpha - 1000 and  # 有希望的分支
                    self.time_manager.get_remaining_for_phase('heuristic_search') > 0.05):

                next_steps = self.rule.getLegalActions(current_state, self.id)
                next_steps = self.smart_sort_candidates_optimized(current_state, next_steps)

                for next_move in next_steps[:2]:  # 只扩展最好的2个
                    quick_eval = self.quick_evaluate_move(current_state, next_move)
                    if quick_eval < alpha - 200:  # 剪枝
                        continue

                    next_g = g + 1
                    next_h = self.heuristic(current_state, next_move)
                    heapq.heappush(pending, (
                        next_g + next_h, next(self.counter),
                        next_g, next_h,
                        self.fast_simulate(current_state, next_move),
                        move_history + [next_move]
                    ))

        return best_sequence[0] if best_sequence else None

    def smart_sort_candidates_optimized(self, state, candidates):
        """智能排序候选动作（性能优化版）"""

        def move_priority(action):
            if not action.get('coords'):
                return 0
            r, c = action['coords']

            board = state.board.chips
            agent = state.agents[self.id]
            my_color = agent.colour

            # 基础位置价值（使用预计算）
            score = self.position_values.get((r, c), 0)

            # 快速威胁评估
            temp_board = [row[:] for row in board]
            temp_board[r][c] = my_color

            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count, openings = self.analyze_chain_pattern(
                    temp_board, r, c, dx, dy, my_color)

                # 统一使用calculate_immediate_play_value的评分逻辑
                if count >= 5:
                    score += 10000  # 直接获胜
                    break  # 找到获胜就不需要继续
                elif count >= 4:
                    score += 3000
                elif count >= 3:
                    score += 800
                elif count >= 2:
                    score += 200

            return score

        return sorted(candidates, key=move_priority, reverse=True)

    def get_turn_complexity(self, actions, game_state):
        """评估当前回合的复杂度"""
        complexity = 0.1  # 基础复杂度

        # 可选动作数量
        complexity += min(len(actions) * 0.05, 0.3)

        # 棋盘密度
        if game_state and hasattr(game_state, 'board'):
            board = game_state.board.chips
            piece_count = sum(1 for row in board for cell in row if cell != '0')
            complexity += min(piece_count * 0.01, 0.2)

        # 是否存在关键威胁
        if self.has_critical_threats(game_state):
            complexity += 0.3

        return min(complexity, 1.0)

    def is_game_over(self, state):
        """检查游戏是否结束"""
        try:
            # 简单的游戏结束检查
            board = state.board.chips

            # 检查是否有玩家获胜（5子连线）
            for r in range(10):
                for c in range(10):
                    if board[r][c] in ['r', 'b']:
                        color = board[r][c]
                        # 检查四个方向
                        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                            count = 1
                            # 正向检查
                            for i in range(1, 5):
                                x, y = r + dx * i, c + dy * i
                                if (0 <= x < 10 and 0 <= y < 10 and
                                        (board[x][y] == color or self.is_corner_position(x, y))):
                                    count += 1
                                else:
                                    break
                            # 负向检查
                            for i in range(1, 5):
                                x, y = r - dx * i, c - dy * i
                                if (0 <= x < 10 and 0 <= y < 10 and
                                        (board[x][y] == color or self.is_corner_position(x, y))):
                                    count += 1
                                else:
                                    break

                            if count >= 5:
                                return True

            return False
        except:
            return False


class TimeManager:
    def __init__(self, total_time_limit=0.95):
        self.total_limit = total_time_limit
        self.start_time = None
        self.safety_buffer = 0.05

        # 定义不同决策阶段的时间分配
        self.phase_limits = {
            'quick_decision': 0.15,
            'heuristic_search': 0.50,
            'monte_carlo': 0.80
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
        """判断是否应该继续当前阶段"""
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

    def allocate_time_by_complexity(self, complexity):
        """根据复杂度分配时间"""
        if complexity > 0.7:
            return {
                'quick_decision': 0.10,
                'heuristic_search': 0.65,
                'monte_carlo': 0.85
            }
        elif complexity > 0.4:
            return {
                'quick_decision': 0.15,
                'heuristic_search': 0.55,
                'monte_carlo': 0.80
            }
        else:
            return {
                'quick_decision': 0.20,
                'heuristic_search': 0.45,
                'monte_carlo': 0.70
            }

class WeightLearningSystem:
    """
    权重学习系统：基于游戏结果和关键决策点来优化权重
    """

    def __init__(self):
        self.decision_records = []  # 记录关键决策点
        self.game_outcomes = []  # 记录游戏结果
        self.learning_rate = 0.1  # 学习率

    def record_critical_decision(self, game_state, action, evaluation_details):
        """记录关键决策点的详细信息"""
        decision_record = {
            'board_state': self.compress_board_state(game_state.board.chips),
            'action': action,
            'evaluation_breakdown': evaluation_details,
            'game_phase': self.determine_game_phase(game_state.board.chips),
            'timestamp': time.time()
        }
        self.decision_records.append(decision_record)

    def record_game_outcome(self, won, final_score_difference):
        """记录游戏结果"""
        outcome = {
            'won': won,
            'score_difference': final_score_difference,
            'decisions': self.decision_records.copy()
        }
        self.game_outcomes.append(outcome)
        self.decision_records.clear()  # 清空当前游戏的决策记录

    def compress_board_state(self, board):
        """压缩棋盘状态以节省存储空间"""
        return hash(str(board))

    def determine_game_phase(self, board):
        """确定游戏阶段"""
        total_pieces = sum(1 for row in board for cell in row if cell != '0')
        if total_pieces < 12:
            return 'opening'
        elif total_pieces < 35:
            return 'middle'
        else:
            return 'endgame'

    def analyze_weight_performance(self):
        """
        分析权重性能：找出哪些权重配置导致了更好的结果
        这是一个简化的分析方法，实际应用中可能需要更复杂的机器学习算法
        """
        if len(self.game_outcomes) < 10:
            return None  # 样本不足

        # 分析获胜游戏和失败游戏的权重使用模式
        winning_patterns = []
        losing_patterns = []

        for outcome in self.game_outcomes:
            if outcome['won']:
                winning_patterns.extend(outcome['decisions'])
            else:
                losing_patterns.extend(outcome['decisions'])

        # 简化的权重优化建议
        optimization_suggestions = self.generate_optimization_suggestions(
            winning_patterns, losing_patterns
        )

        return optimization_suggestions

    def generate_optimization_suggestions(self, winning_patterns, losing_patterns):
        """基于成功和失败模式生成权重优化建议"""
        suggestions = {}

        # 这里是一个简化的示例
        # 实际应用中，你可能需要使用更复杂的统计分析或机器学习方法

        if len(winning_patterns) > len(losing_patterns):
            # 如果获胜游戏较多，提取成功模式的特征
            suggestions['chain_threat_4'] = 1.1  # 稍微提高4连威胁的重视度
            suggestions['fork_attack'] = 1.2  # 提高叉攻的权重
        else:
            # 如果失败较多，采用更保守的策略
            suggestions['block_enemy_threat'] = 1.2  # 更重视防守
            suggestions['space_control'] = 1.1  # 更重视空间控制

        return suggestions


# 游戏统计系统
class GameStatistics:
    """记录和分析游戏统计信息"""

    def __init__(self):
        self.move_count = 0
        self.decision_times = []
        self.evaluation_history = []

    def record_decision_time(self, time_taken):
        """记录决策时间"""
        self.decision_times.append(time_taken)

    def record_evaluation(self, position, evaluation_score):
        """记录位置评估分数"""
        self.evaluation_history.append({
            'position': position,
            'score': evaluation_score,
            'move_number': self.move_count
        })
        self.move_count += 1

    def get_performance_metrics(self):
        """获取性能指标"""
        if not self.decision_times:
            return {}

        return {
            'average_decision_time': sum(self.decision_times) / len(self.decision_times),
            'max_decision_time': max(self.decision_times),
            'total_moves': self.move_count,
            'evaluation_trend': self.analyze_evaluation_trend()
        }

    def analyze_evaluation_trend(self):
        """分析评估分数的趋势"""
        if len(self.evaluation_history) < 5:
            return "insufficient_data"

        recent_scores = [e['score'] for e in self.evaluation_history[-5:]]
        early_scores = [e['score'] for e in self.evaluation_history[:5]]

        recent_avg = sum(recent_scores) / len(recent_scores)
        early_avg = sum(early_scores) / len(early_scores)

        if recent_avg > early_avg * 1.1:
            return "improving"
        elif recent_avg < early_avg * 0.9:
            return "declining"
        else:
            return "stable"


class MCTSNode:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action  # 到达此状态的动作
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = []
        self.player_id = None  # 当前玩家ID

    def ucb1_score(self, exploration_weight=1.414):
        """UCB1选择策略"""
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return float('inf')

        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        """选择UCB1分数最高的子节点"""
        if not self.children:
            return self
        return max(self.children, key=lambda c: c.ucb1_score())

    def add_child(self, action, state):
        """添加子节点"""
        child = MCTSNode(state, action, self)
        if self.player_id in [0, 1]:
            child.player_id = 1 - self.player_id
        else:
            child.player_id = 1 if self.player_id == 0 else 0
        self.children.append(child)
        return child
    def update(self, result):
        """更新节点统计信息"""
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        """检查是否完全扩展"""
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def is_terminal(self):
        """检查是否为终端节点 - 基于Sequence游戏规则（两个五连获胜）"""
        try:
            if not hasattr(self, 'state') or not self.state:
                return True

            board = self.state.board.chips

            # 检查两个玩家的五连数量
            red_sequences = self._count_sequences(board, 'r')
            blue_sequences = self._count_sequences(board, 'b')

            # 游戏结束条件：任一玩家达到2个五连
            if red_sequences >= 2 or blue_sequences >= 2:
                return True

            # 检查是否平局（棋盘满了但没人获胜）
            if self._is_board_full(board):
                return True

            return False

        except:
            return True

    def _is_board_full(self, board):
        """检查棋盘是否已满"""
        for r in range(10):
            for c in range(10):
                if board[r][c] == '0':  # 还有空位
                    return False
        return True


# 对手模型类
class OpponentModel:
    """对手行为建模和预测"""

    def __init__(self):
        self.opponent_moves = []
        self.position_preferences = {}
        self.style_indicators = {
            'aggressive': 0.5,
            'defensive': 0.5,
            'strategic': 0.5
        }

    def record_opponent_move(self, board_before, board_after, turn):
        """记录对手移动"""
        # 找出对手放置的位置
        move_pos = None
        for r in range(10):
            for c in range(10):
                if board_before[r][c] != board_after[r][c]:
                    move_pos = (r, c)
                    break

        if move_pos:
            self.opponent_moves.append({
                'position': move_pos,
                'turn': turn,
                'board_state': self.compress_board(board_before)
            })

            # 更新位置偏好
            if move_pos not in self.position_preferences:
                self.position_preferences[move_pos] = 0
            self.position_preferences[move_pos] += 1

    def predict_position_value(self, board, r, c, color, turn):
        """预测对手对某位置的价值评估"""
        base_value = 0

        # 基于历史偏好
        if (r, c) in self.position_preferences:
            base_value += self.position_preferences[(r, c)] * 20

        # 基于游戏阶段调整
        if turn <= 6:  # 开局
            if (r, c) in HOTB_COORDS:
                base_value += 100
        elif turn <= 20:  # 中局
            # 更重视连子
            base_value += self.estimate_chain_value(board, r, c, color)
        else:  # 残局
            # 更重视阻断和角落
            base_value += self.estimate_blocking_value(board, r, c, color)

        return base_value

    def estimate_chain_value(self, board, r, c, color):
        """估算连子价值"""
        # 简化的连子价值估算
        value = 0
        temp_board = [row[:] for row in board]
        temp_board[r][c] = color

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for direction in [1, -1]:
                for i in range(1, 5):
                    x, y = r + dx * direction * i, c + dy * direction * i
                    if 0 <= x < 10 and 0 <= y < 10 and temp_board[x][y] == color:
                        count += 1
                    else:
                        break

            if count >= 3:
                value += count * 50

        return value

    def estimate_blocking_value(self, board, r, c, enemy_color):
        """估算阻断价值"""
        my_color = 'r' if enemy_color == 'b' else 'b'
        value = 0

        # 检查阻断我方威胁的价值
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            my_count = 0
            for direction in [1, -1]:
                for i in range(1, 5):
                    x, y = r + dx * direction * i, c + dy * direction * i
                    if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == my_color:
                        my_count += 1
                    else:
                        break

            if my_count >= 2:
                value += my_count * 30

        return value

    def compress_board(self, board):
        """压缩棋盘状态"""
        return hash(str(board))


# 开局库类
class OpeningBook:
    """简单高效的开局策略库"""

    def __init__(self):
        # HOTB中心区域的实际卡片（最重要的4个位置）
        self.HOTB_CARDS = ['5h', '4h', '2h', '3h']
        self.HOTB_POSITIONS = [(4, 4), (4, 5), (5, 4), (5, 5)]

        # 简单的开局优先级表
        self.opening_priorities = {
            # 第一优先级：HOTB卡片
            'hotb_cards': ['5h', '4h', '2h', '3h'],
            # 第二优先级：HOTB周围支援卡片
            'hotb_support': ['6h', '7h', '8h', '9h', 'ah', 'kh', 'qh', 'th'],
            # 第三优先级：角落和Jack
            'corner_cards': ['jk', 'as', 'ks'],
            # 第四优先级：其他有用卡片
            'other_useful': ['2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']
        }

        # 位置价值表（简化版）
        self.position_values = {
            # HOTB核心 - 最高价值
            (4, 4): 100, (4, 5): 100, (5, 4): 100, (5, 5): 100,
            # HOTB周围 - 高价值
            (3, 4): 70, (3, 5): 70, (4, 3): 70, (4, 6): 70,
            (5, 3): 70, (5, 6): 70, (6, 4): 70, (6, 5): 70,
            # 角落 - 中等价值
            (0, 0): 60, (0, 9): 60, (9, 0): 60, (9, 9): 60,
            # 角落邻近 - 中等价值
            (1, 1): 50, (1, 8): 50, (8, 1): 50, (8, 8): 50
        }

    def get_card_bonus(self, card_actions, board):
        """为每张可选卡片计算开局奖励分数"""
        if not card_actions:
            return {}

        # 快速回合数检查
        turn = self._count_pieces(board) + 1
        if turn > 10:  # 只在前10回合使用开局库
            return {}

        card_bonuses = {}

        for card_action in card_actions:
            card_name = self._extract_card_name(card_action)
            if card_name:
                bonus = self._calculate_simple_bonus(card_name, board, turn)
                if bonus > 0:
                    card_bonuses[card_action] = bonus

        return card_bonuses

    def _extract_card_name(self, card_action):
        """简单提取卡片名称"""
        try:
            if isinstance(card_action, dict):
                card = (card_action.get('draft_card') or
                        card_action.get('card') or
                        card_action.get('play_card'))

                if isinstance(card, str):
                    return card
                elif isinstance(card, (tuple, list)) and len(card) == 2:
                    # 处理 (rank, suit) 格式
                    rank, suit = card
                    return f"{rank}{suit}"
            return None
        except:
            return None

    def _count_pieces(self, board):
        """快速计算棋盘上的棋子数"""
        return sum(1 for row in board for cell in row if cell != '0')

    def _calculate_simple_bonus(self, card_name, board, turn):
        """简单快速的奖励计算"""
        bonus = 0

        # 基础卡片价值
        if card_name in self.opening_priorities['hotb_cards']:
            bonus = 100
        elif card_name in self.opening_priorities['hotb_support']:
            bonus = 70
        elif card_name in self.opening_priorities['corner_cards']:
            bonus = 50
        elif card_name in self.opening_priorities['other_useful']:
            bonus = 30
        else:
            bonus = 10  # 基础分

        # 早期回合HOTB更重要
        if turn <= 4 and card_name in self.opening_priorities['hotb_cards']:
            bonus *= 1.5

        # Jack在任何时候都有额外价值（万能牌）
        if card_name == 'jk':
            bonus += 40

        # 简单的对手反制：如果对手控制了太多HOTB，优先选HOTB卡片
        hotb_threat = self._check_hotb_threat(board)
        if hotb_threat and card_name in self.opening_priorities['hotb_cards']:
            bonus += 50

        return bonus

    def _check_hotb_threat(self, board):
        """检查对手是否对HOTB构成威胁"""
        my_pieces = sum(1 for row in board for cell in row if cell == 'b')
        enemy_pieces = sum(1 for row in board for cell in row if cell == 'r')

        if my_pieces >= enemy_pieces:
            enemy_color = 'r'  # 假设我方是蓝色
        else:
            enemy_color = 'b'  # 假设我方是红色

        enemy_hotb_count = 0
        for r, c in self.HOTB_POSITIONS:
            if board[r][c] == enemy_color:
                enemy_hotb_count += 1
        return enemy_hotb_count >= 2

    def get_best_position_for_card(self, card_name, available_positions):
        """为特定卡片选择最佳位置"""
        if not available_positions:
            return None

        best_pos = None
        best_value = 0

        for r, c in available_positions:
            value = self.position_values.get((r, c), 0)

            # HOTB卡片在HOTB位置有额外奖励
            if card_name in self.HOTB_CARDS and (r, c) in self.HOTB_POSITIONS:
                value += 50

            if value > best_value:
                best_value = value
                best_pos = (r, c)

        return best_pos

    def get_strategy_name(self, card_name):
        """获取卡片对应的策略名称"""
        if card_name in self.opening_priorities['hotb_cards']:
            return 'center_control'
        elif card_name in self.opening_priorities['hotb_support']:
            return 'center_control'
        elif card_name == 'jk':
            return 'corner_development'
        elif card_name in self.opening_priorities['corner_cards']:
            return 'corner_development'
        else:
            return 'balanced_development'