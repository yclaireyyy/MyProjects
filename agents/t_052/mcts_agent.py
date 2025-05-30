from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import random
import time
import copy
import math
import itertools
from collections import defaultdict, deque

# ===========================
# 1. Constants Definition Section
# ===========================
MAX_THINK_TIME = 0.95  # Maximum thinking time (seconds)
EXPLORATION_WEIGHT = 1.2  # Exploration parameter in UCB formula
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]  # Center hotspot positions
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]  # Corner positions (free spaces)
SIMULATION_LIMIT = 200  # Maximum number of MCTS simulations

# Pre-computed direction vectors and position weights
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
POSITION_WEIGHTS = {}  # Position weights cache
ACTION_CACHE = {}  # Action evaluation cache

# Initialize position weights
for i in range(10):
    for j in range(10):
        if (i, j) in HOTB_COORDS:
            POSITION_WEIGHTS[(i, j)] = 1.5
        elif i in [0, 9] or j in [0, 9]:
            POSITION_WEIGHTS[(i, j)] = 0.8
        else:
            POSITION_WEIGHTS[(i, j)] = 1.0


# ===========================
# 2. Game Phase Definition
# ===========================
class GamePhase:
    OPENING = "opening"  # Opening phase (less than 15 pieces)
    MIDDLE = "middle"  # Middle game (15-35 pieces)
    CRITICAL = "critical"  # Critical phase (sequences about to complete)
    ENDGAME = "endgame"  # End game (more than 35 pieces)


# ===========================
# 3. LRU Cache Implementation
# ===========================
class LRUCache:
    """Simple LRU cache implementation for limiting cache size"""

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            oldest = self.order.popleft()
            del self.cache[oldest]

        self.cache[key] = value
        self.order.append(key)

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.order.clear()


class BoardState:
    """Lightweight board state for fast copying"""

    def __init__(self, chips):
        self.chips = chips
        self._hash = None

    def get_hash(self):
        """Get board state hash value for caching"""
        if self._hash is None:
            self._hash = hash(tuple(tuple(row) for row in self.chips))
        return self._hash

    def copy(self):
        """Fast board copy"""
        return BoardState([row[:] for row in self.chips])


class CardEvaluator:
    def __init__(self, agent):
        self.agent = agent
        self._card_cache = LRUCache(capacity=5000)  # Use LRU cache
        self._hand_diversity_cache = {}

    def _evaluate_card(self, card, state, consider_opponent=True):
        """Evaluate card value in current state"""
        # Generate cache key
        board_hash = state.board.chips if hasattr(state.board, 'get_hash') else hash(
            tuple(tuple(row) for row in state.board.chips))
        cache_key = (str(card), board_hash, consider_opponent)

        cached_result = self._card_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        board = state.board.chips

        # Priority 1: Two-eyed Jack - directly highest score
        if self._is_two_eyed_jack(card):
            result = 10000
        # Priority 2: One-eyed Jack - second highest score
        elif self._is_one_eyed_jack(card):
            result = 5000
        # Priority 3: Regular cards - use exponential scoring
        elif card in COORDS:
            my_score = self._exponential_card_evaluation(card, state)

            # Consider opponent value
            if consider_opponent and hasattr(state, 'agents') and len(state.agents) > 1:
                opp_score = self._evaluate_card_for_opponent(card, state)
                result = my_score - opp_score * 0.3  # Subtract 30% of opponent value
            else:
                result = my_score
        else:
            result = 0

        self._card_cache.put(cache_key, result)
        return result

    def _evaluate_card_for_opponent(self, card, state):
        """Evaluate card value for opponent"""
        # Temporarily switch color for evaluation
        original_color = self.agent.my_color
        self.agent.my_color = self.agent.opp_color
        opp_value = self._exponential_card_evaluation(card, state)
        self.agent.my_color = original_color
        return opp_value

    def _calculate_hand_diversity(self, hand):
        """Calculate hand diversity score"""
        if not hand:
            return 0

        # Count different card types
        card_counts = defaultdict(int)
        for card in hand:
            if card in COORDS:
                card_counts[str(card)] += 1

        # Diversity score: more different card types is better
        diversity = len(card_counts)
        # Penalty for duplicate cards
        penalty = sum(count - 1 for count in card_counts.values() if count > 1)

        return diversity - penalty * 0.5

    def _exponential_card_evaluation(self, card, state):
        """Exponential-based regular card evaluation (optimized version)"""
        if card not in COORDS:
            return 0

        board = state.board.chips
        total_score = 0
        # Get all possible positions for this card
        positions = COORDS[card] if isinstance(COORDS[card], list) else [COORDS[card]]
        valid_positions = 0

        for pos in positions:
            if isinstance(pos, list):  # Handle nested lists
                pos = tuple(pos)
            r, c = pos
            # Check if position is available
            if not self._is_position_available(board, r, c):
                continue
            # Calculate exponential score for this position
            position_score = self._calculate_position_score(board, r, c)
            total_score += position_score
            valid_positions += 1

        # If multiple positions available, take average
        return total_score / max(1, valid_positions) if valid_positions > 0 else 0

    def _calculate_position_score(self, board, r, c):
        """Calculate exponential score for single position (optimized version)"""
        total_score = 0

        # Four main directions: horizontal, vertical, main diagonal, anti-diagonal
        for dx, dy in DIRECTIONS:
            my_pieces = self._count_my_pieces_in_direction(board, r, c, dx, dy)
            direction_score = self._exponential_scoring(my_pieces)
            total_score += direction_score

        return total_score

    def _count_my_pieces_in_direction(self, board, r, c, dx, dy):
        """Count my pieces within 5 positions in specific direction"""
        my_pieces = 0
        my_color = self.agent.my_color

        # Check 4 positions in each direction (total 8 positions)
        for i in range(-4, 5):
            # Skip center position (position to be placed)
            if i == 0:
                continue
            x, y = r + i * dx, c + i * dy
            # Combined boundary check and color check
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == my_color:
                my_pieces += 1

        return my_pieces

    def _exponential_scoring(self, piece_count):
        """Exponential scoring rule: 1 piece=10 points, 2 pieces=100 points, 3 pieces=1000 points"""
        if piece_count == 0:
            return 1  # Base score
        elif piece_count == 1:
            return 10
        elif piece_count == 2:
            return 100
        elif piece_count == 3:
            return 1000
        elif piece_count >= 4:
            return 10000  # 4 or more - close to winning

        return 0

    def _is_two_eyed_jack(self, card):
        """Check if card is two-eyed Jack"""
        try:
            card_str = str(card).lower()
            return card_str in ['jc', 'jd']  # Two-eyed Jacks
        except:
            return False

    def _is_one_eyed_jack(self, card):
        """Check if card is one-eyed Jack"""
        try:
            card_str = str(card).lower()
            return card_str in ['js', 'jh']  # One-eyed Jacks
        except:
            return False

    def _is_position_available(self, board, r, c):
        """Check if position is available"""
        if not (0 <= r < 10 and 0 <= c < 10):
            return False
        # Do not consider free space
        if (r, c) in CORNERS:
            return False
        return board[r][c] == 0 or board[r][c] == '0'  # Empty position


class ActionEvaluator:
    _evaluation_cache = LRUCache(capacity=10000)  # Use LRU cache
    _threat_cache = LRUCache(capacity=5000)

    @staticmethod
    def evaluate_action_quality(state, action):
        """Evaluate action quality score (lower is better)"""
        if action.get('type') != 'place' or 'coords' not in action:
            return 100  # Non-place actions or no coordinates

        r, c = action['coords']
        if (r, c) in CORNERS:
            return 100  # Corner positions

        # Generate cache key
        board_hash = hash(tuple(tuple(row) for row in state.board.chips))
        cache_key = (board_hash, r, c)

        cached_result = ActionEvaluator._evaluation_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        board = state.board.chips

        # Get player colors
        if hasattr(state, 'my_color'):
            color = state.my_color
            enemy = state.opp_color
        else:
            # Infer color from action
            agent_id = state.current_player_id if hasattr(state, 'current_player_id') else 0
            color = state.agents[agent_id].colour
            enemy = 'r' if color == 'b' else 'b'

        # Calculate various scores
        score = 0

        # Center preference (using pre-computed distance)
        distance = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 5 - distance) * 2

        # Consecutive chain scoring (optimized version)
        for dx, dy in DIRECTIONS:
            count = ActionEvaluator._count_consecutive_fast(board, r, c, dx, dy, color)
            # Score based on consecutive length
            if count >= 5:
                score += 2000  # Form sequence - significantly increase score
            elif count == 4:
                score += 500  # Close to winning
            elif count == 3:
                score += 100
            elif count == 2:
                score += 20

        # Enhanced defense scoring
        critical_block = False
        for dx, dy in DIRECTIONS:
            enemy_chain = ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy)
            if enemy_chain >= 4:
                score += 1000  # Extremely high priority block
                critical_block = True
            elif enemy_chain >= 3:
                score += 300  # High priority block

            # Check double threat
            if ActionEvaluator._check_double_threat(board, r, c, dx, dy, enemy):
                score += 400

        # Special case: if critical defense, significantly boost score
        if critical_block:
            score *= 2

        # Center control scoring (using position weights)
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if (x, y) == (r, c))
        score += hotb_controlled * 15

        # Convert to quality score (lower is better)
        result = 100 - score
        ActionEvaluator._evaluation_cache.put(cache_key, result)
        return result

    @staticmethod
    def _check_double_threat(board, r, c, dx, dy, enemy):
        """Check if double threat exists"""
        # Check if both ends can be extended
        threats = 0

        # Forward check
        for i in range(1, 5):
            x, y = r + i * dx, c + i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == 0 or board[x][y] == '0':
                    threats += 1
                    break
                elif board[x][y] != enemy:
                    break

        # Backward check
        for i in range(1, 5):
            x, y = r - i * dx, c - i * dy
            if 0 <= x < 10 and 0 <= y < 10:
                if board[x][y] == 0 or board[x][y] == '0':
                    threats += 1
                    break
                elif board[x][y] != enemy:
                    break

        return threats >= 2

    @staticmethod
    def is_winning_move(state, action, color):
        """Check if action is winning move"""
        if action.get('type') != 'place' or 'coords' not in action:
            return False

        r, c = action['coords']
        board = state.board.chips

        # Simulate placement
        board_copy = [row[:] for row in board]
        board_copy[r][c] = color

        # Check if sequence is formed
        for dx, dy in DIRECTIONS:
            if ActionEvaluator._count_consecutive_fast(board_copy, r, c, dx, dy, color) >= 5:
                return True
        return False

    @staticmethod
    def blocks_opponent_win(state, action, enemy):
        """Check if action blocks opponent win"""
        if action.get('type') != 'place' or 'coords' not in action:
            return False

        r, c = action['coords']
        board = state.board.chips

        # Check if this position is opponent's critical position
        for dx, dy in DIRECTIONS:
            if ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy) >= 4:
                return True
        return False

    @staticmethod
    def _count_consecutive_fast(board, x, y, dx, dy, color):
        """Optimized consecutive counting"""
        count = 1  # Starting position counts as one

        # Forward check
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < 10 and 0 <= ny < 10 and board[nx][ny] == color:
                count += 1
            else:
                break

        # Backward check
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if 0 <= nx < 10 and 0 <= ny < 10 and board[nx][ny] == color:
                count += 1
            else:
                break

        return min(count, 5)  # Return maximum 5 (forms one sequence)

    @staticmethod
    def _count_enemy_threat_fast(board, r, c, dx, dy, enemy):
        """Optimized enemy threat counting"""
        enemy_chain = 0
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
                enemy_chain += 1
            else:
                break
        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if 0 <= x < 10 and 0 <= y < 10 and board[x][y] == enemy:
                enemy_chain += 1
            else:
                break
        return enemy_chain

    @staticmethod
    def _calculate_action_score(board, r, c, color, enemy):
        """Calculate action score"""
        score = 0

        # Create hypothetical board
        board_copy = [row[:] for row in board]
        board_copy[r][c] = color

        # Center preference
        distance = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 5 - distance) * 2

        # Consecutive chain scoring
        for dx, dy in DIRECTIONS:
            count = ActionEvaluator._count_consecutive_fast(board_copy, r, c, dx, dy, color)
            if count >= 5:
                score += 2000
            elif count == 4:
                score += 500
            elif count == 3:
                score += 100
            elif count == 2:
                score += 20

        # Defense scoring
        for dx, dy in DIRECTIONS:
            enemy_threat = ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy)
            if enemy_threat >= 4:
                score += 1000
            elif enemy_threat >= 3:
                score += 300

        # Center control
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if board_copy[x][y] == color)
        score += hotb_controlled * 15

        return score

    @staticmethod
    def _count_consecutive(board, x, y, dx, dy, color):
        """Backward compatible method"""
        return ActionEvaluator._count_consecutive_fast(board, x, y, dx, dy, color)

    @staticmethod
    def _count_enemy_threat(board, r, c, dx, dy, enemy):
        """Backward compatible method"""
        return ActionEvaluator._count_enemy_threat_fast(board, r, c, dx, dy, enemy)


class StateEvaluator:
    _state_cache = LRUCache(capacity=5000)  # Use LRU cache

    @staticmethod
    def evaluate(state, last_action=None):
        """Evaluate game state value (with caching)"""
        # Generate cache key
        board_hash = hash(tuple(tuple(row) for row in state.board.chips))

        cached_result = StateEvaluator._state_cache.get(board_hash)
        if cached_result is not None:
            return cached_result

        board = state.board.chips

        # Get player colors
        if hasattr(state, 'my_color'):
            my_color = state.my_color
            opp_color = state.opp_color
        else:
            # Infer color from state
            agent_id = state.current_player_id if hasattr(state, 'current_player_id') else 0
            my_color = state.agents[agent_id].colour
            opp_color = 'r' if my_color == 'b' else 'b'

        # 1. Position scoring (using pre-computed weights)
        position_score = 0
        for i in range(10):
            for j in range(10):
                cell = board[i][j]
                if cell == my_color:
                    position_score += POSITION_WEIGHTS[(i, j)]
                elif cell == opp_color:
                    position_score -= POSITION_WEIGHTS[(i, j)]

        # 2. Sequence potential scoring (batch processing)
        sequence_score = StateEvaluator._calculate_sequence_score_fast(board, my_color)

        # 3. Defense scoring - prevent opponent sequences
        defense_score = StateEvaluator._calculate_defense_score_fast(board, opp_color)

        # 4. Center control scoring
        hotb_score = 0
        for x, y in HOTB_COORDS:
            cell = board[x][y]
            if cell == my_color:
                hotb_score += 5
            elif cell == opp_color:
                hotb_score -= 5

        # 5. Combined scoring
        total_score = position_score + sequence_score + defense_score + hotb_score

        # Normalize to [-1, 1] range
        result = max(-1, min(1, total_score / 200))
        StateEvaluator._state_cache.put(board_hash, result)
        return result

    @staticmethod
    def _calculate_sequence_score_fast(board, color):
        """Optimized sequence score calculation"""
        sequence_score = 0
        # Use set to avoid duplicate calculations
        counted_sequences = set()

        for i in range(10):
            for j in range(10):
                if board[i][j] == color:
                    for dx, dy in DIRECTIONS:
                        # Generate unique identifier for sequence
                        sequence_id = (i, j, dx, dy)
                        if sequence_id not in counted_sequences:
                            count = ActionEvaluator._count_consecutive_fast(board, i, j, dx, dy, color)
                            if count >= 2:  # Only count meaningful sequences
                                # Mark entire sequence as counted
                                for k in range(count):
                                    counted_sequences.add((i + k * dx, j + k * dy, dx, dy))

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
    def _calculate_defense_score_fast(board, opp_color):
        """Optimized defense score calculation"""
        defense_score = 0
        # Use set to avoid duplicate calculations
        counted_threats = set()

        for i in range(10):
            for j in range(10):
                if board[i][j] == opp_color:
                    for dx, dy in DIRECTIONS:
                        threat_id = (i, j, dx, dy)
                        if threat_id not in counted_threats:
                            count = ActionEvaluator._count_consecutive_fast(board, i, j, dx, dy, opp_color)
                            if count >= 3:  # Only focus on real threats
                                # Mark entire threat sequence
                                for k in range(count):
                                    counted_threats.add((i + k * dx, j + k * dy, dx, dy))

                                if count >= 4:
                                    defense_score -= 50
                                elif count == 3:
                                    defense_score -= 10

        return defense_score

    @staticmethod
    def _calculate_sequence_score(board, color):
        """Backward compatible method"""
        return StateEvaluator._calculate_sequence_score_fast(board, color)

    @staticmethod
    def _calculate_defense_score(board, opp_color):
        """Backward compatible method"""
        return StateEvaluator._calculate_defense_score_fast(board, opp_color)


class ActionSimulator:
    def __init__(self, agent):
        self.agent = agent

    def simulate_action(self, state, action):
        """Simulate action execution"""
        new_state = self._copy_state(state)

        if action['type'] == 'place':
            self._simulate_place(new_state, action)
        elif action['type'] == 'remove':
            self._simulate_remove(new_state, action)

        return new_state

    def _simulate_place(self, state, action):
        """Simulate place action"""
        if 'coords' not in action:
            return

        r, c = action['coords']
        color = self._get_current_color(state)
        state.board.chips[r][c] = color

        # Update hand
        self._update_hand(state, action)

    def _simulate_remove(self, state, action):
        """Simulate remove action"""
        if 'coords' not in action:
            return

        r, c = action['coords']
        state.board.chips[r][c] = 0  # Remove piece

        # Update hand
        self._update_hand(state, action)

    def _get_current_color(self, state):
        """Get current player color"""
        if hasattr(state, 'current_player_id'):
            return state.agents[state.current_player_id].colour
        return self.agent.my_color

    def _update_hand(self, state, action):
        """Update hand cards"""
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
        """Copy state"""
        if hasattr(state, "copy"):
            return state.copy()
        else:
            return copy.deepcopy(state)


class Node:
    """
    MCTS search tree node with integrated heuristic evaluation
    """

    def __init__(self, state, parent=None, action=None):
        # State representation (optimized copying)
        self.state = self._efficient_copy_state(state)
        # Node relationships
        self.parent = parent
        self.children = []
        self.action = action
        # MCTS statistics
        self.visits = 0
        self.value = 0.0
        # Action management (lazy initialization)
        self.untried_actions = None
        # Terminal node flag
        self.is_terminal = False

    def _efficient_copy_state(self, state):
        """Efficient state copying"""
        try:
            return state.clone()
        except:
            # Only copy essential parts
            if hasattr(state, 'board') and hasattr(state.board, 'chips'):
                new_state = copy.copy(state)
                new_state.board = copy.copy(state.board)
                new_state.board.chips = [row[:] for row in state.board.chips]
                return new_state
            return copy.deepcopy(state)

    def get_untried_actions(self):
        """Get untried actions with heuristic sorting"""
        if self.untried_actions is None:
            # Initialize untried actions list
            if hasattr(self.state, 'available_actions'):
                self.untried_actions = list(self.state.available_actions)
            else:
                self.untried_actions = []
            # Sort using heuristic evaluation (smaller values first)
            self.untried_actions.sort(key=lambda a: ActionEvaluator.evaluate_action_quality(self.state, a))
        return self.untried_actions

    def is_fully_expanded(self):
        """Check if node is fully expanded"""
        return len(self.get_untried_actions()) == 0

    def select_child(self):
        """Select most promising child using UCB formula (optimized version)"""
        best_score = float('-inf')
        best_child = None
        log_visits = math.log(self.visits) if self.visits > 0 else 0  # BUG FIX: avoid log(0)

        for child in self.children:
            # UCB calculation
            if child.visits == 0:
                score = float('inf')
            else:
                # UCB calculation with heuristic evaluation
                exploitation = child.value / child.visits
                exploration = EXPLORATION_WEIGHT * math.sqrt(2 * log_visits / child.visits)
                # Heuristic adjustment
                if child.action:
                    heuristic_score = ActionEvaluator.evaluate_action_quality(self.state, child.action)
                    heuristic_factor = 1.0 / (1.0 + max(0, heuristic_score) / 100)  # BUG FIX: avoid negative denominator
                else:
                    heuristic_factor = 1.0

                score = exploitation + exploration * heuristic_factor

            # Update best node
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, agent):
        """Expand new child node using heuristic selection of most promising action"""
        untried = self.get_untried_actions()

        if not untried:
            return None

        # Select (and remove) first action from list (already sorted by heuristics)
        action = untried.pop(0)
        # Create new state
        new_state = agent.fast_simulate(self.state, action)
        # Create child node
        child = Node(new_state, parent=self, action=action)
        self.children.append(child)

        return child

    def update(self, result):
        """Update node statistics"""
        self.visits += 1
        self.value += result


class AdaptiveTimeManager:
    """Adaptive time manager"""

    def __init__(self):
        self.start_time = 0
        self.time_budget = MAX_THINK_TIME
        self.move_count = 0
        self.phase = GamePhase.OPENING

    def start_timing(self):
        self.start_time = time.time()
        self.move_count += 1

    def get_remaining_time(self):
        elapsed = time.time() - self.start_time
        return self.time_budget - elapsed

    def is_timeout(self, buffer=0.05):
        return self.get_remaining_time() < buffer

    def should_use_quick_mode(self):
        return self.get_remaining_time() < 0.25

    def get_time_budget(self, state, importance=0.5):
        """Dynamically allocate time based on game phase and importance"""
        # Determine game phase
        piece_count = sum(1 for i in range(10) for j in range(10)
                          if state.board.chips[i][j] not in [0, '0'])

        if piece_count < 15:
            self.phase = GamePhase.OPENING
            base_time = 0.4
        elif piece_count < 35:
            self.phase = GamePhase.MIDDLE
            base_time = 0.6
        else:
            self.phase = GamePhase.ENDGAME
            base_time = 0.8

        # Check if critical state
        if self._is_critical_state(state):
            self.phase = GamePhase.CRITICAL
            base_time = 0.9

        # Adjust based on importance
        return min(MAX_THINK_TIME, base_time * (1 + importance * 0.5))

    def _is_critical_state(self, state):
        """Check if state is critical"""
        board = state.board.chips
        # Check for 4-in-a-row situations
        for i in range(10):
            for j in range(10):
                if board[i][j] not in [0, '0'] and board[i][j] not in CORNERS:
                    color = board[i][j]
                    for dx, dy in DIRECTIONS:
                        if ActionEvaluator._count_consecutive_fast(board, i, j, dx, dy, color) >= 4:
                            return True
        return False


class myAgent(Agent):
    """Intelligent Agent myAgent - Enhanced Version"""

    def __init__(self, _id):
        """Initialize Agent"""
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)  # 2-player game
        self.counter = itertools.count()  # Unique identifier for search

        self.card_evaluator = CardEvaluator(self)
        self.time_manager = AdaptiveTimeManager()  # Use adaptive time management

        # Player color initialization
        self.my_color = None
        self.opp_color = None

        # Search parameters (optimized)
        self.simulation_depth = 4  # Slightly reduce depth but improve quality
        self.candidate_limit = 12  # Slightly increase candidate count

        # Time control
        self.start_time = 0

        # Opening book and pattern recognition
        self._initialize_opening_book()
        self.move_history = []

        # Performance statistics
        self.performance_stats = {
            'mcts_iterations': [],
            'decision_times': []
        }

    def _initialize_opening_book(self):
        """Initialize opening book"""
        self.opening_book = {
            # First move prioritizes center control
            0: [(4, 4), (4, 5), (5, 4), (5, 5)],
            # Second move responds to opponent
            1: {
                (4, 4): [(5, 5), (4, 5), (5, 4)],
                (4, 5): [(5, 4), (4, 4), (5, 5)],
                (5, 4): [(4, 5), (5, 5), (4, 4)],
                (5, 5): [(4, 4), (5, 4), (4, 5)]
            }
        }

    def _initialize_colors(self, game_state):
        """Initialize color information"""
        if self.my_color is None:
            self.my_color = game_state.agents[self.id].colour
            self.opp_color = game_state.agents[1 - self.id].colour

    def _is_card_selection(self, actions):
        """Check if this is card selection phase"""
        return any(a.get('type') == 'trade' for a in actions)

    def _select_strategic_card(self, actions, game_state):
        """Enhanced card selection logic"""
        trade_actions = [a for a in actions if a.get('type') == 'trade']

        if not hasattr(game_state, 'display_cards') or not game_state.display_cards:
            return random.choice(trade_actions) if trade_actions else None

        # Get current hand
        current_hand = []
        if hasattr(game_state.agents[self.id], 'hand'):
            current_hand = game_state.agents[self.id].hand

        # Evaluate all display cards
        best_card = None
        best_score = float('-inf')

        for card in game_state.display_cards:
            # Base value
            immediate_score = self.card_evaluator._evaluate_card(card, game_state, consider_opponent=True)

            # Hand diversity bonus
            diversity_bonus = self._calculate_diversity_bonus(card, current_hand)

            # Special card bonus
            if self.card_evaluator._is_two_eyed_jack(card):
                special_bonus = 5000
            elif self.card_evaluator._is_one_eyed_jack(card):
                special_bonus = 2500
            else:
                special_bonus = 0

            total_score = immediate_score + diversity_bonus + special_bonus

            if total_score > best_score:
                best_score = total_score
                best_card = card

        # Find corresponding action
        for action in trade_actions:
            if action.get('draft_card') == best_card:
                return action

        return random.choice(trade_actions) if trade_actions else None

    def _calculate_diversity_bonus(self, card, hand):
        """Calculate contribution of selecting this card to hand diversity"""
        if not hand:
            return 100  # Bonus for first card

        # Penalty if hand already has same card
        card_str = str(card)
        count = sum(1 for c in hand if str(c) == card_str)

        if count == 0:
            return 50  # New card type
        elif count == 1:
            return 0  # Already have one
        else:
            return -50 * count  # Penalty for duplicates

    def SelectAction(self, actions, game_state):
        """Main decision function - integrates heuristic filtering and MCTS"""
        self.time_manager.start_timing()
        self._initialize_colors(game_state)

        # Check if actions is empty
        if not actions:
            return None

        if self._is_card_selection(actions):
            return self._select_strategic_card(actions, game_state)

        # Quick win detection
        for action in actions:
            if ActionEvaluator.is_winning_move(game_state, action, self.my_color):
                return action

        # Block opponent win detection
        for action in actions:
            if ActionEvaluator.blocks_opponent_win(game_state, action, self.opp_color):
                return action

        # Opening book query
        move_num = len(self.move_history)
        if move_num < 2 and self._check_opening_book(actions, game_state, move_num):
            opening_move = self._get_opening_move(actions, game_state, move_num)
            if opening_move:
                return opening_move

        # Heuristic filtering of candidate actions
        candidates = self._heuristic_filter(actions, game_state)

        # Dynamic time allocation
        importance = self._evaluate_move_importance(game_state, candidates)
        time_budget = self.time_manager.get_time_budget(game_state, importance)

        # Emergency mode: aggressive pruning when time is short
        if self.time_manager.get_remaining_time() < 0.1:
            # Only consider top 5 best actions
            candidates = candidates[:5]
            self.simulation_depth = 2
        elif self.time_manager.get_remaining_time() < time_budget:
            return candidates[0] if candidates else random.choice(actions)

        # MCTS deep search
        try:
            result = self._mcts_search(candidates, game_state)
            # Record performance statistics
            self.performance_stats['decision_times'].append(time.time() - self.time_manager.start_time)
            return result
        except Exception as e:
            print(f"MCTS search error: {e}")
            return candidates[0] if candidates else random.choice(actions)

    def _check_opening_book(self, actions, game_state, move_num):
        """Check if opening book can be used"""
        if move_num >= len(self.opening_book):
            return False

        # Check if recommended positions are available
        if move_num == 0:
            recommendations = self.opening_book[0]
        else:
            last_opp_move = self._get_last_opponent_move(game_state)
            if last_opp_move and last_opp_move in self.opening_book[1]:
                recommendations = self.opening_book[1][last_opp_move]
            else:
                return False

        # Check if any recommended positions are available
        for r, c in recommendations:
            for action in actions:
                if action.get('coords') == (r, c):
                    return True
        return False

    def _get_opening_move(self, actions, game_state, move_num):
        """Get move from opening book"""
        if move_num == 0:
            recommendations = self.opening_book[0]
        else:
            last_opp_move = self._get_last_opponent_move(game_state)
            if not last_opp_move:
                return None
            recommendations = self.opening_book[1].get(last_opp_move, [])

        for r, c in recommendations:
            for action in actions:
                if action.get('coords') == (r, c):
                    return action

        return None

    def _get_last_opponent_move(self, game_state):
        """Get opponent's last move position"""
        # Simplified handling: scan board for opponent pieces
        board = game_state.board.chips
        for r, c in HOTB_COORDS:
            if board[r][c] == self.opp_color:
                return (r, c)
        return None

    def _evaluate_move_importance(self, game_state, candidates):
        """Evaluate importance of current move"""
        if not candidates:
            return 0.5

        # Get best candidate's score
        best_score = ActionEvaluator.evaluate_action_quality(game_state, candidates[0])

        # Lower score (higher quality) means higher importance
        if best_score < -500:  # Extremely high quality action (like winning move)
            return 1.0
        elif best_score < -100:  # High quality action
            return 0.8
        elif best_score < 0:  # Medium quality
            return 0.6
        elif best_score < 50:  # Average quality
            return 0.4
        else:  # Low quality
            return 0.2

    def _heuristic_filter(self, actions, game_state):
        """Filter most promising actions using heuristic evaluation"""
        # Exclude corner positions (unless special actions)
        valid_actions = []
        for a in actions:
            if a.get('type') == 'remove':  # Don't filter remove actions
                valid_actions.append(a)
            elif 'coords' not in a or a['coords'] not in CORNERS:
                valid_actions.append(a)

        if not valid_actions:
            return actions[:1]  # If no valid actions, return first action

        # Batch evaluate and sort
        scored_actions = [(a, ActionEvaluator.evaluate_action_quality(game_state, a)) for a in valid_actions]
        scored_actions.sort(key=lambda x: x[1])

        # Dynamically adjust candidate count
        if self.time_manager.phase == GamePhase.CRITICAL:
            limit = min(20, len(scored_actions))  # Consider more options in critical moments
        elif self.time_manager.get_remaining_time() < 0.3:
            limit = min(8, len(scored_actions))  # Reduce candidates when time is tight
        else:
            limit = self.candidate_limit

        # Return top N candidate actions
        return [a for a, _ in scored_actions[:limit]]

    def _mcts_search(self, candidate_actions, game_state):
        """Analyze candidate actions using MCTS"""
        if not candidate_actions:
            return None

        # Prepare MCTS state
        mcts_state = self._prepare_state_for_mcts(game_state, candidate_actions)
        root = Node(mcts_state)

        # Directly create child nodes for root
        for action in candidate_actions:
            next_state = self.fast_simulate(mcts_state, action)
            child = Node(next_state, parent=root, action=action)
            root.children.append(child)

        # Dynamically set search depth and iteration count
        if self.time_manager.phase == GamePhase.CRITICAL:
            max_iterations = min(SIMULATION_LIMIT * 2, 400)
            self.simulation_depth = 6
        elif self.time_manager.get_remaining_time() < 0.3:
            max_iterations = 50  # Quick decisions when time is tight
            self.simulation_depth = 3
        else:
            max_iterations = SIMULATION_LIMIT
            self.simulation_depth = 4

        # MCTS main loop
        iterations = 0
        time_check_interval = 10  # Check time every 10 iterations

        # Early termination related variables
        best_child_visits = defaultdict(int)
        dominant_threshold = 0.7  # If one choice has 70%+ visits, can terminate early

        while not self.time_manager.is_timeout() and iterations < max_iterations:
            iterations += 1

            # 1. Selection phase
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.select_child()
                if node is None:  # Check for null node
                    break

            if node is None:
                continue

            # 2. Expansion phase
            if node.visits > 0 and not node.is_fully_expanded():
                child = node.expand(self)
                if child:
                    node = child

            # 3. Simulation phase
            value = self._heuristic_guided_simulate(node.state)

            # 4. Backpropagation phase
            while node:
                node.update(value)
                node = node.parent

            # Periodic checks
            if iterations % time_check_interval == 0:
                # Time check
                if self.time_manager.is_timeout():
                    break

                # Early termination check
                if iterations > 50 and root.children:
                    max_visits = max(child.visits for child in root.children)
                    total_visits = sum(child.visits for child in root.children)
                    if total_visits > 0 and max_visits / total_visits > dominant_threshold:
                        # One choice clearly dominates, end early
                        break

        # Record iteration count
        self.performance_stats['mcts_iterations'].append(iterations)

        # Select best action
        if not root.children:
            return candidate_actions[0] if candidate_actions else None

        # Select based on visit count (well-explored choice)
        best_child = max(root.children, key=lambda c: c.visits)

        # Record move history
        if best_child.action and 'coords' in best_child.action:
            self.move_history.append(best_child.action['coords'])

        return best_child.action

    def _heuristic_guided_simulate(self, state):
        """Heuristic-guided MCTS simulation"""
        state_copy = self.custom_shallow_copy(state)
        current_depth = 0

        while current_depth < self.simulation_depth:
            current_depth += 1

            # Get available actions
            if hasattr(state_copy, 'available_actions'):
                actions = state_copy.available_actions
            else:
                try:
                    actions = self.rule.getLegalActions(state_copy, self.id)
                except:
                    actions = []

            if not actions:
                break

            # Adjust heuristic proportion based on game phase
            if self.time_manager.phase == GamePhase.CRITICAL:
                heuristic_prob = 0.95  # More reliant on heuristics in critical moments
            elif current_depth == 1:
                heuristic_prob = 0.9  # First step is more important
            else:
                heuristic_prob = 0.85

            # Select action
            if random.random() < heuristic_prob:
                # Use heuristic to select action
                # Optimization: only evaluate top few most promising actions
                sample_size = min(10, len(actions))
                sampled_actions = actions[:sample_size] if len(actions) <= sample_size else random.sample(actions,
                                                                                                          sample_size)

                scored_actions = [(a, ActionEvaluator.evaluate_action_quality(state_copy, a)) for a in sampled_actions]
                scored_actions.sort(key=lambda x: x[1])
                action = scored_actions[0][0] if scored_actions else random.choice(actions)
            else:
                action = random.choice(actions)

            # Apply action
            state_copy = self.fast_simulate(state_copy, action)

            # Check if terminal state reached (winning)
            if self._check_terminal_state(state_copy):
                break

            # Simulate card selection (specifically for 5-card display variant)
            self._simulate_card_selection(state_copy)

        # Evaluate final state
        return StateEvaluator.evaluate(state_copy)

    def _check_terminal_state(self, state):
        """Check if terminal state reached"""
        board = state.board.chips
        # Simplified check: only check for 5-in-a-row
        for i in range(10):
            for j in range(10):
                if board[i][j] not in [0, '0'] and board[i][j] not in CORNERS:
                    color = board[i][j]
                    for dx, dy in DIRECTIONS:
                        if ActionEvaluator._count_consecutive_fast(board, i, j, dx, dy, color) >= 5:
                            return True
        return False

    def fast_simulate(self, state, action):
        """Fast simulation of action execution"""
        new_state = self.custom_shallow_copy(state)

        # Handle place actions
        if action['type'] == 'place' and 'coords' in action:
            r, c = action['coords']
            # Determine correct color
            if hasattr(state, 'current_player_id') and hasattr(state, 'agents'):
                color = state.agents[state.current_player_id].colour
            else:
                color = self.my_color
            new_state.board.chips[r][c] = color
            self._update_hand(new_state, action)

        # Handle remove actions
        elif action['type'] == 'remove' and 'coords' in action:
            r, c = action['coords']
            new_state.board.chips[r][c] = 0
            self._update_hand(new_state, action)

        return new_state

    def _update_hand(self, state, action):
        """Update hand cards"""
        if 'play_card' not in action:
            return
        card = action['play_card']
        try:
            player_id = getattr(state, 'current_player_id', self.id)
            if (hasattr(state, 'agents') and
                    0 <= player_id < len(state.agents) and
                    hasattr(state.agents[player_id], 'hand')):
                if card in state.agents[player_id].hand:  # check if card exists
                    state.agents[player_id].hand.remove(card)
        except Exception:
            pass

    def _simulate_card_selection(self, state):
        """Simulate selecting a card from 5 display cards"""
        if not (hasattr(state, 'display_cards') and state.display_cards):
            return

        # Use same evaluation logic
        best_card = None
        best_value = float('-inf')

        for card in state.display_cards:
            value = self.card_evaluator._evaluate_card(card, state)
            if value > best_value:
                best_value = value
                best_card = card

        if best_card:
            # Update player hand
            if hasattr(state, 'current_player_id'):
                player_id = state.current_player_id
                if 0 <= player_id < len(state.agents) and hasattr(state.agents[player_id], 'hand'):
                    state.agents[player_id].hand.append(best_card)

            # Remove selected card from display
            if best_card in state.display_cards:  # Check if card exists
                state.display_cards.remove(best_card)

            # Replenish one card (if deck exists)
            if hasattr(state, 'deck') and state.deck:
                state.display_cards.append(state.deck.pop(0))

    def _prepare_state_for_mcts(self, game_state, actions):
        """Prepare game state for MCTS"""
        # Create state copy
        mcts_state = self.custom_shallow_copy(game_state)
        mcts_state.my_color = self.my_color
        mcts_state.opp_color = self.opp_color
        mcts_state.current_player_id = self.id
        mcts_state.available_actions = actions

        return mcts_state

    def custom_shallow_copy(self, state):
        """Optimized state copying"""
        if hasattr(state, 'board') and hasattr(state.board, 'chips'):
            # Efficient shallow copy + deep copy of critical parts
            new_state = copy.copy(state)
            new_state.board = copy.copy(state.board)
            new_state.board.chips = [row[:] for row in state.board.chips]

            # Copy agents list (if exists)
            if hasattr(state, 'agents'):
                new_state.agents = list(state.agents)

            return new_state
        else:
            # Fall back to deep copy
            return copy.deepcopy(state)