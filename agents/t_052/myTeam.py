from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import random
import time
import copy
import math
import itertools
import heapq

# ===========================
# Constants and Configuration
# ===========================
MAX_THINK_TIME = 0.95
EXPLORATION_WEIGHT = 1.2  # UCB exploration parameter
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]  # Center hotspot positions
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]
SIMULATION_LIMIT = 150  # Maximum MCTS simulations
CANDIDATE_LIMIT = 8  # Number of top actions to consider in MCTS


class SafetyUtils:
    """Utility class for safe operations and error handling"""

    @staticmethod
    def safe_board_access(board, r, c, default=0):
        """Safely access board positions with bounds checking"""
        try:
            if board and 0 <= r < len(board) and 0 <= c < len(board[0]):
                return board[r][c]
        except (IndexError, TypeError):
            pass
        return default

    @staticmethod
    def safe_coords_access(coords):
        """Safely extract coordinates from action"""
        if coords and len(coords) == 2:
            try:
                r, c = coords
                if isinstance(r, (int, float)) and isinstance(c, (int, float)):
                    return int(r), int(c)
            except (ValueError, TypeError):
                pass
        return None, None

    @staticmethod
    def safe_copy_state(state):
        """Safely create a deep copy of game state"""
        try:
            if hasattr(state, "copy"):
                return state.copy()
            else:
                return copy.deepcopy(state)
        except Exception:
            return state


class CardEvaluator:
    """Enhanced card evaluator with robust error handling"""

    def __init__(self, agent):
        self.agent = agent

    def evaluate_card(self, card, state):
        """Evaluate card value in current state with comprehensive scoring"""
        if not card or not state or not hasattr(state, 'board'):
            return 0

        board = state.board.chips

        # Priority 1: Two-eyed Jacks (Wild cards) - highest value
        if self._is_two_eyed_jack(card):
            return 10000

        # Priority 2: One-eyed Jacks (Remove opponent pieces) - high value
        if self._is_one_eyed_jack(card):
            return 5000

        # Priority 3: Regular cards - use exponential evaluation
        if card in COORDS:
            return self._exponential_card_evaluation(card, state)

        return 0

    def _exponential_card_evaluation(self, card, state):
        """Advanced exponential scoring for regular cards"""
        if card not in COORDS:
            return 0

        board = state.board.chips
        total_score = 0

        # Get all possible positions for this card
        positions = COORDS[card] if isinstance(COORDS[card], list) else [COORDS[card]]

        for pos in positions:
            if len(pos) != 2:
                continue

            r, c = pos
            # Check if position is available
            if not self._is_position_available(board, r, c):
                continue

            # Calculate exponential score for this position
            position_score = self._calculate_position_score(board, r, c)
            total_score += position_score

        # Return average score if multiple positions available
        return total_score / max(1, len(positions))

    def _calculate_position_score(self, board, r, c):
        """Calculate exponential score for a single position"""
        my_color = self._get_my_color()
        if not my_color:
            return 0

        total_score = 0
        # Four main directions: horizontal, vertical, main diagonal, anti-diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            my_pieces = self._count_my_pieces_in_direction(board, r, c, dx, dy)
            direction_score = self._exponential_scoring(my_pieces)
            total_score += direction_score

        return total_score

    def _count_my_pieces_in_direction(self, board, r, c, dx, dy):
        """Count friendly pieces in a specific direction within 5-position window"""
        my_color = self._get_my_color()
        if not my_color or not board:
            return 0

        my_pieces = 0
        # Check 4 positions in each direction (excluding center position)
        for i in range(-4, 5):
            if i == 0:  # Skip center position (where we'll place)
                continue

            x, y = r + i * dx, c + i * dy
            # Bounds checking
            if 0 <= x < 10 and 0 <= y < 10:
                if SafetyUtils.safe_board_access(board, x, y) == my_color:
                    my_pieces += 1

        return my_pieces

    def _exponential_scoring(self, piece_count):
        """Exponential scoring: 1=10, 2=100, 3=1000, 4+=10000"""
        if piece_count == 0:
            return 1  # Base score
        elif piece_count == 1:
            return 10
        elif piece_count == 2:
            return 100
        elif piece_count == 3:
            return 1000
        elif piece_count >= 4:
            return 10000  # Near-winning position
        return 0

    def _get_my_color(self):
        """Get agent's color with fallback mechanisms"""
        if hasattr(self.agent, 'my_color') and self.agent.my_color:
            return self.agent.my_color
        elif hasattr(self.agent, 'colour'):
            return self.agent.colour
        return None

    def _is_two_eyed_jack(self, card):
        """Check if card is a two-eyed Jack (wild card)"""
        if not card:
            return False
        card_str = str(card).lower()
        return card_str in ['jc', 'jd']  # Jack of Clubs/Diamonds

    def _is_one_eyed_jack(self, card):
        """Check if card is a one-eyed Jack (remove card)"""
        if not card:
            return False
        card_str = str(card).lower()
        return card_str in ['js', 'jh']  # Jack of Spades/Hearts

    def _is_position_available(self, board, r, c):
        """Check if board position is available for placement"""
        if not board or not (0 <= r < 10 and 0 <= c < 10):
            return False
        cell_value = SafetyUtils.safe_board_access(board, r, c)
        return cell_value == 0 or cell_value == '0'  # Empty position


class ActionEvaluator:
    """Action evaluation functions from version 1 with safety enhancements"""

    @staticmethod
    def heuristic(state, action, agent_id=None):
        """A* heuristic function - evaluate action potential value (lower is better)"""
        if action.get('type') != 'place' or 'coords' not in action:
            return 100  # Non-placement action gets neutral score

        coords = action.get('coords')
        r, c = SafetyUtils.safe_coords_access(coords)
        if r is None or c is None:
            return 100

        if (r, c) in CORNERS:
            return 100  # Corner positions are generally poor

        board = state.board.chips if hasattr(state, 'board') else None
        if not board:
            return 100

        # Get player colors
        if hasattr(state, 'my_color'):
            color = state.my_color
            enemy = state.opp_color
        else:
            # Infer from agent context
            agent_id = agent_id or (state.current_player_id if hasattr(state, 'current_player_id') else 0)
            if (hasattr(state, 'agents') and 0 <= agent_id < len(state.agents)):
                color = state.agents[agent_id].colour
                enemy = 'r' if color == 'b' else 'b'
            else:
                return 100

        # Create hypothetical board after placement
        board_copy = [row[:] for row in board]
        board_copy[r][c] = color

        score = 0

        # Center preference
        distance = abs(r - 4.5) + abs(c - 4.5)
        score += max(0, 5 - distance) * 2

        # Sequence formation scoring
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = ActionEvaluator._count_consecutive(board_copy, r, c, dx, dy, color)
            if count >= 5:
                score += 200  # Winning sequence
            elif count == 4:
                score += 100
            elif count == 3:
                score += 30
            elif count == 2:
                score += 10

        # Defensive scoring
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            enemy_threat = ActionEvaluator._count_enemy_threat(board, r, c, dx, dy, enemy)
            if enemy_threat >= 3:
                score += 50  # High priority block

        # Center control scoring
        hotb_controlled = sum(1 for x, y in HOTB_COORDS if SafetyUtils.safe_board_access(board_copy, x, y) == color)
        score += hotb_controlled * 15

        # Convert to heuristic score (lower is better)
        return 100 - score

    @staticmethod
    def _count_consecutive(board, x, y, dx, dy, color):
        """Count consecutive pieces of same color from position (x,y) in direction (dx,dy)"""
        count = 1  # Starting position counts as one

        # Check forward direction
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if SafetyUtils.safe_board_access(board, nx, ny) == color:
                count += 1
            else:
                break

        # Check backward direction
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if SafetyUtils.safe_board_access(board, nx, ny) == color:
                count += 1
            else:
                break

        return min(count, 5)  # Cap at 5 (one complete sequence)

    @staticmethod
    def _count_enemy_threat(board, r, c, dx, dy, enemy):
        """Count enemy threat level in a direction"""
        enemy_chain = 0
        for i in range(1, 5):
            x, y = r + dx * i, c + dy * i
            if SafetyUtils.safe_board_access(board, x, y) == enemy:
                enemy_chain += 1
            else:
                break

        for i in range(1, 5):
            x, y = r - dx * i, c - dy * i
            if SafetyUtils.safe_board_access(board, x, y) == enemy:
                enemy_chain += 1
            else:
                break

        return enemy_chain


class StateEvaluator:
    """State evaluation functions from version 1 with enhancements"""

    @staticmethod
    def evaluate(state, agent_id, last_action=None):
        """Evaluate game state value with comprehensive factors"""
        if not state:
            return 0

        board = state.board.chips if hasattr(state, 'board') else None
        if not board:
            return 0

        # Get player colors
        if hasattr(state, 'my_color'):
            my_color = state.my_color
            opp_color = state.opp_color
        else:
            # Infer from state
            if (hasattr(state, 'agents') and 0 <= agent_id < len(state.agents)):
                my_color = state.agents[agent_id].colour
                opp_color = 'r' if my_color == 'b' else 'b'
            else:
                return 0

        # 1. Position scoring
        position_score = 0
        for i in range(10):
            for j in range(10):
                cell_value = SafetyUtils.safe_board_access(board, i, j)
                if cell_value == my_color:
                    if (i, j) in HOTB_COORDS:
                        position_score += 1.5  # Center positions
                    elif i in [0, 9] or j in [0, 9]:
                        position_score += 0.8  # Edge positions
                    else:
                        position_score += 1.0  # Regular positions
                elif cell_value == opp_color:
                    if (i, j) in HOTB_COORDS:
                        position_score -= 1.5
                    elif i in [0, 9] or j in [0, 9]:
                        position_score -= 0.8
                    else:
                        position_score -= 1.0

        # 2. Sequence potential scoring
        sequence_score = StateEvaluator._calculate_sequence_score(board, my_color)

        # 3. Defense scoring
        defense_score = StateEvaluator._calculate_defense_score(board, opp_color)

        # 4. Center control scoring
        hotb_score = 0
        for x, y in HOTB_COORDS:
            cell_value = SafetyUtils.safe_board_access(board, x, y)
            if cell_value == my_color:
                hotb_score += 5
            elif cell_value == opp_color:
                hotb_score -= 5

        # 5. Combined score
        total_score = position_score + sequence_score + defense_score + hotb_score

        # Normalize to [-1, 1] range
        return max(-1, min(1, total_score / 200))

    @staticmethod
    def _calculate_sequence_score(board, color):
        """Calculate sequence formation potential"""
        sequence_score = 0
        for i in range(10):
            for j in range(10):
                if SafetyUtils.safe_board_access(board, i, j) == color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = ActionEvaluator._count_consecutive(board, i, j, dx, dy, color)
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
    def _calculate_defense_score(board, opp_color):
        """Calculate defensive necessity against opponent threats"""
        defense_score = 0
        for i in range(10):
            for j in range(10):
                if SafetyUtils.safe_board_access(board, i, j) == opp_color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = ActionEvaluator._count_consecutive(board, i, j, dx, dy, opp_color)
                        if count >= 4:
                            defense_score -= 50
                        elif count == 3:
                            defense_score -= 10
        return defense_score


class ActionSimulator:
    """Action simulation functions from version 1 with safety enhancements"""

    def __init__(self, agent):
        self.agent = agent

    def simulate_action(self, state, action):
        """Simulate executing an action and return new state"""
        new_state = SafetyUtils.safe_copy_state(state)

        if action['type'] == 'place':
            self._simulate_place(new_state, action)
        elif action['type'] == 'remove':
            self._simulate_remove(new_state, action)

        return new_state

    def _simulate_place(self, state, action):
        """Simulate placement action"""
        if 'coords' not in action:
            return

        coords = action['coords']
        r, c = SafetyUtils.safe_coords_access(coords)
        if r is None or c is None:
            return

        color = self._get_current_color(state)
        if (hasattr(state, 'board') and hasattr(state.board, 'chips') and
                0 <= r < len(state.board.chips) and 0 <= c < len(state.board.chips[0])):
            state.board.chips[r][c] = color

        # Update hand
        self._update_hand(state, action)

    def _simulate_remove(self, state, action):
        """Simulate removal action"""
        if 'coords' not in action:
            return

        coords = action['coords']
        r, c = SafetyUtils.safe_coords_access(coords)
        if r is None or c is None:
            return

        if (hasattr(state, 'board') and hasattr(state.board, 'chips') and
                0 <= r < len(state.board.chips) and 0 <= c < len(state.board.chips[0])):
            state.board.chips[r][c] = 0  # Remove piece

        # Update hand
        self._update_hand(state, action)

    def _get_current_color(self, state):
        """Get current player color"""
        if hasattr(state, 'current_player_id'):
            player_id = state.current_player_id
            if (hasattr(state, 'agents') and 0 <= player_id < len(state.agents) and
                    hasattr(state.agents[player_id], 'colour')):
                return state.agents[player_id].colour
        return self.agent.my_color

    def _update_hand(self, state, action):
        """Update player's hand after playing a card"""
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


class Node:
    """MCTS tree node with enhanced selection and expansion"""

    def __init__(self, state, parent=None, action=None):
        # State representation
        self.state = SafetyUtils.safe_copy_state(state)

        # Tree structure
        self.parent = parent
        self.children = []
        self.action = action

        # MCTS statistics
        self.visits = 0
        self.value = 0.0

        # Lazy initialization of untried actions
        self.untried_actions = None

    def get_untried_actions(self, agent):
        """Get untried actions with heuristic sorting"""
        if self.untried_actions is None:
            # Initialize untried actions list
            if hasattr(self.state, 'available_actions'):
                self.untried_actions = list(self.state.available_actions)
            else:
                try:
                    self.untried_actions = agent.rule.getLegalActions(self.state, agent.id)
                except:
                    self.untried_actions = []

            # Sort by heuristic (best actions first)
            self.untried_actions.sort(
                key=lambda a: ActionEvaluator.heuristic(self.state, a, agent.id)
            )

        return self.untried_actions

    def is_fully_expanded(self, agent):
        """Check if all possible actions have been tried"""
        return len(self.get_untried_actions(agent)) == 0

    def select_child(self):
        """Select most promising child using UCB formula"""
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            if child.visits == 0:
                score = float('inf')  # Prioritize unvisited children
            else:
                # Standard UCB calculation with heuristic enhancement
                exploitation = child.value / child.visits
                exploration = EXPLORATION_WEIGHT * math.sqrt(2 * math.log(self.visits) / child.visits)

                # Add small heuristic bias
                heuristic_factor = 1.0
                if child.action:
                    heuristic_score = ActionEvaluator.heuristic(self.state, child.action, 0)
                    heuristic_factor = 1.0 / (1.0 + heuristic_score / 200)

                score = exploitation + exploration * heuristic_factor

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, agent):
        """Expand node by adding one new child"""
        untried = self.get_untried_actions(agent)

        if not untried:
            return None

        # Take the best untried action (list is pre-sorted)
        action = untried.pop(0)

        # Create new state by simulating the action
        new_state = agent.action_simulator.simulate_action(self.state, action)

        # Create and add child node
        child = Node(new_state, parent=self, action=action)
        self.children.append(child)

        return child

    def update(self, result):
        """Update node statistics with simulation result"""
        self.visits += 1
        self.value += result


class TimeManager:
    """Enhanced time management with adaptive strategies"""

    def __init__(self):
        self.start_time = 0
        self.time_budget = MAX_THINK_TIME

    def start_timing(self):
        """Start timing the current decision"""
        self.start_time = time.time()

    def get_remaining_time(self):
        """Get remaining time in budget"""
        elapsed = time.time() - self.start_time
        return self.time_budget - elapsed

    def is_timeout(self, buffer=0.05):
        """Check if we're approaching time limit"""
        return self.get_remaining_time() < buffer

    def should_use_quick_mode(self):
        """Determine if we should use quick decision mode"""
        return self.get_remaining_time() < 0.3


class myAgent(Agent):
    """Complete agent combining MCTS, A*, and advanced evaluation"""

    def __init__(self, _id):
        """Initialize the agent with all functional components"""
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)  # Two-player game
        self.counter = itertools.count()  # For A* search unique identifiers

        # Initialize all components
        self.card_evaluator = CardEvaluator(self)
        self.action_simulator = ActionSimulator(self)
        self.time_manager = TimeManager()

        # Player colors (initialized lazily)
        self.my_color = None
        self.opp_color = None

        # Search parameters
        self.simulation_depth = 5
        self.candidate_limit = CANDIDATE_LIMIT

        # A* search timing
        self.start_time = 0

    def _initialize_colors(self, game_state):
        """Initialize player color information"""
        if self.my_color is None and game_state and hasattr(game_state, 'agents'):
            if 0 <= self.id < len(game_state.agents):
                agent = game_state.agents[self.id]
                if hasattr(agent, 'colour'):
                    self.my_color = agent.colour
                    self.opp_color = 'r' if self.my_color == 'b' else 'b'

    def _is_card_selection(self, actions):
        """Determine if this is a card selection phase"""
        return any(a and a.get('type') == 'trade' for a in actions) if actions else False

    def _select_strategic_card(self, actions, game_state):
        """Strategic card selection using advanced evaluation"""
        trade_actions = [a for a in actions if a and a.get('type') == 'trade']

        if not trade_actions:
            return random.choice(actions) if actions else None

        if not hasattr(game_state, 'display_cards') or not game_state.display_cards:
            return random.choice(trade_actions)

        # Evaluate all available cards
        best_card = None
        best_score = float('-inf')

        for card in game_state.display_cards:
            if card:
                score = self.card_evaluator.evaluate_card(card, game_state)
                if score > best_score:
                    best_score = score
                    best_card = card

        # Find corresponding action
        if best_card:
            for action in trade_actions:
                if action.get('draft_card') == best_card:
                    return action

        return random.choice(trade_actions)

    def SelectAction(self, actions, game_state):
        """Main decision function with hybrid search strategy"""
        if not actions:
            return None

        self.time_manager.start_timing()
        self._initialize_colors(game_state)

        # Handle card selection phase
        if self._is_card_selection(actions):
            return self._select_strategic_card(actions, game_state)

        # Decide between A* and MCTS based on available time and action count
        if len(actions) <= 5 or self.time_manager.should_use_quick_mode():
            return self.a_star_search(game_state, actions)
        else:
            # Filter candidates and use MCTS for deeper analysis
            candidates = self._filter_candidates(actions, game_state)
            return self._mcts_search(candidates, game_state)

    def a_star_search(self, initial_state, candidate_moves):
        """Complete A* search implementation from version 2"""
        pending = []
        seen_states = set()
        best_sequence = []
        top_reward = float('-inf')

        self.start_time = time.time()

        # Initialize priority queue with candidate moves
        for move in candidate_moves:
            if not move:
                continue

            g = 1
            h = ActionEvaluator.heuristic(initial_state, move, self.id)
            f = g + h
            heapq.heappush(pending, (f, next(self.counter), g, h,
                                     self.action_simulator.simulate_action(initial_state, move), [move]))

        while pending and (time.time() - self.start_time < MAX_THINK_TIME):
            f, _, g, h, current_state, move_history = heapq.heappop(pending)

            if not current_state or not move_history:
                continue

            last_move = move_history[-1]

            # State deduplication
            state_signature = self.get_state_signature(current_state, last_move)
            if state_signature in seen_states:
                continue
            seen_states.add(state_signature)

            # Evaluate current state
            reward = StateEvaluator.evaluate(current_state, self.id, last_move)
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history

            # Get next possible actions
            try:
                next_steps = self.rule.getLegalActions(current_state, self.id)
                if next_steps:
                    # Sort by heuristic and limit search width
                    next_steps.sort(key=lambda act: ActionEvaluator.heuristic(current_state, act, self.id))

                    for next_move in next_steps[:5]:  # Limit search width
                        if next_move:
                            next_g = g + 1
                            next_h = ActionEvaluator.heuristic(current_state, next_move, self.id)
                            next_state = self.action_simulator.simulate_action(current_state, next_move)
                            heapq.heappush(pending, (
                                next_g + next_h, next(self.counter),
                                next_g, next_h, next_state,
                                move_history + [next_move]
                            ))
            except:
                continue

        return best_sequence[0] if best_sequence else (candidate_moves[0] if candidate_moves else None)

    def get_state_signature(self, state, last_move):
        """Generate state signature for deduplication"""
        try:
            board_hash = self.board_hash(state)
            play_card = last_move.get('play_card') if last_move else None

            # Safe hand access
            hand_tuple = ()
            if (hasattr(state, 'agents') and
                    0 <= self.id < len(state.agents) and
                    hasattr(state.agents[self.id], 'hand')):
                hand = state.agents[self.id].hand
                hand_tuple = tuple(hand) if hand else ()

            return (board_hash, play_card, hand_tuple)
        except:
            return (id(state), str(last_move))

    def board_hash(self, state):
        """Generate board hash for state signature"""
        try:
            if hasattr(state, 'board') and hasattr(state.board, 'chips'):
                return tuple(tuple(row) for row in state.board.chips)
            return id(state)
        except:
            return id(state)

    def _filter_candidates(self, actions, game_state):
        """Filter top candidate actions using heuristic evaluation"""
        # Remove obviously poor actions (like corners)
        valid_actions = [a for a in actions
                         if 'coords' not in a or a['coords'] not in CORNERS]

        if not valid_actions:
            return actions[:1]  # Fallback to first action if no valid ones

        # Score and sort actions
        scored_actions = []
        for action in valid_actions:
            score = ActionEvaluator.heuristic(game_state, action, self.id)
            scored_actions.append((action, score))

        # Sort by score (lower is better) and return top candidates
        scored_actions.sort(key=lambda x: x[1])
        return [a for a, _ in scored_actions[:self.candidate_limit]]

    def _mcts_search(self, candidate_actions, game_state):
        """MCTS search with advanced evaluation and time management"""
        # Prepare state for MCTS
        mcts_state = self._prepare_mcts_state(game_state, candidate_actions)
        root = Node(mcts_state)

        # Pre-create children for candidate actions
        for action in candidate_actions:
            next_state = self.action_simulator.simulate_action(mcts_state, action)
            child = Node(next_state, parent=root, action=action)
            root.children.append(child)

        # MCTS main loop
        iterations = 0
        while not self.time_manager.is_timeout() and iterations < SIMULATION_LIMIT:
            iterations += 1

            # Selection phase - traverse tree to promising leaf
            node = root
            while node.is_fully_expanded(self) and node.children:
                node = node.select_child()

            # Expansion phase - add new child if possible
            if node.visits > 0 and not node.is_fully_expanded(self):
                expanded_child = node.expand(self)
                if expanded_child:
                    node = expanded_child

            # Simulation phase - evaluate leaf node
            value = self._guided_simulation(node.state)

            # Backpropagation phase - update all ancestors
            while node:
                node.update(value)
                node = node.parent

        # Select best action based on visit count
        if not root.children:
            return candidate_actions[0] if candidate_actions else None

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _guided_simulation(self, state):
        """Simulation guided by heuristics for better evaluation"""
        current_state = SafetyUtils.safe_copy_state(state)
        depth = 0

        while depth < self.simulation_depth:
            depth += 1

            # Get available actions
            try:
                actions = self.rule.getLegalActions(current_state, self.id)
            except:
                actions = []

            if not actions:
                break

            # Choose action: 80% heuristic-guided, 20% random (from version 1)
            if random.random() < 0.8:
                # Use heuristic to guide action selection
                scored_actions = [(a, ActionEvaluator.heuristic(current_state, a, self.id))
                                  for a in actions]
                scored_actions.sort(key=lambda x: x[1])
                action = scored_actions[0][0] if scored_actions else random.choice(actions)
            else:
                action = random.choice(actions)

            # Apply action and continue
            current_state = self.action_simulator.simulate_action(current_state, action)

            # Simulate opponent card selection if needed
            self._simulate_card_selection(current_state)

        # Evaluate final state
        return StateEvaluator.evaluate(current_state, self.id)

    def _simulate_card_selection(self, state):
        """Simulate intelligent card selection from display"""
        if not (hasattr(state, 'display_cards') and state.display_cards):
            return

        # Use same evaluation logic as main card selection
        best_card = None
        best_value = float('-inf')

        for card in state.display_cards:
            value = self.card_evaluator.evaluate_card(card, state)
            if value > best_value:
                best_value = value
                best_card = card

        if best_card:
            # Update current player's hand
            player_id = getattr(state, 'current_player_id', self.id)
            if (hasattr(state, 'agents') and
                    0 <= player_id < len(state.agents) and
                    hasattr(state.agents[player_id], 'hand')):
                state.agents[player_id].hand.append(best_card)

            # Remove from display and replenish if possible
            try:
                state.display_cards.remove(best_card)
                if hasattr(state, 'deck') and state.deck:
                    state.display_cards.append(state.deck.pop(0))
            except (ValueError, IndexError):
                pass

    def _prepare_mcts_state(self, game_state, actions):
        """Prepare game state for MCTS with necessary attributes"""
        mcts_state = SafetyUtils.safe_copy_state(game_state)

        # Add essential attributes for MCTS
        mcts_state.my_color = self.my_color
        mcts_state.opp_color = self.opp_color
        mcts_state.current_player_id = self.id
        mcts_state.available_actions = actions

        return mcts_state