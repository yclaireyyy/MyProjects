from template import Agent
import random
import time
from collections import defaultdict


class myAgent(Agent):
    def __init__(self, _id):
        """Agent initialization constructor
        Args:
            _id (int): Player identifier (0 or 1)
        """
        super().__init__(_id)
        self.id = _id  # Player ID (0/1)
        self.my_color = None
        self.opponent_color = None
        self.start_time = time.time()  # Decision timer start
        self.current_phase = "early"  # Game stage tracking
        self.position_strategy = self._init_position_strategy()
        self.sequence_cache = defaultdict(int)  # Sequence pattern cache
        self.timeout_limit = 1.0

    # ===== Strategy Initialization Module =====
    def _init_position_strategy(self):
        return {
            'early': self._generate_center_heatmap(),  # Early game: center focus
            'mid': self._generate_balanced_heatmap(),  # Mid game: balanced (TODO)
            'late': self._generate_aggressive_heatmap()  # Late game: edge focus
        }

    def _generate_center_heatmap(self):
        """Generate center-focused position values using Manhattan distance"""
        heatmap = {}
        for x in range(10):
            for y in range(10):
                # Calculate distance to virtual center (4.5,4.5)
                distance = abs(x - 4.5) + abs(y - 4.5)
                # Value formula: 10 - distance (minimum 0)
                heatmap[(x, y)] = max(10 - distance, 0)
        return heatmap

    def _generate_aggressive_heatmap(self):
        """Generate edge-focused position values with corner bonuses
            Set corner_value = 2, edge_value = 1,
            corner_weight = 3, edge_weight = 2
        """
        heatmap = {}
        for x in range(10):
            for y in range(10):
                corner_bonus = 2 if (x in [0, 9] and y in [0, 9]) else 0
                edge_bonus = 1 if x in [0, 9] or y in [0, 9] else 0
                # Composite formula: corner*3 + edge*2
                heatmap[(x, y)] = corner_bonus * 3 + edge_bonus * 2
        return heatmap

    # ===== Game State Analysis Module =====
    def _update_game_phase(self, game_state):
        """Dynamically update game phase based on placed chips"""
        placed_chips = sum(1 for row in self._get_board_state(game_state)
                           for cell in row if cell != -1)
        # Phase thresholds
        if placed_chips < 15:
            self.game_phase = "early"
        elif placed_chips < 30:
            self.game_phase = "mid"
        else:
            self.game_phase = "late"

    def _get_board_state(self, game_state):
        """Convert board to numerical matrix
        1 = own, 0 = opponent, -1 = empty"""
        try:
            return [[1 if chip == self.my_color else
                     0 if chip == self.opponent_color else -1
                     for chip in row]
                    for row in game_state.board.chips]
        except Exception as e:
            return [[-1] * 10 for _ in range(10)]  # Fallback empty board

    # ===== Core Decision Module =====
    def _find_winning_move(self, actions, board):
        """Immediate win detection"""
        for action in actions:
            x, y = action['coords']
            # Simulate move on temp board
            temp_board = [row[:] for row in board]
            temp_board[x][y] = 1
            if self._check_immediate_win(temp_board):
                return action
        return None

    def _check_immediate_win(self, board):
        """5-consecutive chip detection
        Detect the consecutive five pieces in four directions:
        1. Horizontal →
        2. Vertical ↓
        3. Main diagonal ↘
        4. Secondary diagonal ↖
        """
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for x in range(10):
            for y in range(10):
                if board[x][y] != 1: continue
                for dx, dy in directions:
                    try:
                        if all(board[x + dx * i][y + dy * i] == 1 for i in range(5)):
                            return True
                    except IndexError:
                        continue
        return False

    # ===== Threat Detection Module =====
    def _enhanced_block_detection(self, actions, board):
        """Multi-dimensional threat analysis
        Detect three threat patterns:
        1. Four consecutive gaps (missing one piece to five consecutive)
        2. Three consecutive gaps (with empty Spaces in the middle)
        3. Double triple structure (two potential triple connections)
        """
        threat_actions = []
        for action in actions:
            x, y = action['coords']
            temp_board = [row[:] for row in board]
            temp_board[x][y] = 0  # Simulate opponent move

            threat_score = 0
            for dx, dy in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                # Pattern 1: 4-consecutive gap
                if self._count_consecutive(temp_board, x, y, dx, dy, 0) >= 4:
                    threat_score += 100
                # Pattern 2: 3-consecutive with gap (TODO)
                # Pattern 3: Double-three formation (TODO)

            if threat_score > 0:
                threat_actions.append((action, threat_score))

        return max(threat_actions, key=lambda x: x[1])[0] if threat_actions else None

    def _count_consecutive(self, board, x, y, dx, dy, target):
        """Count consecutive chips in specific direction"""
        count = 1
        # Forward direction
        i = 1
        while 0 <= x + dx * i < 10 and 0 <= y + dy * i < 10:
            if board[x + dx * i][y + dy * i] == target:
                count += 1
                i += 1
            else:
                break
        # Backward direction
        i = 1
        while 0 <= x - dx * i < 10 and 0 <= y - dy * i < 10:
            if board[x - dx * i][y - dy * i] == target:
                count += 1
                i += 1
            else:
                break
        return count

    # ===== Dynamic Evaluation Module =====
    def _dynamic_evaluation(self, action, board):
        """Comprehensive action scoring
        Score = Base Value + Phase Weight * Offense + (1-Weight) * Defense
        """
        x, y = action['coords']
        base_score = self.position_strategy[self.game_phase].get((x, y), 0)
        offensive_score = self._calculate_potential(x, y, board, 1) * 2
        defensive_score = self._calculate_potential(x, y, board, 0) * 1.5
        phase_weights = {'early': 0.3, 'mid': 0.5, 'late': 0.7}
        return base_score + phase_weights[self.game_phase] * offensive_score + \
               (1 - phase_weights[self.game_phase]) * defensive_score

    def _calculate_potential(self, x, y, board, target):
        """Multi-directional sequence potential
        1. Horizontal →
        2. Vertical ↓
        3. Main diagonal ↘
        4. Secondary diagonal ↖
        """
        max_potential = 0
        for dx, dy in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
            line = []
            for i in range(-4, 5):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < 10 and 0 <= ny < 10:
                    line.append(1 if board[nx][ny] == target else 0)
            max_potential = max(max_potential, self._evaluate_line_potential(line))
        return max_potential

    def _evaluate_line_potential(self, line):
        """Line evaluation with gap tolerance"""
        max_score = current = gaps = 0
        for n in line:
            if n == 1:
                current += 1
                max_score = max(max_score, current + gaps)
            else:
                if current > 0 and gaps < 1:
                    gaps += 1
                else:
                    current = gaps = 0
        return max_score

    # ===== Main Decision Entry =====
    def SelectAction(self, actions, game_state):
        """Decision pipeline with priority:
        1. Immediate win
        2. Threat blocking
        3. Optimized evaluation
        4. Random fallback
        """
        # Color initialization
        if self.my_color is None:
            self.my_color = game_state.agents[self.id].colour
            self.opponent_color = game_state.agents[1 - self.id].colour

        # Phase update
        self._update_game_phase(game_state)

        # Card selection handling
        if any('draft_card' in a for a in actions):
            return self._select_strategic_card(actions)

        # Board analysis
        board = self._get_board_state(game_state)

        # Decision priority
        if win_action := self._find_winning_move(actions, board):
            return win_action

        if block_action := self._enhanced_block_detection(actions, board):
            return block_action

        # Dynamic evaluation
        try:
            evaluated_actions = [(a, self._dynamic_evaluation(a, board)) for a in actions]
            return max(evaluated_actions, key=lambda x: x[1])[0]
        except:
            return random.choice(actions) if actions else None

    def _select_strategic_card(self, actions):
        """Strategic card selection priority:
        1. Center-related cards
        2. Flexible jokers
        3. Random selection
        """
        center_cards = ['5h', '4h', '6h', '5d', '5c', '5s']
        for action in actions:
            if action['play_card'] in center_cards:
                return action

        flexible_cards = ['js', 'jh']
        for action in actions:
            if action['play_card'] in flexible_cards:
                return action

        return random.choice(actions)