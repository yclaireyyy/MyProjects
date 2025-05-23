from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
import random
import time
import copy
import math
import itertools
import heapq
from collections import defaultdict, deque
import statistics

# ===========================
# Constants and Configuration
# ===========================
MAX_THINK_TIME = 0.95
EXPLORATION_WEIGHT = 1.2  # UCB exploration parameter
HOTB_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]  # Center hotspot positions
CORNERS = [(0, 0), (0, 9), (9, 0), (9, 9)]  # Corner positions (free spaces)
SIMULATION_LIMIT = 150  # Maximum MCTS simulations
CANDIDATE_LIMIT = 8  # Number of top actions to consider in MCTS


class GamePhaseAnalyzer:
    """Intelligent game phase detection for dynamic strategy adaptation"""

    @staticmethod
    def analyze_game_phase(state):
        """
        Analyze current game phase and return strategic weights.
        Returns tuple: (phase_name, weight_adjustments)
        """
        if not state or not hasattr(state, 'board'):
            return 'early', GamePhaseAnalyzer._get_early_weights()

        board = state.board.chips

        # Calculate board occupation rate
        total_positions = 100  # 10x10 board
        occupied_positions = sum(1 for i in range(10) for j in range(10)
                                 if SafetyUtils.safe_board_access(board, i, j) not in [0, '0'])
        occupation_rate = occupied_positions / total_positions

        # Detect immediate threats (sequences of length 4)
        threat_level = GamePhaseAnalyzer._assess_threat_level(board)

        # Determine game phase based on multiple factors
        if occupation_rate < 0.15:
            return 'early', GamePhaseAnalyzer._get_early_weights()
        elif occupation_rate > 0.6 or threat_level >= 2:
            return 'endgame', GamePhaseAnalyzer._get_endgame_weights()
        else:
            return 'midgame', GamePhaseAnalyzer._get_midgame_weights()

    @staticmethod
    def _assess_threat_level(board):
        """Count immediate winning threats on board"""
        threat_count = 0
        for i in range(10):
            for j in range(10):
                cell_value = SafetyUtils.safe_board_access(board, i, j)
                if cell_value not in [0, '0']:
                    # Check if this position is part of a 4-sequence
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        consecutive = GamePhaseAnalyzer._count_consecutive_from_pos(board, i, j, dx, dy, cell_value)
                        if consecutive >= 4:
                            threat_count += 1
        return threat_count

    @staticmethod
    def _count_consecutive_from_pos(board, x, y, dx, dy, color):
        """Helper to count consecutive pieces from a specific position"""
        count = 1
        # Forward direction
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if SafetyUtils.safe_board_access(board, nx, ny) == color:
                count += 1
            else:
                break
        # Backward direction
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if SafetyUtils.safe_board_access(board, nx, ny) == color:
                count += 1
            else:
                break
        return min(count, 5)

    @staticmethod
    def _get_early_weights():
        """Weight adjustments for early game - prioritize center control and development"""
        return {
            'center_control': 2.5,  # Heavily prioritize center control
            'position_development': 2.0,  # Focus on spreading pieces
            'sequence_formation': 1.0,  # Normal sequence building
            'defense': 0.7,  # Lower defensive priority
            'blocking': 0.5  # Minimal blocking focus
        }

    @staticmethod
    def _get_midgame_weights():
        """Weight adjustments for mid game - balanced approach"""
        return {
            'center_control': 1.5,  # Moderate center focus
            'position_development': 1.2,  # Continued development
            'sequence_formation': 1.8,  # Increased sequence focus
            'defense': 1.5,  # Increased defensive awareness
            'blocking': 1.2  # Moderate blocking
        }

    @staticmethod
    def _get_endgame_weights():
        """Weight adjustments for endgame - prioritize winning and defense"""
        return {
            'center_control': 0.8,  # Reduced center importance
            'position_development': 0.5,  # Minimal new development
            'sequence_formation': 3.0,  # Maximum sequence priority
            'defense': 2.5,  # High defensive priority
            'blocking': 3.0  # Maximum blocking priority
        }


class OpponentModeler:
    """Advanced opponent behavior analysis and strategic adaptation"""

    def __init__(self):
        self.opponent_history = deque(maxlen=20)  # Track last 20 opponent moves
        self.move_patterns = defaultdict(int)
        self.aggression_indicators = []
        self.defensive_indicators = []
        self.predicted_style = 'balanced'  # 'aggressive', 'defensive', 'balanced'

    def record_opponent_move(self, state_before, action, state_after):
        """Record and analyze opponent's move for pattern recognition"""
        if not action or not state_before or not state_after:
            return

        # Record the move
        move_data = {
            'action': action,
            'timestamp': time.time(),
            'board_before': self._extract_board_state(state_before),
            'board_after': self._extract_board_state(state_after)
        }
        self.opponent_history.append(move_data)

        # Analyze move characteristics
        self._analyze_move_style(action, state_before, state_after)

        # Update opponent style prediction
        self._update_style_prediction()

    def _extract_board_state(self, state):
        """Extract relevant board state information"""
        if hasattr(state, 'board') and hasattr(state.board, 'chips'):
            return [row[:] for row in state.board.chips]
        return None

    def _analyze_move_style(self, action, state_before, state_after):
        """Analyze individual move to determine strategic style indicators"""
        if action.get('type') != 'place' or 'coords' not in action:
            return

        coords = action['coords']
        r, c = SafetyUtils.safe_coords_access(coords)
        if r is None or c is None:
            return

        board_before = state_before.board.chips if hasattr(state_before, 'board') else None
        if not board_before:
            return

        # Check if move creates immediate threat (aggressive indicator)
        if self._creates_immediate_threat(board_before, r, c):
            self.aggression_indicators.append(1)

        # Check if move blocks opponent threat (defensive indicator)
        if self._blocks_opponent_threat(board_before, r, c):
            self.defensive_indicators.append(1)

        # Check if move prioritizes center control (balanced/aggressive indicator)
        if (r, c) in HOTB_COORDS:
            self.aggression_indicators.append(0.7)

        # Limit indicator history to prevent memory bloat
        if len(self.aggression_indicators) > 15:
            self.aggression_indicators.pop(0)
        if len(self.defensive_indicators) > 15:
            self.defensive_indicators.pop(0)

    def _creates_immediate_threat(self, board, r, c):
        """Check if placing at (r,c) creates a winning threat"""
        # Simulate placing piece and check for 4+ sequences
        board_copy = [row[:] for row in board]
        board_copy[r][c] = 'test_color'  # Use placeholder color

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = self._count_consecutive_from_pos(board_copy, r, c, dx, dy, 'test_color')
            if count >= 4:
                return True
        return False

    def _blocks_opponent_threat(self, board, r, c):
        """Check if placing at (r,c) blocks an opponent threat"""
        # Check if any adjacent opponent sequences would be broken
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # Check both directions for opponent sequences
            for direction in [1, -1]:
                opp_count = 0
                for i in range(1, 5):
                    x, y = r + direction * i * dx, c + direction * i * dy
                    cell = SafetyUtils.safe_board_access(board, x, y)
                    if cell not in [0, '0', None] and cell != 'my_color':  # Opponent piece
                        opp_count += 1
                    else:
                        break
                if opp_count >= 3:  # Blocking a strong threat
                    return True
        return False

    def _count_consecutive_from_pos(self, board, x, y, dx, dy, color):
        """Count consecutive pieces from position - helper method"""
        count = 1
        # Forward
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if SafetyUtils.safe_board_access(board, nx, ny) == color:
                count += 1
            else:
                break
        # Backward
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if SafetyUtils.safe_board_access(board, nx, ny) == color:
                count += 1
            else:
                break
        return min(count, 5)

    def _update_style_prediction(self):
        """Update opponent style prediction based on accumulated evidence"""
        if len(self.aggression_indicators) < 3 and len(self.defensive_indicators) < 3:
            return  # Not enough data

        avg_aggression = statistics.mean(self.aggression_indicators) if self.aggression_indicators else 0
        avg_defense = statistics.mean(self.defensive_indicators) if self.defensive_indicators else 0

        # Determine style based on behavioral indicators
        if avg_aggression > 0.7 and avg_aggression > avg_defense * 1.5:
            self.predicted_style = 'aggressive'
        elif avg_defense > 0.6 and avg_defense > avg_aggression * 1.5:
            self.predicted_style = 'defensive'
        else:
            self.predicted_style = 'balanced'

    def get_counter_strategy_weights(self):
        """Return strategic weight adjustments to counter opponent's style"""
        if self.predicted_style == 'aggressive':
            # Counter aggressive opponents with strong defense and blocking
            return {
                'defense_multiplier': 1.8,
                'blocking_multiplier': 2.0,
                'sequence_formation_multiplier': 0.9,
                'center_control_multiplier': 1.2
            }
        elif self.predicted_style == 'defensive':
            # Counter defensive opponents with increased aggression
            return {
                'defense_multiplier': 0.8,
                'blocking_multiplier': 0.7,
                'sequence_formation_multiplier': 1.6,
                'center_control_multiplier': 1.4
            }
        else:  # balanced
            return {
                'defense_multiplier': 1.0,
                'blocking_multiplier': 1.0,
                'sequence_formation_multiplier': 1.0,
                'center_control_multiplier': 1.0
            }


class StrategicPlanner:
    """Advanced multi-move strategic planning and pattern recognition"""

    @staticmethod
    def identify_strategic_patterns(state, agent_id):
        """Identify and evaluate multi-move strategic opportunities"""
        if not state or not hasattr(state, 'board'):
            return {}

        board = state.board.chips
        patterns = {}

        # Identify fork opportunities (threatening multiple sequences simultaneously)
        patterns['fork_opportunities'] = StrategicPlanner._find_fork_opportunities(board, agent_id)

        # Identify sacrifice plays (giving up immediate gain for larger strategic advantage)
        patterns['sacrifice_potential'] = StrategicPlanner._evaluate_sacrifice_potential(board, agent_id)

        # Identify tempo plays (forcing opponent responses)
        patterns['tempo_opportunities'] = StrategicPlanner._find_tempo_plays(board, agent_id)

        # Identify positional advantages (controlling key intersections)
        patterns['positional_strength'] = StrategicPlanner._assess_positional_strength(board, agent_id)

        return patterns

    @staticmethod
    def _find_fork_opportunities(board, agent_id):
        """Find positions that threaten multiple sequences simultaneously"""
        fork_positions = []

        for r in range(10):
            for c in range(10):
                if SafetyUtils.safe_board_access(board, r, c) in [0, '0']:  # Empty position
                    # Check how many potential sequences this position could contribute to
                    sequence_contributions = 0

                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        # Simulate placing piece and count potential sequence length
                        potential_length = StrategicPlanner._calculate_potential_sequence_length(
                            board, r, c, dx, dy, agent_id
                        )
                        if potential_length >= 3:  # Meaningful sequence potential
                            sequence_contributions += 1

                    if sequence_contributions >= 2:  # True fork opportunity
                        fork_positions.append({
                            'position': (r, c),
                            'threat_count': sequence_contributions,
                            'strategic_value': sequence_contributions * 50
                        })

        return sorted(fork_positions, key=lambda x: x['strategic_value'], reverse=True)[:3]

    @staticmethod
    def _calculate_potential_sequence_length(board, r, c, dx, dy, agent_id):
        """Calculate potential sequence length if piece placed at (r,c)"""
        # This is a simplified version - could be enhanced with more sophisticated analysis
        friendly_count = 0

        # Check both directions from the potential placement
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + direction * i * dx, c + direction * i * dy
                cell = SafetyUtils.safe_board_access(board, x, y)
                # Need to determine friendly color based on agent_id
                # This is a simplification - in real implementation would use proper color detection
                if cell and cell not in [0, '0'] and cell == 'friendly_placeholder':
                    friendly_count += 1
                else:
                    break

        return friendly_count + 1  # +1 for the piece we would place

    @staticmethod
    def _evaluate_sacrifice_potential(board, agent_id):
        """Evaluate positions where sacrificing immediate gain leads to larger advantage"""
        # This could involve giving up a good position to force opponent into bad position
        # Or allowing opponent to block one threat while setting up a bigger threat
        sacrifice_opportunities = []

        # Simplified implementation - look for positions that set up multiple future threats
        for r in range(10):
            for c in range(10):
                if SafetyUtils.safe_board_access(board, r, c) in [0, '0']:
                    # Evaluate if this position enables multiple future strong moves
                    future_potential = StrategicPlanner._calculate_future_move_potential(board, r, c)
                    if future_potential > 3:  # High future potential
                        sacrifice_opportunities.append({
                            'position': (r, c),
                            'future_value': future_potential,
                            'sacrifice_score': future_potential * 20
                        })

        return sorted(sacrifice_opportunities, key=lambda x: x['sacrifice_score'], reverse=True)[:2]

    @staticmethod
    def _calculate_future_move_potential(board, r, c):
        """Calculate how many strong future moves this position enables"""
        # Simplified heuristic - count nearby empty positions that could form sequences
        potential_score = 0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    if SafetyUtils.safe_board_access(board, nr, nc) in [0, '0']:
                        potential_score += 1

        return potential_score

    @staticmethod
    def _find_tempo_plays(board, agent_id):
        """Find moves that force opponent to respond defensively"""
        tempo_moves = []

        # Look for moves that create immediate threats opponent must address
        for r in range(10):
            for c in range(10):
                if SafetyUtils.safe_board_access(board, r, c) in [0, '0']:
                    # Check if placing here creates a threat that must be blocked
                    threat_level = StrategicPlanner._assess_threat_creation(board, r, c)
                    if threat_level >= 2:  # Significant threat
                        tempo_moves.append({
                            'position': (r, c),
                            'threat_level': threat_level,
                            'tempo_value': threat_level * 30
                        })

        return sorted(tempo_moves, key=lambda x: x['tempo_value'], reverse=True)[:3]

    @staticmethod
    def _assess_threat_creation(board, r, c):
        """Assess level of threat created by placing at position"""
        # Simplified - count potential sequences of length 3+ that would be formed
        threat_count = 0

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # Simulate placement and check sequence length
            simulated_length = 1  # The piece we place

            # Count in both directions
            for direction in [1, -1]:
                for i in range(1, 5):
                    x, y = r + direction * i * dx, c + direction * i * dy
                    cell = SafetyUtils.safe_board_access(board, x, y)
                    if cell and cell not in [0, '0']:  # Assume friendly for simplification
                        simulated_length += 1
                    else:
                        break

            if simulated_length >= 3:
                threat_count += 1

        return threat_count

    @staticmethod
    def _assess_positional_strength(board, agent_id):
        """Assess control of key board intersections and strategic positions"""
        positional_factors = {
            'center_control': 0,
            'line_intersection_control': 0,
            'corner_approach_control': 0
        }

        # Assess center control
        center_positions = [(3, 3), (3, 6), (6, 3), (6, 6)] + list(HOTB_COORDS)
        controlled_centers = sum(1 for r, c in center_positions
                                 if SafetyUtils.safe_board_access(board, r, c) not in [0, '0'])
        positional_factors['center_control'] = controlled_centers / len(center_positions)

        # Assess line intersection control (positions that affect multiple potential sequences)
        intersection_score = 0
        for r in range(2, 8):  # Focus on middle area
            for c in range(2, 8):
                cell = SafetyUtils.safe_board_access(board, r, c)
                if cell not in [0, '0']:
                    # Count how many lines this position influences
                    line_influence = len([(dx, dy) for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]
                                          if StrategicPlanner._position_influences_line(board, r, c, dx, dy)])
                    intersection_score += line_influence

        positional_factors['line_intersection_control'] = intersection_score / 100  # Normalize

        return positional_factors

    @staticmethod
    def _position_influences_line(board, r, c, dx, dy):
        """Check if position influences a potential sequence line"""
        # Simplified check - see if there are other pieces in this line direction
        for direction in [1, -1]:
            for i in range(1, 5):
                x, y = r + direction * i * dx, c + direction * i * dy
                if SafetyUtils.safe_board_access(board, x, y) not in [0, '0']:
                    return True
        return False


class ProbabilisticEvaluator:
    """Advanced probabilistic evaluation considering uncertainty and risk"""

    @staticmethod
    def evaluate_with_uncertainty(base_score, state, action, confidence_factors=None):
        """
        Enhance base evaluation score with uncertainty analysis.
        Returns: (adjusted_score, confidence_level)
        """
        if confidence_factors is None:
            confidence_factors = ProbabilisticEvaluator._calculate_confidence_factors(state, action)

        # Calculate uncertainty adjustments
        uncertainty_penalty = ProbabilisticEvaluator._calculate_uncertainty_penalty(state, action)
        risk_adjustment = ProbabilisticEvaluator._calculate_risk_adjustment(state, action)

        # Apply probabilistic adjustments
        adjusted_score = base_score * (1 - uncertainty_penalty) + risk_adjustment
        confidence_level = ProbabilisticEvaluator._calculate_confidence_level(confidence_factors)

        return adjusted_score, confidence_level

    @staticmethod
    def _calculate_confidence_factors(state, action):
        """Calculate various factors affecting decision confidence"""
        factors = {
            'board_clarity': 1.0,  # How clear the board state is
            'action_complexity': 1.0,  # How complex the action consequences are
            'opponent_predictability': 0.5,  # How predictable opponent is
            'time_pressure': 1.0  # How much time pressure affects decision
        }

        if not state or not action:
            return factors

        # Calculate board clarity (more pieces = clearer situation)
        if hasattr(state, 'board'):
            board = state.board.chips
            occupied = sum(1 for i in range(10) for j in range(10)
                           if SafetyUtils.safe_board_access(board, i, j) not in [0, '0'])
            factors['board_clarity'] = min(1.0, occupied / 50)  # Normalize to 0-1

        # Calculate action complexity (placement in contested areas is more complex)
        if action.get('type') == 'place' and 'coords' in action:
            coords = action['coords']
            r, c = SafetyUtils.safe_coords_access(coords)
            if r is not None and c is not None:
                # Check neighborhood complexity
                neighbor_complexity = ProbabilisticEvaluator._assess_neighborhood_complexity(state, r, c)
                factors['action_complexity'] = 1.0 - (neighbor_complexity / 8.0)  # 8 max neighbors

        return factors

    @staticmethod
    def _assess_neighborhood_complexity(state, r, c):
        """Assess complexity of 3x3 neighborhood around position"""
        if not hasattr(state, 'board'):
            return 0

        board = state.board.chips
        complexity = 0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    cell = SafetyUtils.safe_board_access(board, nr, nc)
                    if cell not in [0, '0']:
                        complexity += 1

        return complexity

    @staticmethod
    def _calculate_uncertainty_penalty(state, action):
        """Calculate penalty for uncertain outcomes"""
        # Base uncertainty penalty
        base_penalty = 0.1

        # Higher penalty for actions in unclear board areas
        if action and action.get('type') == 'place' and 'coords' in action:
            coords = action['coords']
            r, c = SafetyUtils.safe_coords_access(coords)
            if r is not None and c is not None:
                # Actions near board edges have higher uncertainty
                edge_distance = min(r, c, 9 - r, 9 - c)
                if edge_distance <= 1:
                    base_penalty += 0.05

                # Actions in contested areas have higher uncertainty
                if hasattr(state, 'board'):
                    neighbor_count = ProbabilisticEvaluator._assess_neighborhood_complexity(state, r, c)
                    if neighbor_count >= 4:  # Highly contested
                        base_penalty += 0.1

        return min(base_penalty, 0.3)  # Cap at 30% penalty

    @staticmethod
    def _calculate_risk_adjustment(state, action):
        """Calculate risk-based score adjustment"""
        # Conservative adjustment for high-risk moves
        risk_bonus = 0

        if action and action.get('type') == 'place' and 'coords' in action:
            coords = action['coords']
            r, c = SafetyUtils.safe_coords_access(coords)
            if r is not None and c is not None:
                # Reward moves that secure center positions (lower risk)
                if (r, c) in HOTB_COORDS:
                    risk_bonus += 5

                # Reward moves that complete sequences (despite risk)
                # This would need access to color information - simplified here
                risk_bonus += ProbabilisticEvaluator._estimate_sequence_completion_bonus(state, r, c)

        return risk_bonus

    @staticmethod
    def _estimate_sequence_completion_bonus(state, r, c):
        """Estimate bonus for potential sequence completion"""
        # Simplified estimation - in practice would need proper color analysis
        if not hasattr(state, 'board'):
            return 0

        board = state.board.chips
        bonus = 0

        # Check if position is adjacent to existing pieces (sequence potential)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    if SafetyUtils.safe_board_access(board, nr, nc) not in [0, '0']:
                        bonus += 2  # Bonus for being near existing pieces

        return min(bonus, 10)  # Cap bonus

    @staticmethod
    def _calculate_confidence_level(confidence_factors):
        """Calculate overall confidence level from individual factors"""
        # Weighted average of confidence factors
        weights = {
            'board_clarity': 0.3,
            'action_complexity': 0.3,
            'opponent_predictability': 0.2,
            'time_pressure': 0.2
        }

        weighted_sum = sum(confidence_factors.get(factor, 0.5) * weight
                           for factor, weight in weights.items())

        return max(0.1, min(1.0, weighted_sum))  # Clamp to [0.1, 1.0]


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
    """Enhanced card evaluator with dynamic phase-aware scoring"""

    def __init__(self, agent):
        self.agent = agent

    def evaluate_card(self, card, state):
        """Evaluate card value with phase-aware and probabilistic enhancements"""
        if not card or not state or not hasattr(state, 'board'):
            return 0

        # Get base evaluation
        base_score = self._get_base_card_value(card, state)

        # Apply phase-aware adjustments
        phase, phase_weights = GamePhaseAnalyzer.analyze_game_phase(state)
        phase_adjusted_score = self._apply_phase_adjustments(base_score, card, phase_weights)

        # Apply probabilistic uncertainty analysis
        final_score, confidence = ProbabilisticEvaluator.evaluate_with_uncertainty(
            phase_adjusted_score, state, {'type': 'card_evaluation', 'card': card}
        )

        return final_score

    def _get_base_card_value(self, card, state):
        """Get base card value using original evaluation logic"""
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

    def _apply_phase_adjustments(self, base_score, card, phase_weights):
        """Apply game phase specific adjustments to card evaluation"""
        if self._is_two_eyed_jack(card):
            # Two-eyed Jacks are always valuable but more so in endgame
            return base_score * phase_weights.get('sequence_formation', 1.0)
        elif self._is_one_eyed_jack(card):
            # One-eyed Jacks become more valuable when blocking is important
            return base_score * phase_weights.get('blocking', 1.0)
        else:
            # Regular cards benefit from center control in early game
            return base_score * phase_weights.get('center_control', 1.0)

    def _exponential_card_evaluation(self, card, state):
        """Enhanced exponential scoring with strategic pattern recognition"""
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

            # Calculate base exponential score
            position_score = self._calculate_position_score(board, r, c)

            # Add strategic pattern bonuses
            strategic_bonus = self._calculate_strategic_bonus(state, r, c)

            total_score += position_score + strategic_bonus

        # Return average score if multiple positions available
        return total_score / max(1, len(positions))

    def _calculate_strategic_bonus(self, state, r, c):
        """Calculate bonus based on strategic patterns this card enables"""
        # Identify strategic patterns this position contributes to
        patterns = StrategicPlanner.identify_strategic_patterns(state, self.agent.id)

        bonus = 0

        # Bonus for fork opportunities
        for fork in patterns.get('fork_opportunities', []):
            if fork['position'] == (r, c):
                bonus += fork['strategic_value']

        # Bonus for tempo plays
        for tempo in patterns.get('tempo_opportunities', []):
            if tempo['position'] == (r, c):
                bonus += tempo['tempo_value']

        return bonus

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
    """Enhanced action evaluation with phase awareness and opponent modeling"""

    @staticmethod
    def heuristic(state, action, agent_id=None, opponent_model=None):
        """Enhanced A* heuristic with dynamic adaptation"""
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

        # Get base heuristic score
        base_score = ActionEvaluator._calculate_base_heuristic(state, action, agent_id)

        # Apply phase-aware adjustments
        phase, phase_weights = GamePhaseAnalyzer.analyze_game_phase(state)
        phase_adjusted_score = ActionEvaluator._apply_phase_adjustments(base_score, phase_weights, action, state)

        # Apply opponent-aware adjustments
        if opponent_model:
            counter_weights = opponent_model.get_counter_strategy_weights()
            opponent_adjusted_score = ActionEvaluator._apply_opponent_adjustments(
                phase_adjusted_score, counter_weights, action, state
            )
        else:
            opponent_adjusted_score = phase_adjusted_score

        # Apply probabilistic uncertainty
        final_score, confidence = ProbabilisticEvaluator.evaluate_with_uncertainty(
            opponent_adjusted_score, state, action
        )

        return max(0, 200 - final_score)  # Convert to heuristic (lower is better)

    @staticmethod
    def _calculate_base_heuristic(state, action, agent_id):
        """Calculate base heuristic score using original logic"""
        coords = action.get('coords')
        r, c = SafetyUtils.safe_coords_access(coords)
        board = state.board.chips

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
                return 0

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

        return score

    @staticmethod
    def _apply_phase_adjustments(base_score, phase_weights, action, state):
        """Apply game phase specific weight adjustments"""
        adjusted_score = base_score

        coords = action.get('coords')
        r, c = SafetyUtils.safe_coords_access(coords)

        # Apply center control weight
        if (r, c) in HOTB_COORDS:
            center_bonus = 15 * phase_weights.get('center_control', 1.0)
            adjusted_score += center_bonus - 15  # Replace original bonus

        # Apply sequence formation weight (affects all sequence-related scoring)
        sequence_multiplier = phase_weights.get('sequence_formation', 1.0)
        if sequence_multiplier != 1.0:
            # Identify sequence-related portion of score and adjust
            sequence_portion = ActionEvaluator._calculate_sequence_portion(base_score, action, state)
            adjusted_score += sequence_portion * (sequence_multiplier - 1.0)

        # Apply defense weight
        defense_multiplier = phase_weights.get('defense', 1.0)
        if defense_multiplier != 1.0:
            defense_portion = ActionEvaluator._calculate_defense_portion(base_score, action, state)
            adjusted_score += defense_portion * (defense_multiplier - 1.0)

        return adjusted_score

    @staticmethod
    def _apply_opponent_adjustments(base_score, counter_weights, action, state):
        """Apply opponent-specific strategic adjustments"""
        adjusted_score = base_score

        # Apply multipliers from opponent model
        defense_mult = counter_weights.get('defense_multiplier', 1.0)
        blocking_mult = counter_weights.get('blocking_multiplier', 1.0)
        sequence_mult = counter_weights.get('sequence_formation_multiplier', 1.0)
        center_mult = counter_weights.get('center_control_multiplier', 1.0)

        # Adjust based on action characteristics
        coords = action.get('coords')
        r, c = SafetyUtils.safe_coords_access(coords)

        # Center control adjustment
        if (r, c) in HOTB_COORDS and center_mult != 1.0:
            center_bonus_adjustment = 15 * (center_mult - 1.0)
            adjusted_score += center_bonus_adjustment

        # Defense/blocking adjustments would need more sophisticated analysis
        # of whether this specific action is defensive/blocking

        return adjusted_score

    @staticmethod
    def _calculate_sequence_portion(base_score, action, state):
        """Estimate what portion of base score comes from sequence formation"""
        # Simplified estimation - in practice would need more detailed analysis
        return base_score * 0.4  # Assume 40% of score is sequence-related

    @staticmethod
    def _calculate_defense_portion(base_score, action, state):
        """Estimate what portion of base score comes from defensive play"""
        # Simplified estimation
        return base_score * 0.3  # Assume 30% of score is defense-related

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
    """Enhanced state evaluation with comprehensive strategic analysis"""

    @staticmethod
    def evaluate(state, agent_id, last_action=None, opponent_model=None):
        """Enhanced state evaluation with phase awareness and strategic patterns"""
        if not state:
            return 0

        board = state.board.chips if hasattr(state, 'board') else None
        if not board:
            return 0

        # Get base evaluation
        base_score = StateEvaluator._calculate_base_evaluation(state, agent_id)

        # Apply phase-aware adjustments
        phase, phase_weights = GamePhaseAnalyzer.analyze_game_phase(state)
        phase_adjusted_score = StateEvaluator._apply_phase_adjustments(base_score, phase_weights)

        # Add strategic pattern analysis
        strategic_bonus = StateEvaluator._calculate_strategic_bonus(state, agent_id)

        # Apply opponent modeling adjustments
        if opponent_model:
            opponent_adjustment = StateEvaluator._calculate_opponent_adjustment(state, agent_id, opponent_model)
        else:
            opponent_adjustment = 0

        # Combine all factors
        total_score = phase_adjusted_score + strategic_bonus + opponent_adjustment

        # Apply probabilistic uncertainty
        final_score, confidence = ProbabilisticEvaluator.evaluate_with_uncertainty(
            total_score, state, last_action if last_action else {'type': 'state_evaluation'}
        )

        # Normalize to [-1, 1] range
        return max(-1, min(1, final_score / 200))

    @staticmethod
    def _calculate_base_evaluation(state, agent_id):
        """Calculate base evaluation using original logic"""
        board = state.board.chips

        # Get player colors
        if hasattr(state, 'my_color'):
            my_color = state.my_color
            opp_color = state.opp_color
        else:
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
        return position_score + sequence_score + defense_score + hotb_score

    @staticmethod
    def _apply_phase_adjustments(base_score, phase_weights):
        """Apply phase-specific weight adjustments to evaluation components"""
        # This is a simplified implementation - in practice would break down
        # base_score into components and apply specific weights

        # Calculate average weight factor
        avg_weight = statistics.mean(phase_weights.values())

        # Apply graduated adjustment based on phase characteristics
        return base_score * avg_weight

    @staticmethod
    def _calculate_strategic_bonus(state, agent_id):
        """Calculate bonus based on strategic patterns and long-term positioning"""
        patterns = StrategicPlanner.identify_strategic_patterns(state, agent_id)

        bonus = 0

        # Fork opportunity bonus
        fork_count = len(patterns.get('fork_opportunities', []))
        bonus += fork_count * 20

        # Tempo opportunity bonus
        tempo_count = len(patterns.get('tempo_opportunities', []))
        bonus += tempo_count * 15

        # Positional strength bonus
        positional_factors = patterns.get('positional_strength', {})
        bonus += positional_factors.get('center_control', 0) * 30
        bonus += positional_factors.get('line_intersection_control', 0) * 25

        return bonus

    @staticmethod
    def _calculate_opponent_adjustment(state, agent_id, opponent_model):
        """Calculate adjustments based on opponent behavior model"""
        adjustment = 0

        # Adjust based on predicted opponent style
        style = opponent_model.predicted_style

        if style == 'aggressive':
            # Against aggressive opponents, reward defensive positioning
            adjustment += StateEvaluator._calculate_defensive_positioning_bonus(state, agent_id)
        elif style == 'defensive':
            # Against defensive opponents, reward aggressive positioning
            adjustment += StateEvaluator._calculate_aggressive_positioning_bonus(state, agent_id)
        # Balanced opponents get no special adjustment

        return adjustment

    @staticmethod
    def _calculate_defensive_positioning_bonus(state, agent_id):
        """Calculate bonus for defensive positioning against aggressive opponents"""
        # Simplified implementation - reward having pieces that can block multiple threats
        if not hasattr(state, 'board'):
            return 0

        board = state.board.chips
        bonus = 0

        # Get player color
        if hasattr(state, 'agents') and 0 <= agent_id < len(state.agents):
            my_color = state.agents[agent_id].colour
        else:
            return 0

        # Reward pieces in positions that can defend multiple lines
        for i in range(10):
            for j in range(10):
                if SafetyUtils.safe_board_access(board, i, j) == my_color:
                    # Count how many potential opponent threats this position can block
                    blocking_potential = 0
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        # Simplified threat detection
                        if StateEvaluator._can_block_threat_in_direction(board, i, j, dx, dy, my_color):
                            blocking_potential += 1

                    if blocking_potential >= 2:  # Can block multiple threat directions
                        bonus += 10

        return bonus

    @staticmethod
    def _calculate_aggressive_positioning_bonus(state, agent_id):
        """Calculate bonus for aggressive positioning against defensive opponents"""
        # Simplified implementation - reward having multiple active threats
        if not hasattr(state, 'board'):
            return 0

        board = state.board.chips
        bonus = 0

        # Get player color
        if hasattr(state, 'agents') and 0 <= agent_id < len(state.agents):
            my_color = state.agents[agent_id].colour
        else:
            return 0

        # Count active threats (sequences of 3+ that could extend to 5)
        threat_count = 0
        for i in range(10):
            for j in range(10):
                if SafetyUtils.safe_board_access(board, i, j) == my_color:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        consecutive = ActionEvaluator._count_consecutive(board, i, j, dx, dy, my_color)
                        if consecutive >= 3:  # Active threat
                            threat_count += 1

        bonus += threat_count * 8  # Reward having multiple threats

        return bonus

    @staticmethod
    def _can_block_threat_in_direction(board, r, c, dx, dy, my_color):
        """Check if position (r,c) can potentially block threats in given direction"""
        # Simplified check - look for potential opponent sequences this position interrupts
        enemy_color = 'r' if my_color == 'b' else 'b'

        # Check both sides of this position for enemy pieces
        enemy_before = 0
        enemy_after = 0

        # Check one direction
        for i in range(1, 4):
            x, y = r + i * dx, c + i * dy
            if SafetyUtils.safe_board_access(board, x, y) == enemy_color:
                enemy_after += 1
            else:
                break

        # Check opposite direction
        for i in range(1, 4):
            x, y = r - i * dx, c - i * dy
            if SafetyUtils.safe_board_access(board, x, y) == enemy_color:
                enemy_before += 1
            else:
                break

        # If enemies on both sides, this position blocks a potential sequence
        return enemy_before > 0 and enemy_after > 0

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
    """Enhanced action simulation with strategic context tracking"""

    def __init__(self, agent):
        self.agent = agent

    def simulate_action(self, state, action):
        """Simulate executing an action and return new state"""
        new_state = SafetyUtils.safe_copy_state(state)

        if action['type'] == 'place':
            self._simulate_place(new_state, action)
        elif action['type'] == 'remove':
            self._simulate_remove(new_state, action)

        # Track action in opponent model if this is opponent's move
        if hasattr(self.agent, 'opponent_model') and hasattr(state, 'current_player_id'):
            if state.current_player_id != self.agent.id:
                self.agent.opponent_model.record_opponent_move(state, action, new_state)

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
    """Enhanced MCTS tree node with strategic context"""

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

        # Enhanced statistics for better decision making
        self.confidence_sum = 0.0  # Track confidence levels
        self.strategic_value = 0.0  # Track strategic pattern value

        # Lazy initialization of untried actions
        self.untried_actions = None

    def get_untried_actions(self, agent):
        """Get untried actions with enhanced heuristic sorting"""
        if self.untried_actions is None:
            # Initialize untried actions list
            if hasattr(self.state, 'available_actions'):
                self.untried_actions = list(self.state.available_actions)
            else:
                try:
                    self.untried_actions = agent.rule.getLegalActions(self.state, agent.id)
                except:
                    self.untried_actions = []

            # Sort by enhanced heuristic (best actions first)
            opponent_model = getattr(agent, 'opponent_model', None)
            self.untried_actions.sort(
                key=lambda a: ActionEvaluator.heuristic(self.state, a, agent.id, opponent_model)
            )

        return self.untried_actions

    def is_fully_expanded(self, agent):
        """Check if all possible actions have been tried"""
        return len(self.get_untried_actions(agent)) == 0

    def select_child(self):
        """Select most promising child using enhanced UCB formula"""
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            if child.visits == 0:
                score = float('inf')  # Prioritize unvisited children
            else:
                # Standard UCB calculation
                exploitation = child.value / child.visits
                exploration = EXPLORATION_WEIGHT * math.sqrt(2 * math.log(self.visits) / child.visits)

                # Add confidence weighting
                avg_confidence = child.confidence_sum / child.visits if child.visits > 0 else 0.5
                confidence_factor = 0.5 + 0.5 * avg_confidence  # Scale between 0.5 and 1.0

                # Add strategic value weighting
                strategic_factor = 1.0 + (child.strategic_value / 100.0)  # Small bonus for strategic moves

                score = (exploitation + exploration) * confidence_factor * strategic_factor

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, agent):
        """Expand node by adding one new child with strategic analysis"""
        untried = self.get_untried_actions(agent)

        if not untried:
            return None

        # Take the best untried action (list is pre-sorted)
        action = untried.pop(0)

        # Create new state by simulating the action
        new_state = agent.action_simulator.simulate_action(self.state, action)

        # Create and add child node
        child = Node(new_state, parent=self, action=action)

        # Calculate strategic value for this node
        if hasattr(agent, 'calculate_strategic_value'):
            child.strategic_value = agent.calculate_strategic_value(new_state, action)

        self.children.append(child)

        return child

    def update(self, result, confidence=0.5):
        """Update node statistics with enhanced tracking"""
        self.visits += 1
        self.value += result
        self.confidence_sum += confidence


class TimeManager:
    """Enhanced time management with adaptive strategies"""

    def __init__(self):
        self.start_time = 0
        self.time_budget = MAX_THINK_TIME
        self.decision_history = deque(maxlen=10)  # Track recent decision times

    def start_timing(self):
        """Start timing the current decision"""
        self.start_time = time.time()

    def get_remaining_time(self):
        """Get remaining time in budget"""
        elapsed = time.time() - self.start_time
        return self.time_budget - elapsed

    def is_timeout(self, buffer=0.05):
        """Check if we're approaching time limit with adaptive buffer"""
        # Adjust buffer based on recent decision complexity
        adaptive_buffer = buffer
        if len(self.decision_history) > 3:
            avg_decision_time = statistics.mean(self.decision_history)
            if avg_decision_time > 0.8:  # Recent decisions have been slow
                adaptive_buffer = buffer + 0.02  # Increase buffer

        return self.get_remaining_time() < adaptive_buffer

    def should_use_quick_mode(self):
        """Determine if we should use quick decision mode"""
        return self.get_remaining_time() < 0.3

    def record_decision_time(self):
        """Record time taken for this decision"""
        if self.start_time > 0:
            decision_time = time.time() - self.start_time
            self.decision_history.append(decision_time)


class myAgent(Agent):
    """Advanced agent with comprehensive strategic intelligence"""

    def __init__(self, _id):
        """Initialize the agent with all enhanced components"""
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)  # Two-player game
        self.counter = itertools.count()  # For A* search unique identifiers

        # Initialize enhanced components
        self.card_evaluator = CardEvaluator(self)
        self.action_simulator = ActionSimulator(self)
        self.time_manager = TimeManager()
        self.opponent_model = OpponentModeler()  # New: opponent behavior tracking

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
        """Enhanced strategic card selection with phase awareness"""
        trade_actions = [a for a in actions if a and a.get('type') == 'trade']

        if not trade_actions:
            return random.choice(actions) if actions else None

        if not hasattr(game_state, 'display_cards') or not game_state.display_cards:
            return random.choice(trade_actions)

        # Enhanced evaluation with phase awareness
        best_card = None
        best_score = float('-inf')

        for card in game_state.display_cards:
            if card:
                # Use enhanced card evaluation
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
        """Enhanced main decision function with intelligent strategy selection"""
        if not actions:
            return None

        self.time_manager.start_timing()
        self._initialize_colors(game_state)

        # Handle card selection phase
        if self._is_card_selection(actions):
            selected_action = self._select_strategic_card(actions, game_state)
            self.time_manager.record_decision_time()
            return selected_action

        # Intelligent strategy selection based on game complexity and available time
        action_count = len(actions)
        remaining_time = self.time_manager.get_remaining_time()

        # Determine optimal search strategy
        if action_count <= 3 or remaining_time < 0.2:
            # Quick heuristic-only decision for simple situations
            selected_action = self._quick_heuristic_decision(actions, game_state)
        elif action_count <= 8 or remaining_time < 0.5:
            # A* search for moderate complexity
            selected_action = self.a_star_search(game_state, actions)
        else:
            # Full MCTS with A* filtering for complex situations
            candidates = self._filter_candidates(actions, game_state)
            selected_action = self._mcts_search(candidates, game_state)

        self.time_manager.record_decision_time()
        return selected_action

    def _quick_heuristic_decision(self, actions, game_state):
        """Quick decision using only heuristic evaluation"""
        if not actions:
            return None

        best_action = None
        best_score = float('inf')  # Lower is better for heuristic

        for action in actions:
            score = ActionEvaluator.heuristic(game_state, action, self.id, self.opponent_model)
            if score < best_score:
                best_score = score
                best_action = action

        return best_action if best_action else actions[0]

    def a_star_search(self, initial_state, candidate_moves):
        """Enhanced A* search with opponent modeling"""
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
            h = ActionEvaluator.heuristic(initial_state, move, self.id, self.opponent_model)
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

            # Enhanced state evaluation
            reward = StateEvaluator.evaluate(current_state, self.id, last_move, self.opponent_model)
            if reward > top_reward:
                top_reward = reward
                best_sequence = move_history

            # Get next possible actions
            try:
                next_steps = self.rule.getLegalActions(current_state, self.id)
                if next_steps:
                    # Sort by enhanced heuristic and limit search width
                    next_steps.sort(
                        key=lambda act: ActionEvaluator.heuristic(current_state, act, self.id, self.opponent_model))

                    for next_move in next_steps[:5]:  # Limit search width
                        if next_move:
                            next_g = g + 1
                            next_h = ActionEvaluator.heuristic(current_state, next_move, self.id, self.opponent_model)
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
        """Enhanced candidate filtering with strategic pattern analysis"""
        # Remove obviously poor actions (like corners)
        valid_actions = [a for a in actions
                         if 'coords' not in a or a['coords'] not in CORNERS]

        if not valid_actions:
            return actions[:1]  # Fallback to first action if no valid ones

        # Enhanced scoring with strategic patterns
        scored_actions = []
        for action in valid_actions:
            # Base heuristic score
            heuristic_score = ActionEvaluator.heuristic(game_state, action, self.id, self.opponent_model)

            # Strategic pattern bonus
            strategic_bonus = self._calculate_strategic_bonus(game_state, action)

            # Combined score (lower is better for heuristic, so subtract bonus)
            combined_score = heuristic_score - strategic_bonus

            scored_actions.append((action, combined_score))

        # Sort by combined score and return top candidates
        scored_actions.sort(key=lambda x: x[1])
        return [a for a, _ in scored_actions[:self.candidate_limit]]

    def _calculate_strategic_bonus(self, state, action):
        """Calculate strategic pattern bonus for action filtering"""
        if not action or action.get('type') != 'place' or 'coords' not in action:
            return 0

        coords = action['coords']
        r, c = SafetyUtils.safe_coords_access(coords)
        if r is None or c is None:
            return 0

        patterns = StrategicPlanner.identify_strategic_patterns(state, self.id)
        bonus = 0

        # Check if this action contributes to any strategic patterns
        for fork in patterns.get('fork_opportunities', []):
            if fork['position'] == (r, c):
                bonus += 20  # High bonus for fork opportunities

        for tempo in patterns.get('tempo_opportunities', []):
            if tempo['position'] == (r, c):
                bonus += 15  # Good bonus for tempo plays

        return bonus

    def _mcts_search(self, candidate_actions, game_state):
        """Enhanced MCTS search with all optimizations"""
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

            # Simulation phase - evaluate leaf node with uncertainty
            value, confidence = self._enhanced_guided_simulation(node.state)

            # Backpropagation phase - update all ancestors with confidence
            while node:
                node.update(value, confidence)
                node = node.parent

        # Select best action based on enhanced criteria
        if not root.children:
            return candidate_actions[0] if candidate_actions else None

        # Select based on visit count weighted by confidence
        best_child = max(root.children,
                         key=lambda c: c.visits * (c.confidence_sum / max(c.visits, 1)))
        return best_child.action

    def _enhanced_guided_simulation(self, state):
        """Enhanced simulation with uncertainty and strategic awareness"""
        current_state = SafetyUtils.safe_copy_state(state)
        depth = 0
        total_confidence = 0
        confidence_count = 0

        while depth < self.simulation_depth:
            depth += 1

            # Get available actions
            try:
                actions = self.rule.getLegalActions(current_state, self.id)
            except:
                actions = []

            if not actions:
                break

            # Enhanced action selection with probabilistic weighting
            if random.random() < 0.8:  # 80% strategic, 20% random
                # Use enhanced heuristic to guide action selection
                scored_actions = [(a, ActionEvaluator.heuristic(current_state, a, self.id, self.opponent_model))
                                  for a in actions]
                scored_actions.sort(key=lambda x: x[1])
                action = scored_actions[0][0] if scored_actions else random.choice(actions)
            else:
                action = random.choice(actions)

            # Apply action and continue
            current_state = self.action_simulator.simulate_action(current_state, action)

            # Simulate opponent card selection if needed
            self._simulate_card_selection(current_state)

            # Track confidence for this simulation step
            _, step_confidence = ProbabilisticEvaluator.evaluate_with_uncertainty(
                0, current_state, action
            )
            total_confidence += step_confidence
            confidence_count += 1

        # Enhanced final state evaluation
        final_value = StateEvaluator.evaluate(current_state, self.id, None, self.opponent_model)
        avg_confidence = total_confidence / max(confidence_count, 1)

        return final_value, avg_confidence

    def calculate_strategic_value(self, state, action):
        """Calculate strategic value for node enhancement"""
        patterns = StrategicPlanner.identify_strategic_patterns(state, self.id)

        strategic_value = 0

        # Value from strategic patterns
        strategic_value += len(patterns.get('fork_opportunities', [])) * 25
        strategic_value += len(patterns.get('tempo_opportunities', [])) * 20

        # Positional value
        positional_strength = patterns.get('positional_strength', {})
        strategic_value += positional_strength.get('center_control', 0) * 30

        return strategic_value

    def _simulate_card_selection(self, state):
        """Enhanced card selection simulation with phase awareness"""
        if not (hasattr(state, 'display_cards') and state.display_cards):
            return

        # Use enhanced card evaluation
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
        """Prepare game state for MCTS with enhanced attributes"""
        mcts_state = SafetyUtils.safe_copy_state(game_state)

        # Add essential attributes for MCTS
        mcts_state.my_color = self.my_color
        mcts_state.opp_color = self.opp_color
        mcts_state.current_player_id = self.id
        mcts_state.available_actions = actions

        return mcts_state