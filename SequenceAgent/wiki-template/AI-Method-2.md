# AI Method 2 - Computational Approach

Your notes about this part of the project, including acknowledgement, comments, strengths and limitations, etc.

You **do not** need to explain the algorithm. Please tell us how you used it and how you applied it in your team.

If you use greed best first search, then, you can explain about what is the problem (state space model, especially how you define the state, how your define the goal), and heuristic function (as specific as possible) that you used. 

If you use MCTS, then, you can explain about what tree policy/simulation policy you used, how many iteration did you run, what is your reward function, the depth of each simulation etc.

# Table of Contents
  * [Motivation](#motivation)
  * [Application](#application)
  * [Solved challenges](#solved-challenges)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)
 
### Motivation  
To balance planning depth and efficiency, We use A* to manage following advantages:
1. Lookahead Planning. A* allows the agent to simulate sequences of actions, now just immediate rewards.
2. By using a domain-specific heuristic, like favoring central control, penalizing opponent threats and rewarding chain growth, the search is directed toward promising actions.
3. Simulation Accuracy: Each candidate action is simulated using the game’s transition model to avoid illegal or suboptimal moves
4. Controlling the center of the board in Sequence offers the highest flexibility, as central positions can connect to multiple directions (horizontal, vertical, and both diagonals). Moves played near the center have greater long-term potential for forming sequences and blocking opponents.
5. The primary goal of the game is to create sequences of five chips in a row.  Recognizing partial chains and maximizing their growth potential is crucial.  We distinguish between live chains (both ends open) and blocked chains (at least one end blocked), and prioritize live chains more heavily.
6. The four central HOTB (Heart of the Board) positions are especially valuable because if controlled simultaneously, they may count as an extra sequence.  Securing these early can be a winning strategy.

[Back to top](#table-of-contents)

### Application  
A* search is applied during the action selection phase of the agent’s turn.  Given the current game state and a list of legal actions, A* explores future action sequences by simulating game progress and estimating the outcome through a domain-specific evaluation function.
1. Evaluate multi-step action consequences
2. Plan ahead by simulating potential opponent responses and counter-strategies.
3. Select the best initial move from a sequence that leads to the highest predicted board score.
4. Initialization of Candidate Moves:
For each legal candidate move, I compute a cost g = 1 and an estimated heuristic value h = self.heuristic(state, move). These are combined as f = g + h, and the tuple (f, g, h, state, move_history) is pushed into a priority queue (heapq) sorted by f.
    1. Main Search Loop (Time-Limited):
While within the maximum allowed think time (MAX_THINK_TIME), I iteratively pop the lowest-f node from the queue. I then simulate the result of applying the move using self.fast_simulate(state, move) to generate a lightweight copy of the new state.
To avoid re-expanding duplicate states, a state_signature—constructed from the hashed board state, played card, and current hand—is stored in a seen_states set.
    2. Evaluating and Updating Best Path:
I evaluate each simulated state using self.evaluate_state(state, move), which scores the board based on strategic features like center control, potential chains, enemy blocking, and HOTB coverage. If the reward is better than the current best, I update the best_sequence.
    3. Successor Generation and Pruning:
For each expanded state, I use self.rule.getLegalActions() to generate legal moves, sort them by their heuristic values, and only explore the top 5 to limit branching and maintain efficiency.
    4. Termination and Output:
When time runs out or the queue is exhausted, I return the first move in the best sequence found so far.

Key Features and Strengths
	•	Forward Planning: Uses A* to explore multi-step strategies rather than short-sighted greedy actions.
	•	Domain-Aware Heuristics: Scoring includes central positioning (center_bias), formation chains (chain_score), enemy threat blocking (block_enemy_score), and HOTB control (hotb_score).
	•	Efficiency Techniques: Applies state caching (self.state_cache), pruning, and duplicate detection to keep computation within time limits.
	•	Rule-Safe Simulation: Relies on GameRule.generateSuccessor or a lightweight simulation (fast_simulate) to ensure legal state transitions.

This A* approach significantly outperformed random and rule-based baselines in experiments, demonstrating the power of heuristic-guided search in tactical board game environments like Sequence.


[Back to top](#table-of-contents)

### Solved Challenges
Ar first, the A* is low efficiency and poor performance, because it evaluated moves based solely on short term heuristics. This will cause invalid moves and missing critical opportunities.
To solve this, we:
1. Introduced a domain-specific evaluation function that considers sequence formation, blocking enemy chains, and HOTB control.
2. Added center bias and enemy threat assessment to make the heuristic more strategic.
These improvements led to a significant increase in win rate.

[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  
1. By incorporating domain-specific heuristics, the agent makes smarter and more strategic moves.
2. The modular design of the evaluation function makes it easy to adjust and improve in the future.


#### *Disadvantages*
1. A* search with multiple heuristics and lookahead requires more time per move compared to random or greedy baselines.
2.  The search depth must be shallow (e.g., 2–3 moves ahead), otherwise the agent may time out, especially in the late game when the board is full.

[Back to top](#table-of-contents)

### Future improvements  
1. Currently, the search depth is shallow due to time limits.  A time-adaptive depth adjustment mechanism could allow deeper planning in earlier rounds and quicker moves in late game.
2. Heuristic weights (e.g., center bias, enemy block, chain score) are currently manually set.  These could be optimized through automated tuning or self-play learning.
3. Better hashing or board abstraction may help reduce redundant exploration and speed up evaluation.

[Back to top](#table-of-contents)
