# AI Method 3 - MCTS

# Table of Contents
  * [Motivation](#motivation)
  * [Application](#application)
  * [Solved challenges](#solved-challenges)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)

### Motivation  
The motivation for choosing to use Monte Carlo Tree Search (MCTS) in the Sequence game is based on an in-depth analysis of the game's characteristics and considerations of computational limitations:
 - Complexity analysis of the state space: It can be seen that the Sequence game is played on a 10x10 chessboard, and each position may have multiple states. More importantly, the number of optional actions for each step is usually large. It can be seen from the processing of the actions list that the algorithm needs to select from a large number of candidate actions. The traditional complete search method is difficult to reach sufficient depth within a time limit of 0.95 seconds in such a search space.
 - Practical considerations of time constraints: The code sets a strict time limit of MAX_THINK_TIME = 1 seconds, which requires the algorithm to make decisions within an extremely short period of time. The anytime algorithm of MCTS enables it to stop the search at anytime and give the current optimal decision, which is crucial for satisfying real-time constraints.
 - Processing requirements for uncertain factors: There are hidden information such as hand cards in the game. So the algorithm needs to handle the randomness of card selection. The random simulation feature of MCTS is naturally suitable for handling this uncertainty and statistically evaluating the advantages and disadvantages of strategies through a large number of simulations.
 - The necessity of balancing exploration and utilization: The UCB formula in the code, exploitation + EXPLORATION_WEIGHT * sqrt(2 * log_visits/child_visits), reflects this requirement. This balancing mechanism can prevent the algorithm from falling into local optimum too early, while ensuring that promising branches are fully explored.

[Back to top](#table-of-contents)

### Application  
**Problem modeling**
**1. State space model**
 - Status: Defined by the following components
 - 10x10 game board status (Each position can be 0/ empty, 'r'/ red, 'b'/ blue)
 - The current player's hand cards
 - Players of the current round
 - Available display cards (if applicable)
 - Action space:
   - Place pieces: {' type ':' place ', 'coords: (r, c),' play_card: card}
   - Remove pieces: {' type ', 'remove', 'coords: (r, c),' play_card: card}
   - Select the card: {'type': 'trade', 'draft_card': card}
   Objective: To form five sub-lines (horizontally, vertically, or diagonally)

**Implementation details of MCTS**
 - Tree Policy
   - Use the UCB1 formula: UCB = Average reward + C * sqrt(ln(number of visits to the parent node)/Number of visits to the child node)
   - The exploration weight C = 1.2, balancing exploration and utilization
   - Combine heuristic evaluation to adjust the UCB score and improve the search efficiency

 - Simulation Policy
   - euristic guided random simulation**
   - Use 95% heuristic selection in the critical stage and 85% in the general case
   - Give priority to high-quality actions and avoid complete randomness

 - Reward function based on multi-factor state assessment
   - Location value: Central area (1.5 weight), marginal area (0.8 weight)
   - Sequence potential: Exponential scoring system (1 =10 points, 2 =100 points, 3 =1000 points, 4 =10000 points)
   - Defensive value: Prevent the opponent from forming a threatening sequence
   - Central control: Control the rewards of hotspots (4,4)-(5,5)

 - Number of iterations and depth
   - Under normal circumstances: 200 iterations, simulation depth of 4 layers
   - Critical moment: 400 iterations, simulation depth of 6 layers
   - Time is tight: 50 iterations, simulation depth of 3 layers
   - Dynamic adjustment is based on game stages and remaining time

**Key Optimization Techniques**
1. The implementation of the three-layer LRU cache architecture
   - The code implements a complete LRU (Least Recently Used) caching system to avoid duplicate calculations. This system contains three dedicated cache instances: ActionEvaluator, CardEvaluator, and StateEvaluator, each with carefully designed capacity limits
   - The core mechanism of LRU caching tracks the access order through a double-ended queue (deque). When the cache reaches the upper capacity limit, the item that has not been used for the longest time will be automatically removed. The design of the cache key takes into account the essential characteristics of the computation. For example, action evaluation uses (board_hash, r, c) as the key to ensure that the evaluations at the same position in the same chessboard state can be directly reused.
2. Hierarchical heuristic action filtering strategy
   - The code implements a systematic action quality assessment system, and scores all candidate actions through the ActionEvaluator.evaluate_action_quality method. This evaluation system adopts a hierarchical priority design:
   - First, there is the immediate success test. When an action can form five consecutive moves, a high score of 2000 points will be awarded. Secondly, there is threat detection. 500 points are awarded when a 4-match threat can be formed, and 1000 points are awarded when a 4-match threat can be prevented from the opponent. The third level is the sequence construction assessment. Three consecutive rounds earn 100 points, and two consecutive rounds earn 20 points. Finally, there is the position value assessment, which gives the base score by calculating the distance from the center of the chessboard.
   - This hierarchical evaluation reduces the number of candidate actions from all possible actions originally to a maximum of 20 (at critical moments) or 12 (under normal circumstances), significantly improving the search efficiency.
3. Time allocation mechanism for game stage perception
   - The AdaptiveTimeManager class implements intelligent time allocation based on the progress of the game. This system determines the current game stage by analyzing the number of pieces on the chessboard:
   - The system also includes a critical state detection mechanism, which identifies key decision points by scanning the chessboard to look for four consecutive threats. Once the critical state is detected, the time budget will increase to 0.9 seconds, the number of MCTS iterations will increase accordingly to 400 times, and the simulation depth will also increase from 4 layers to 6 layers.
4. Efficient state replication strategy
   - The traditional deep copy operation will become a performance bottleneck in the frequent state simulation of MCTS. The code implements a customized shallow copy strategy, custom_shallow_copy, which only performs deep copies on the data that will actually change:
   - This selective replication strategy ensures necessary data isolation while avoiding unnecessary memory allocation and replication operations.
5. Optimized continuous counting algorithm
   - In the Sequence game, calculating the number of consecutive chess pieces is a frequently performed operation. The code implements the _count_consecutive_fast method to count consecutive chess pieces through bidirectional scanning:
   - This algorithm reduces unnecessary calculations by interrupting the scan in advance, and uses min(count, 5) to ensure the validity of the return value.

[Back to top](#table-of-contents)

### Solved Challenges
**Challenge 1: Cache capacity control and memory management**
 - The LRU cache system implemented in the code demonstrates precise control over memory usage. Through the implementation of the LRUCache class, the system can find a balance point between performance improvement and memory consumption
 - Different types of caches have been assigned different capacity limits. The action evaluation cache is set at 10,000 items, while the card evaluation and status evaluation caches are set at 5,000 items. This differentiated capacity allocation reflects an in-depth understanding of various computing frequencies. Action evaluation is called more frequently in MCTS search, thus requiring a larger cache space.
**Challenge 2: Computing Resource Allocation under strict time constraints**
 - The code sets a strict time limit through MAX_THINK_TIME = 0.95, which requires the algorithm to complete complex decision calculations within less than one second. The implementation of the AdaptiveTimeManager class demonstrates how to dynamically adjust the computational intensity
 - The system allocates different time budgets according to the game stages. It is 0.4 seconds in the opening stage, 0.6 seconds in the middle stage, 0.8 seconds in the endgame, and up to 0.9 seconds at critical moments. Meanwhile, the time-checking mechanism in the MCTS search loop ensures that the algorithm can stop in time
**Challenge 3: Performance optimization requirements for state replication**
 - The MCTS algorithm requires frequent replication of the game state for simulation, and simple deep copying will bring significant performance overhead. The code implements an accurate state replication strategy through the custom_shallow_copy method:
 - This solution identifies that the chessboard state (chips) is the only data structure that will be modified in the simulation. Therefore, only this part is deeply copied, while other data structures are shallowly copied. This selective replication strategy significantly reduces the overhead of memory allocation and replication operations.
**Challenge 4: Balancing the quality and randomness of the simulation strategy**
 - A purely random simulation strategy will generate a large number of low-quality simulation sequences, affecting the convergence effect of MCTS. The code implements the heuristic guided simulation strategy through the _heuristic_guided_simulate method
 - This hybrid strategy improves the simulation quality while maintaining the necessary randomness, and ADAPTS to different game stages by dynamically adjusting the proportion of heuristic usage.
**Challenge 5: The stability issue of search convergence**
 - The convergence speed and stability of MCTS search directly affect the quality of decision-making. The code implements an early termination mechanism to solve the problem of search efficiency
 - When the access times of a certain action account for more than 70% of the total access times (dominant_threshold = 0.7), the algorithm considers that an obvious optimal choice has been found and the search can be terminated in advance. This mechanism not only improves the computational efficiency but also ensures the stability of decision-making.
**Challenge 6: Effective Pruning of Action Space**
 - Facing a large number of possible action choices, how to quickly identify and filter out the most valuable candidate actions is a key challenge. The code implements systematic action screening through the _heuristic_filter method:
 - Meanwhile, the code also uses batch evaluation and sorting. This filtering process first eliminates obviously inferior choices such as corner positions, then ranks the remaining actions through quality assessment, and finally only retains the candidate actions ranked from the top 12 to 20 to participate in the MCTS search. This hierarchical filtering strategy significantly reduces the size of the search space while maintaining the search quality.

 - [Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  
**Powerful search capability:** MCTS can effectively explore complex state Spaces and avoid local optima
**Anytime algorithm:** It can stop at any time and provide the current best decision
**Self-improvement:** Automatically enhance the quality of decision-making through more simulations
**Dealing with uncertainty:** Random simulation is naturally suitable for handling random elements in games
**Scalability:** It is easy to add new heuristic and optimization techniques

#### *Disadvantages*
**Computationally intensive:** A large amount of simulation is required to obtain high-quality decisions
**Memory overhead:** The tree structure and cache system consume a considerable amount of memory
**Parameter sensitivity:** Parameters such as UCB weight and simulation depth need to be carefully optimized
**Heuristic dependency:** Performance largely depends on the quality of the heuristic function
**Time instability:** Suboptimal decisions may be made under time pressure

[Back to top](#table-of-contents)

### Future improvements  
**1. More intelligent simulation strategies:**
 - Realize the policy network guidance simulation based on neural networks
 - Improve the prediction of opponent behavior in simulation using opponent modeling
**2. Reinforcement learning Integration:**
 - Train the value network by combining historical game data
 - Realize the adaptive parameter adjustment mechanism
**3. Parallelization optimization:**
 - Implement multi-threaded MCTS search
 - Use root parallelization to improve search efficiency
**4. More refined time management:**
 - Dynamically adjust the iterative allocation based on the complexity of the game
 - Realize progressive deepening search
**5. Advanced Heuristics:**
 - Add pattern recognition capabilities
 - Achieve more complex threat detection and sequence analysis
**6. Opponent adaptability:**
 - Learn the game style of your opponents
 - Dynamically adjust strategies to counter specific types of opponents
**7. Incorporate layout strategies for corner positions (not considered yet)**

[Back to top](#table-of-contents)
