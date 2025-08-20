# AI Method 1 - Iterative Deepening Minimax

# Table of Contents
  * [Motivation](#motivation)
  * [Application](#application)
  * [Solved challenges](#solved-challenges)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)
 
### Motivation  

The design of the ID-Minimax algorithm stems from the fundamental flaws of the greedy strategy in the confrontational scenario of the Sequence game. In the confrontation, the single-step optimal decision mode of the greedy algorithm is not the global optimal solution.

The greedy algorithm ignores the opponent's counter-action, and the opponent can construct a strategic trap. At the same time, it does not consider the opponent's hand situation, and will block the placement at useless points. The two-player zero-sum framework of ID-Minimax uses recursive simulation to force the prediction of the opponent's behavior, turning passive response into active control.

The traditional Minimax has a high search timeout rate due to the high branching factor, especially when encountering cards such as Jacks that have too many placement points. Fixed search depth is difficult to deal with.

The four-dimensional value evaluation and center weight model of the greedy algorithm can be directly integrated to improve search efficiency.

[Back to top](#table-of-contents)

### Application  

#### State Space Model
- States:
    1. 10×10 board state.
    2. The current cards in my hand.
    3. The current hand card of opponent.
    4. Draft cards in the public deck.
    5. The unused cards in the remaining deck.
    6. Our ID.
    7. Which player's turn.
    8. Allow trade or not.
- Initial State:
    1. 10×10 board state.
    2. The cards in my hand.
    3. Calculated hand card of opponent.
    4. The 5 visible cards in the public deck.
    5. The unused cards in the remaining deck.
    6. Our ID.
- Goal States:
    1. Form two complete sequences.
    2. Occupy the center 4 squares of the board.
- Actions:
    1. Move Class:
        1. Normal Card: Place chip at the corresponding position on the card;
        2. Two-eyed Jack: Place a piece in any empty position.
    2. Removal Class:
        1. One-eyed Jack: Remove the opponent's non-sequential chess piece.
    3. Trade Class:
        1. Dead card trade: exchange the draft cards with the hand that cannot act.
- Cost:
    1. Step cost: Each action need a turn to act.
    2. Opportunity cost: Each step needs to balance the offensive/defensive value.
- Effect:
    1. Place or remove a certain chip on board with place or remove action, then pick a card from draft.
    2. Discard a dead card and pick a draft card with trade action if available.

[Back to top](#table-of-contents)

#### Search Method

Iterative Deepening-Minimax (ID-Minimax) uses an iterative deepening framework and heuristic search to solve the high branching factor challenge of Sequence games. The algorithm first performs greedy calculations to generate basic guarantee actions as the decision bottom line. Then the system starts deep iterations, gradually advancing from shallow search to deep search, ensuring that the current optimal solution can always be returned when time runs out.

The core algorithm is a heuristic-guided depth-first search. When each node is expanded, the possible actions are sorted based on the four-dimensional value. Then the action branches with the highest evaluation value are explored first. This value-oriented search order improves efficiency and enables the algorithm to quickly focus on high-potential actions. For the evaluation of leaf nodes, we use the maximum potential value of the current hand as the state score.

In the face of the partially observable characteristics of Sequence games, the search process ignores any uncertainty. By dynamically tracking the opponent's action history, the probability distribution model of his hand is reconstructed. At the same time, in order to avoid the computational complexity brought by the randomness of the deck, only the known hand cards and public card information of both parties are used in the deduction, without considering the replenishment of the deck. This allows the algorithm to maintain decision robustness under partial observable constraints.

### Solved Challenges

1. ID-Minimax reduces uncertainty by modeling the opponent (tracking the opponent's hand and the remaining deck). By dynamically updating the opponent's hand information and the remaining deck, the decision blind spot caused by partial observability is reduced.
2. ID-Minimax reduces the branching factor through action pruning and iterative deepening, making the search faster.
3. The heuristic function inherited from the greedy algorithm combines different types of action values ​​well and converts the situation value into hand value.

[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  
This project has the following advantages
1. By calculating the opponent's hand situation and combining the multi-layer search tree to predict the opponent's counter-strategy, the strategic blind spot of the greedy algorithm is broken.
2. The iterative deepening mechanism ensures that any time interruption returns a valid solution. Combined with the heuristic function to sort the actions, the search time is relatively acceptable and meets the needs of real-time competition.
3. The four-dimensional value model and central weight function of the greedy algorithm are integrated to avoid duplication of work.

#### *Disadvantages*
This project has the following shortcomings:
1. The heuristic function still scans full board and does not implement the incremental updates. Every change will triggers a recalculation of the entire board.
2. When facing the Jack card, the action space is still 50+, which exceeds the upper limit of pruning capability.
3. The heuristic value evaluation is used instead of the win rate model, which has great defects in evaluation accuracy.

[Back to top](#table-of-contents)

### Future improvements  

Uses partial update algotithm. Only update the value function near the drop point each time with tracking the 20 windows associated with each position in real time.

The Jack action space is compressed, one-eyed-jacks only searches for the top 5% of opponent pieces, and two-eyed-jacks focuses on unconventional positions with sequence potential higher than the average

Win rate-oriented evaluation function, changing the evaluation function from the maximum hand value to the win rate, and introducing a reinforcement learning mechanism for updating.

[Back to top](#table-of-contents)
