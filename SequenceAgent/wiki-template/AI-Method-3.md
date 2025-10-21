# AI Method 4 - Greedy with Heuristic of Location Value


# Table of Contents
  * [Motivation](#motivation)
  * [Application](#application)
  * [Solved challenges](#solved-challenges)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)
 
### Motivation  

We decided to design a efficient single-step greedy agent that can achieve near-optimal instant decision-making capabilities. We design this through evaluating each position's value. And lay it as the foundation for future expansion of search based agents.

The most important thing here is our heuristic mechanism with sliding windows, which calculate the value of the target point.

We used this sliding window to accelerate the caculation process. The key to winning the Sequence game is to form 2 sequence, however it takes too much calculation and too slow to evaluate each position one by one. So a sliding window of length 5 which moves step by step on the board in four directions: horizontally, vertically, and on two diagonals is applied. Each move only focuses on the current five blocks, calculating how many of our own chips, how many of opponent's chips, and how many chips have been connected in sequence. The advantage of the sliding window is that it doesn't need to count but to remember the previous results, add the new and pop that moves out then update the data instantly. 

Different actions in the game have different effect and need to be calculated separately. Players' actions are divided into placing chips and removing opponents' chips. So different calculation methods are needed, Since the results of these two types of actions are also different. At the same time, these two actions need to be compared on the same level. So we also need to consider the balance between the two. For this reason, we split four values: "place value", "block value", "remove value" and "replace value". And in the last step, they are combined and balanced into a value function that measures the place action and the removal cation.

Considering that the chess pieces in the center naturally have greater potential, we added a Gaussian distribution to all positions. This will make the agent more inclined to place the chess pieces in the center of the board.

In this Greedy Agent, we also use an action splitting mechanism to calculate the value of the hand cards and the public cards separately, and finally select the combination with the greatest value, thereby further shortening the calculation time.

[Back to top](#table-of-contents)

### Application  

#### Position Value

A window can form a sequence only if it meets the following conditions: there are only its own chips and spaces in the 5 grids, and it contains at most 1 sequence chip. For each position, it is in at most 20 such windows (5 windows in each of the horizontal and two vertical diagonals), and considering that there is only 1 sequence can be formed at most on common in each direction. We take the most likely window as the window to evaluate the value of this position.

For each window, we count the following information:
    1. Number of normal chips of ours
    2. Number of sequence chips of ours
    3. Number of normal chips of opponent's
    4. Number of sequence chips of opponent's
    5. Number of Empty
    6. Number of Jokers

Then, based on the state in the window, calculate the benefits of each type of action:

|Value type|Calculation conditions|Calculation formula|Typical scenario|
|------|------|-------|------|
|Place value|No opponent chips and own sequence ≤1|Total number of own chips|Build own sequence in this window|
|Block value|No own chips and opponent sequence ≤1|Total number of opponent chips|Potential to block the opponent's sequence that is about to be completed|
|Remove value|No own chips and opponent sequence ≤1|Total number of opponent chips|Destroy the opponent's key pieces|
|Replace value|No opponent sequence and opponent chips ≤1|Total number of own chips|Immediately seize the pieces after pulling them out to form a new sequence|

Example: There are 3 friendly chess pieces, 1 empty space, and 1 opponent chess piece in the window (no sequence):
    - Place value = 0 (no chance to form a sequence in this window)
    - Block value = 0 (our chip exist, and the condition is not met)
    - Remove value = 1 (number of opponent chips)
    - Replace value = 3 (number of our chips)


We take the exponential sum of the values in four directions as its final value. We also consider the terminal state judgment: if there are 4  pieces in a certain window and there is a formed sequence, we directly mark the value as infinity (inf) and execute it first.

Center Prior Value is calculated as following:

$$ C_p(x, y) = SCALE*exp\left[-SMOOTH * \left[(x - 4.5)^2 + (y - 4.5)^2\right]\right] $$

Heart area special judgment: If the player occupies 3 of the four center squares, the move value is set to inf; if the opponent occupies 3 squares, the interception value is set to inf.

#### Value Balancing

Considering that the value of Empty position contains of place value and block value, and the value of one-eyed-jack contains of remove value and replace value. These four values have different distributions due to different calculation methods. In order to balance, We introduced PLACE_BIAS, REMOVE_BIAS and PLACE_REMOVE_SCALE to balance these two actions.

$$V_{place}' = (1 + PLACE\_BIAS) * v_{place} + (1 - PLACE\_BIAS) * v_{block}$$

$$V_{remove}' = (1 + REMOVE\_BIAS) * v_{place} + (1 - REMOVE\_BIAS) * v_{block}$$

$$V_{place} = (1 + PLACE\_REMOVE\_SCALE)*V_{place}'$$

$$V_{remove} = (1 - PLACE\_REMOVE\_SCALE)*V_{block}'$$

We used hill climbing for hyperparameter tuning to find the optimal parameter. We changed the value and compare each agent with and without change, evaluate its win rate, then select the best one.

#### Action Analysis

We split use card and pick card as two independently actions. Since the probability that the pick card is exactly the used card is very low.

For use card, We first break down the hand into normal card, one_eyed_jack and two_eyed_jack. We analyze the legal positions of normal card and one_eyed_jack at first. Then, for two_eyed_jack, we only consider those positions beyond normal card. After evaluation we sort them by value and select the use action with the highest value.

And for pick card, the draft card is also divided into 3 parts, but the difference is that if there is two_eyed_jack, take it directly. Else take one_eyed_jack if there exists one. Otherwise we analyze the value of the remaining cards and select the draft card with the highest value as the pick action.

For dead card exchange, we always replace the first dead card with the most valuable draft card by default. When the game is deadlocked at the end and all public cards on the field are dead, the exchange action is randomly selected.

Finally, the two parts are combined to synthesize the action that meets the specification and return the output.

[Back to top](#table-of-contents)

### Solved Challenges

This agent solves the following problems:
1. The algorithm uses a sliding window algorithm to quickly scan the four directions of the chessboard, reducing the complexity of position evaluation. The action split design independently handles the card-playing and card-selecting stages, avoiding the exponential computational burden of combinatorial decision-making.
2. Unifies values of two different actions: measure offensive potential through placement value, evaluate defensive benefits through blocking value, remove value for key chips, and calculate strategic advantages for position seizure through replacement value. The center area of ​​the chessboard obtains natural priority through the Gaussian weight formula, and the sequence window to be formed is given an infinite value to ensure instant victory. In view of the trade-off between offensive and defensive actions, the algorithm introduces a dynamic hyperparameter balance formula, combined with the hill climbing method for automatic tuning to maximize the winning rate.
3. Solved the evaluation problem of intermediate states by using an exponential weighting function.
4. In the face of the decision complexity caused by high branching factors, it provides prior knowledge for search-based agents, thereby reducing search nodes.

[Back to top](#table-of-contents)


### Trade-offs  
#### *Advantages*  

The algorithm achieves real-time decision-making at a very low computational cost. The sliding window mechanism reduces the complexity of position evaluation to a constant level, and can respond in milliseconds even on a large-scale chessboard. The four-dimensional value model (placement/blocking/removal/replacement) accurately quantifies the benefits of attack and defense, and the central Gaussian weight and the final infinite value design strengthen the control of key areas. The action layering process customizes strategies for different card types, combined with the dynamic balance of hyperparameters, to achieve near-optimal single-step decisions without the need for global search, providing an efficient baseline for subsequent adversarial algorithms.

#### *Disadvantages*

The fundamental flaw is the lack of long-term planning capability: a single-step greedy strategy cannot foresee the opponent's subsequent actions and is prone to falling into the local optimal trap. Hyperparameters rely on manual tuning and need to be retrained when the environment changes. The rigid rules lead to insufficient adaptability - for example, when faced with unconventional blocking strategies, the four-dimensional model may misjudge asymmetric risks. The unconventional position evaluation of the binocular jack avoids waste of resources, but may ignore opportunities for mixed tactics. Finally, the fixed trigger mechanism of dead card replacement is not flexible enough in complex endgames.

[Back to top](#table-of-contents)

### Future improvements  

Fusion of MCTS and Minimax: Grafting lightweight adversarial search on top of the greedy decision layer. When there is enough time, start 3-5 layers of Minimax to predict the opponent's counter-strategy; call MCTS to simulate key branches in complex scenarios to balance real-time performance and long-term planning. Dynamic pruning mechanism prioritizes exploration of high-value action branches to avoid invalid state explosion.

Phased regional value: Replace fixed Gaussian weights and design time-space sensitive functions - strengthen center control in the early stage (number of steps <15), use uniform weights in the middle stage (number of steps 15-30), and activate edge sequence scanning in the endgame (number of steps >30). 

Local update mechanism: Only update the value function near the drop point each time with tracking the 20 windows associated with each position in real time.

[Back to top](#table-of-contents)
