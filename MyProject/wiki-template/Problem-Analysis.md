# Analysis of the problem

Sequence is a **partially observable**, **non-deterministic** strategic game environment with **discrete** action space and turn-based execution.

## Partially Observable

Sequence game has different types of information.

### Fully Observable:

1. **Board Chips**: Whether a piece is placed at each position and its ownership (own, opponent, whether it is a Sequence).

2. **Draft Cards**: 5 draft cards is visible to all players.

3. **My Hand Cards**: The 6 cards currently held by the player.

### Estimation through calculations:

1. **Opponent Hand**: It can be calculated through the opponent's picking and placing actions.
2. **Remaining Deck**: It can be calculated thorugh known hand cards, public cards, and discarded cards from the deck.

### Unobservable information:

1. **Order of Deck**: Since the deck is random, it is completely unknown which cards you or your opponent will draw in the future.

## High Branching Factor

The action space of Sequence may be very large in each round, mainly due to the following factors:

1. **Number of cards**: Each player holds 6 cards, each card may correspond to 1-2 positions on the board, and each card can become a legal action. If it is a special card Jacks, there may be dozens of positions;

2. **Common card selection (draft picking)**: After playing a hand card, you need to choose one from the 5 public cards to add to the hand card, forming a "playing + drawing" combination decision;

3. **Many empty spaces on the board**: Before the middle of the game, most areas of the board are empty, and the number of positions available for placement is huge;

4. **Opponent game dimension**: High branching factors exist not only in our decision-making, but also in the opponent's round, and the opponent also faces similar combination choices.

## Equilibrium of Benefit

Every choice players make in the Sequence game involves balancing several objectives:

1. **Building a connection (Sequence)**: In order to win the game, the player must either form two sequences or capture the Heart of the Board.
2. **Blocking the opponent (Blocking)**: Players must also be aware of their opponent's layout and block possible connections.
3. **Avoid being interfered with (Anti-Interference)**: During the game, it is also necessary to consider avoiding key positions be taken to maintain safety.
4. **Action Value Evaluation**: Players need to consider values of each location, and the opponent's possible reaction, and choose the action with the greatest benefit.
