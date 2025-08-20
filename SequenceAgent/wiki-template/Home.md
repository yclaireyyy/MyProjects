# UoM COMP90054 Contest Project

This Wiki can be used as external documentation to the project.
1. [Home and Introduction]()
2. [Problem Analysis](Problem-Analysis)

    2.1 [Approach One](AI-Method-1)

    2.2 [Approach Two](AI-Method-2)

    2.3 [Approach Three](AI-Method-3)

    2.4 [Approach Four](AI-Method-4)

3. [Evolution and Experiments](Experiments)
4. [Conclusions and Reflections](Conclusions-and-Reflections)

## Introduction

This project aims to develop an intelligent AI Agent for Sequence games. It uses multiple strategy search and evaluation algorithms such as Value-Based Heuristic, Monte Carlo Tree Search (MCTS), and Minimax, and has good game reasoning ability and dynamic adaptability.

In terms of basic strategy design, we built a heuristic function based on chessboard situation evaluation to measure the contribution of different positions to achieving Sequence goals and guide action selection accordingly. In order to deal with deeper game decisions, we implemented Minimax and MCTS search frameworks to evaluate the long-term value of actions by simulating self-play and improve robustness in complex situations.

Sequence is a battle game combining poker and chessboard strategies. Players place corresponding pieces on the board by playing their cards to form two consecutive five-piece lines (Sequence) as soon as possible. The game includes ordinary card plays and special Jack card operations (placing at everywhere or removing the opponent's piece). It emphasize card management, space control and opponent prediction. It is a classic game of luck and strategy.
