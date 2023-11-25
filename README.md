# Multi-Agent-Pacman

In Project 2: Multi-Agent Search, I designed agents for the classic version of Pacman, which now includes ghosts. The project involved implementing both minimax and expectimax search algorithms and designing an evaluation function.

I started by playing a game of classic Pacman and then ran a provided ReflexAgent. In the first task (Q1), I improved the ReflexAgent to play more effectively by considering both food locations and ghost locations. I tested the agent on different layouts to assess its performance.

Moving on to Q2, I implemented a minimax search agent in the MinimaxAgent class. I created an algorithm that works for any number of ghosts and expands the game tree to an arbitrary depth. I utilized the self.evaluationFunction to score the leaves of the minimax tree.

In Q3, I implemented the AlphaBetaAgent, which uses alpha-beta pruning to more efficiently explore the minimax tree. I extended the alpha-beta pruning logic appropriately to multiple minimizer agents and assessed the agent's performance on speed and correctness compared to the MinimaxAgent.

Q4 focused on implementing the ExpectimaxAgent, a probabilistic search agent that models suboptimal choices of agents. I assumed random choices among legal actions and observed how the ExpectimaxAgent behaved in Pacman, especially in scenarios with random ghosts.

In the final task (Q5), I wrote a better evaluation function for Pacman in the provided betterEvaluationFunction. The evaluation function evaluated states, and with a depth-2 search, Pacman was able to clear the smallClassic layout with one random ghost more than half the time. The function's performance was evaluated based on winning frequency, average score, and computation time.

Throughout the project, I used specific commands to run tests, and the autograder was utilized to assess the correctness and efficiency of the implemented agents and evaluation function.
