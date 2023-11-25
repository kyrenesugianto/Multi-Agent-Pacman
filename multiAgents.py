# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Extract food list and initialize variables
        newFoodList = newFood.asList()
        minFoodDistance = float("inf")

        # Calculate minimum food distance
        for food in newFoodList:
            minFoodDistance = min(minFoodDistance, manhattanDistance(newPos, food))

        # Avoid ghosts if they are too close
        for ghost in successorGameState.getGhostPositions():
            if manhattanDistance(newPos, ghost) < 2:
                return -float('inf')

        # Add a small bonus for eating food and a penalty for getting closer to ghosts
        evaluationScore = successorGameState.getScore() + 1.0 / minFoodDistance

        return evaluationScore
        
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0, 0)[0]

    def custom_Minimax(self, gameState, agent_index, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return gameState.getScore()
        if agent_index == 0:
            return self.max_value(gameState, agent_index, depth)[1]
        else:
            return self.min_value(gameState, agent_index, depth)[1]

    def max_value(self, gameState, agent_index, depth):
        best_action = ("max", -float("inf"))
        for action in gameState.getLegalActions(agent_index):
            successor_action = (action, self.custom_Minimax(gameState.generateSuccessor(agent_index, action),
                                                          (depth + 1) % gameState.getNumAgents(), depth + 1))
            best_action = max(best_action, successor_action, key=lambda x: x[1])
        return best_action

    def min_value(self, gameState, agent_index, depth):
        best_action = ("min", float("inf"))
        for action in gameState.getLegalActions(agent_index):
            successor_action = (action, self.custom_Minimax(gameState.generateSuccessor(agent_index, action),
                                                          (depth + 1) % gameState.getNumAgents(), depth + 1))
            best_action = min(best_action, successor_action, key=lambda x: x[1])
        return best_action

        "util.raiseNotDefined()"

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def alpha_beta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth, alpha, beta)[1]
        else:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)[1]

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        best_action = ("max", -float("inf"))
        legal_actions = gameState.getLegalActions(agentIndex)

        for action in legal_actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorAction = (action, self.alpha_beta(successorGameState,
                                                      (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            best_action = max(best_action, successorAction, key=lambda x: x[1])

            if best_action[1] > beta:
                return best_action
            else:
                alpha = max(alpha, best_action[1])

        return best_action

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        best_action = ("min", float("inf"))
        legal_actions = gameState.getLegalActions(agentIndex)

        for action in legal_actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorAction = (action, self.alpha_beta(successorGameState,
                                                      (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            best_action = min(best_action, successorAction, key=lambda x: x[1])

            if best_action[1] < alpha:
                return best_action
            else:
                beta = min(beta, best_action[1])

        return best_action

        "util.raiseNotDefined()"

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        max_depth = self.depth * gameState.getNumAgents()
        return self.custom_expectimax(gameState, "expect", max_depth, 0)[0]

    def custom_expectimax(self, state, action, depth, agent_index):
        if depth == 0 or state.isLose() or state.isWin():
            return (action, self.evaluationFunction(state))

        if agent_index == 0:
            return self.custom_max_value(state, action, depth, agent_index)
        else:
            return self.custom_exp_value(state, action, depth, agent_index)

    def custom_max_value(self, state, action, depth, agent_index):
        best_action = ("max", -(float('inf')))
        for legalAction in state.getLegalActions(agent_index):
            nextAgent = (agent_index + 1) % state.getNumAgents()
            succ_action = action if depth != self.depth * state.getNumAgents() else legalAction
            succ_value = self.custom_expectimax(state.generateSuccessor(agent_index, legalAction),
                                               succ_action, depth - 1, nextAgent)
            best_action = max(best_action, succ_value, key=lambda x: x[1])
        return best_action

    def custom_exp_value(self, state, action, depth, agent_index):
        legalActions = state.getLegalActions(agent_index)
        total_score = 0
        p = 1.0 / len(legalActions)
    
        for legalAction in legalActions:
            next_agent = (agent_index + 1) % state.getNumAgents()
            succ_value = self.custom_expectimax(state.generateSuccessor(agent_index, legalAction),
                                           action, depth - 1, next_agent)[1]
            total_score += succ_value * p
        return (action, total_score)
    
        "util.raiseNotDefined()"

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    This function is a combination of the following factors:
    1. The distance to the closest food
    2. The distance to the closest ghost
    3. The number of remaining food
    4. The number of remaining capsules

    The distance to the closest food is the most important factor, so it has the highest multiplier.
    The distance to the closest ghost is the second most important factor, so it has the second highest multiplier.
    The number of remaining food is the third most important factor, so it has the third highest multiplier.
    The number of remaining capsules is the least important factor, so it has the lowest multiplier.

    The score is calculated as follows:
    score = (1 / distance to the closest food) * foodDistanceMultiplier
            + (1 / distance to the closest ghost) * ghostDistanceMultiplier
            + (remaining food) * foodLeftMultiplier
            + (remaining capsules) * capsulesLeftMultiplier

    The score is then added to the current score of the game state.


    """
    "*** YOUR CODE HERE ***"
    "util.raiseNotDefined()"

    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()

    # Calculate the minimum distance to food
    minFoodDistance = float('inf')
    for food in foodList:
        minFoodDistance = min(minFoodDistance, manhattanDistance(pacmanPosition, food))

    # Calculate the distance to the closest ghost
    ghostDistance = 0
    for ghost in currentGameState.getGhostPositions():
        ghostDistance = manhattanDistance(pacmanPosition, ghost)
        if ghostDistance < 2:
            return -float('inf')

    # Get information about remaining food and capsules
    remainingFood = currentGameState.getNumFood()
    remainingCapsules = len(currentGameState.getCapsules())

    # Set multipliers for different factors
    foodLeftX = 100000
    capsulesLeftX = 10000
    foodDistanceX = 1000

    additionalScore = 0

    # Adjust score based on game outcome (win/lose)
    if currentGameState.isLose():
        additionalScore -= 10000
    elif currentGameState.isWin():
        additionalScore += 10000

    # Calculate the final evaluation score
    evaluationScore = 1.0 / (remainingFood + 1) * foodLeftX + ghostDistance + \
                      1.0 / (minFoodDistance + 1) * foodDistanceX + \
                      1.0 / (remainingCapsules + 1) * capsulesLeftX + additionalScore

    return evaluationScore

# Abbreviation
better = betterEvaluationFunction
