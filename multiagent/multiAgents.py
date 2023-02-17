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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        score = successorGameState.getScore()
        Food = newFood.asList()
        if Food:
            for food in Food:
                foodDistance = min([manhattanDistance(newPos, food)])
        else:
            foodDistance = 0

        score -= (foodDistance + (100*len(Food))) # Pacman finds food and score is changed

        for ghost in newGhostStates: # Looks for distance from ghost
            ghostPosition = ghost.getPosition()
            ghostDistance = min([manhattanDistance(newPos, ghostPosition)])
            if ghostDistance <= 1: # if distance of ghost is extremely near, points get deducted
                score -= 1000

        return score
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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

        def maxValue(state, depth):
            legalActions = state.getLegalActions(0)
            if len(legalActions) == 0 or depth == self.depth:
                return self.evaluationFunction(state)
            successor = (minValue(state.generateSuccessor(0, action), 0 + 1, depth + 1) for action in legalActions)
            return max(successor)    

        def minValue(state, agentID, depth):
            legalActions = state.getLegalActions(agentID)
            if len(legalActions) == 0:  
                return self.evaluationFunction(state)
            if agentID == state.getNumAgents() - 1:
                successor = (maxValue(state.generateSuccessor(agentID, action), depth) for action in legalActions)
                return min(successor)
            else:
                successor = (minValue(state.generateSuccessor(agentID, action), agentID + 1, depth) for action in legalActions)
                return min(successor)

        optimal = None
        max_value = float('-inf')
        for action in gameState.getLegalActions(0):
            value = minValue(gameState.generateSuccessor(0, action), 1, 1)
            if value > max_value:
                max_value = value
                optimal = action
        return optimal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float('inf')
        direction = None

        def maxValue(state, depth, a, b):
            legalActions = state.getLegalActions(0)
            if len(legalActions) == 0 or depth == self.depth:
                return self.evaluationFunction(state)
            v = -inf
            if depth == 0:
                direction = legalActions[0]
            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                vNext = minValue(successor, 0 + 1, depth + 1, a, b)
                if vNext > v:
                    v = vNext
                    if depth == 0:
                        direction = action
                if v > b:
                    return v
                a = max(a, v)
            if depth == 0:
                return direction
            return v


        def minValue(state, agentID, depth, a, b):
            legalActions = state.getLegalActions(agentID)
            if len(legalActions) == 0:
                return self.evaluationFunction(state)

            v = inf
            for action in legalActions:
                successor = state.generateSuccessor(agentID, action)
                if agentID == state.getNumAgents() - 1:
                    vNext = maxValue(successor, depth, a, b)
                else:
                    vNext = minValue(successor, agentID + 1, depth, a, b)
                v = min(v, vNext)
                if v < a:
                    return v
                b = min(b, v)
            return v

        direction = maxValue(gameState, 0, -inf, inf)
        return direction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth):
            legalActions = state.getLegalActions(0)
            if len(legalActions) == 0 or depth == self.depth:
                return self.evaluationFunction(state)

            vFinal = float("-inf")
            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                v = expValue(successor, 0 + 1, depth + 1)
                vFinal = max(vFinal, v)
            return vFinal

        def expValue(state, agentID, depth):
            v = 0
            
            legalActions = state.getLegalActions(agentID)
            if len(legalActions) == 0:
                return self.evaluationFunction(state)
            probability = 1.0 / len(legalActions)

            for action in legalActions:
                successor = state.generateSuccessor(agentID, action)
                if agentID == state.getNumAgents() - 1:
                    v += maxValue(successor, depth) * probability
                else:
                    v += expValue(successor, agentID + 1, depth) * probability
            return v

        legalActions = gameState.getLegalActions()
        optimal = None
        max_value = float('-inf')
        for action in gameState.getLegalActions(0):
            value = expValue(gameState.generateSuccessor(0, action), 1, 1)
            if value > max_value:
                max_value = value
                optimal = action
        return optimal

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    nearestFoodDist = min(manhattanDistance(position, food) for food in foods) if foods else 0.5
    score = currentGameState.getScore()

    evaluation = 1.0 / nearestFoodDist + score
    return evaluation

# Abbreviation
better = betterEvaluationFunction
