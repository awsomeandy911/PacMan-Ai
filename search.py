# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

   
    fringe = util.Stack() # initialize stack to frontier (stack preferred for DFS)
    startPosition = problem.getStartState() 
    visited = [] # initialize array for nodes visited
    path = [] # initialize path (action)
    startNode = (startPosition, path) # assigning startPosition and path as the starting node
    fringe.push(startNode) # adds the first node (starting node) to the fringe
    
    # enter loop
    while not fringe.isEmpty(): 
        node = fringe.pop() # node contains the position and path
        visited.append(node[0]) # the node is visited and the position is appended to the visited array
        if problem.isGoalState(node[0]): # sees if the current node position is the goal (we do this first in case )
            return node[1] # if the node position is the goal, return the path of node
        for child in problem.getSuccessors(node[0]): # enter loop if we have not reached goal
            if child[0] in visited: # we look at the children nodes and if the location has already been visited, we skip
                continue
            fringe.push((child[0], node[1] + [child[1]])) # otherwise, we add the location of the child and the path of parent and child to the stack
   

    #util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    fringe = util.Queue() # initializes queue to frontier (queue preferred for BFS)
    startPosition = problem.getStartState()
    visited = [] # initialize array for nodes visited
    path = [] # initialize path (action)
    startNode = (startPosition, path)  # assigning startPosition and path as the starting node
    fringe.push(startNode) # adds the first node (starting node) to the fringe
    visited.append(startPosition) # we add the starting position to the visited array

    # enter loop
    while not fringe.isEmpty(): 
        node = fringe.pop() # node contains the position and path
        if problem.isGoalState(node[0]): # sees if the current node position is the goal
            return node[1] # if the node position is the goal, return the path of node
        for child in problem.getSuccessors(node[0]): # enter loop if we have not reached goal
            if child[0] in visited: # we look at the children nodes and if the location has already been visited, we skip
                continue
            visited.append(child[0]) # unlike DFS, we will add the child node's location regardless
            fringe.push((child[0], node[1] + [child[1]])) # otherwise, we add the location of the child and the path of the parent and child to the queue


    #util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    fringe = util.PriorityQueue() # initialize priorityqueue to frontier
    startPosition = problem.getStartState() 
    visited = [] # initialize array for nodes visited 
    path = [] # initialize path (action)
    cost = 0 # initialize cost
    startNode = (startPosition, path, cost) # assigning startPosition, path, and cost to the starting node
    fringe.push(startNode, 0) 

    while not fringe.isEmpty():
        node = fringe.pop() # node contains the position, path, and cost
        if problem.isGoalState(node[0]): # sees if the current node position is the goal
            return node[1] # if the node position is the goal, return the path of node
        if node[0] not in visited: # if the position of the node is not in array
            visited.append(node[0]) # we add the position of the node to the array
            for child in problem.getSuccessors(node[0]): # we look at the children nodes
                if child[0] not in visited: # if the position of the child node is not in the visited array,
                    cost = child[2] + node[2] # we add the cost of the child node and the parent node
                    fringe.push((child[0], node[1] + [child[1]], child[2] + node[2]), cost)

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    fringe = util.PriorityQueue() # initialize priorityqueue to frontier
    startPosition = problem.getStartState() 
    visited = [] # initialize array for nodes visited 
    path = [] # initialize path (action)
    cost = 0 # initialize cost
    startNode = (startPosition, path, cost) # assigning startPosition, path, and cost to the starting node
    fringe.push(startNode, 0) 

    while not fringe.isEmpty():
        node = fringe.pop() # node contains the position, path, and cost
        if problem.isGoalState(node[0]): # sees if the current node position is the goal
            return node[1] # if the node position is the goal, return the path of node
        if node[0] not in visited: # if the position of the node is not in array
            visited.append(node[0]) # we add the position of the node to the array
            for child in problem.getSuccessors(node[0]): # we look at the children nodes
                if child[0] not in visited: # if the position of the child node is not in the visited array,
                    cost = child[2] + node[2] # we add the cost of the child node and the parent node
                    fringe.push((child[0], node[1] + [child[1]], child[2] + node[2]), cost + (heuristic(child[0], problem)))
                    # very similar to the uniformcostsearch function, b

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
