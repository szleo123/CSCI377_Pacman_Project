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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        
        score = successorGameState.getScore()
        # parameters to adjust
        foodIndex = 5
        ghostDistIndex = 6

        if len(newFood.asList()) > 0:
            minimalDistFood = min(manhattanDistance(newPos, food) for food in newFood.asList())
            foodScore = foodIndex/minimalDistFood
            score += foodScore

        minimalDistGhost = min([manhattanDistance(newPos, g.getPosition())  for g in newGhostStates])
        if sum(newScaredTimes) == 0 and minimalDistGhost < ghostDistIndex:
            ghostScore = - (minimalDistGhost - ghostDistIndex) ** 2
            score += ghostScore

        if sum(newScaredTimes) > 0:
            score += 1000 / minimalDistGhost
        return score

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
        """
        actions = gameState.getLegalActions(0)
        score = -float("inf")
        act = None
        for action in actions:
            successor = gameState.generateSuccessor(0,action)
            s = self.getAction_helper(successor,self.depth, 1)
            if s > score:
                score = s
                act = action
        return act

    def getAction_helper(self, gameState, depth, agentIndex):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maximize(gameState, depth, agentIndex)
        if agentIndex != 0:
            return self.minimize(gameState, depth, agentIndex)

    def maximize(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        agentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if agentIndex == 0:
            depth -= 1
        return max([self.getAction_helper(s, depth, agentIndex) for s in successors] + [-float("inf")])


    def minimize(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        agentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if agentIndex == 0:
            depth -= 1
        return min([self.getAction_helper(s, depth, agentIndex) for s in successors] + [float("inf")])
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
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
        """
        actions = gameState.getLegalActions(0)
        score = -float("inf")
        act = None
        alpha = -float("inf")
        beta = float("inf")
        for action in actions:
            successor = gameState.generateSuccessor(0,action)
            s, a, b = self.getAction_helper(successor,self.depth, 1, alpha, beta)
            if s > score:
                score = s
                act = action
            alpha = max(s, alpha)
        return act

    def getAction_helper(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return (self.evaluationFunction(gameState), alpha, beta)
        if agentIndex == 0:
            return self.maximize(gameState, depth, agentIndex, alpha, beta)
        if agentIndex != 0:
            return self.minimize(gameState, depth, agentIndex, alpha, beta)

    def maximize(self, gameState, depth, agentIndex, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        bestval = -float("inf")
        newIndex = (agentIndex + 1) % gameState.getNumAgents()
        if newIndex == 0:
            depth -= 1
        for act in actions:
            if alpha > beta:
                return (bestval, alpha, beta) 
            successor = gameState.generateSuccessor(agentIndex, act)
            bestval = max(bestval, self.getAction_helper(successor, depth, newIndex, alpha, beta)[0])
            alpha = max(alpha, self.getAction_helper(successor, depth, newIndex, alpha, beta)[0])
        return (bestval, alpha, beta)

    def minimize(self, gameState, depth, agentIndex, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        bestval = float("inf")
        newIndex = (agentIndex + 1) % gameState.getNumAgents()
        if newIndex == 0:
            depth -= 1
        for act in actions:
            if alpha > beta:
                return (bestval, alpha, beta) 
            successor = gameState.generateSuccessor(agentIndex, act)
            bestval = min(bestval, self.getAction_helper(successor, depth, newIndex, alpha, beta)[0])
            beta = min(beta, self.getAction_helper(successor, depth, newIndex, alpha, beta)[0])
        return (bestval, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
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
        """
        actions = gameState.getLegalActions(0)
        score = -float("inf")
        act = None
        for action in actions:
            successor = gameState.generateSuccessor(0,action)
            s = self.getAction_helper(successor,self.depth, 1)
            if s > score:
                score = s
                act = action
        return act

    def getAction_helper(self, gameState, depth, agentIndex):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maximize(gameState, depth, agentIndex)
        if agentIndex != 0:
            return self.minimize(gameState, depth, agentIndex)

    def maximize(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        agentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if agentIndex == 0:
            depth -= 1
        return max([self.getAction_helper(s, depth, agentIndex) for s in successors] + [-float("inf")])


    def minimize(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        agentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if agentIndex == 0:
            depth -= 1
        return sum([self.getAction_helper(s, depth, agentIndex) for s in successors] )/float(len(successors))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
   
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
    score = currentGameState.getScore()
    # parameters to adjust
    foodIndex = 5
    ghostDistIndex = 6

    if len(newFood.asList()) > 0:
        minimalDistFood = min(manhattanDistance(newPos, food) for food in newFood.asList())
        foodScore = foodIndex/minimalDistFood
        score += foodScore

    minimalDistGhost = min([manhattanDistance(newPos, g.getPosition())  for g in newGhostStates])
    if sum(newScaredTimes) == 0 and minimalDistGhost < ghostDistIndex:
        ghostScore = - (minimalDistGhost - ghostDistIndex) ** 2
        score += ghostScore

    if sum(newScaredTimes) > 0:
        score += 1000 / minimalDistGhost
    return score

# Abbreviation
better = betterEvaluationFunction

