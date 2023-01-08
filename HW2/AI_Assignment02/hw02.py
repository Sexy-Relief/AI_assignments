from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


## Example Agent
class ReflexAgent(Agent):

    def Action(self, gameState):
        move_candidate = gameState.getLegalActions()

        scores = [self.reflex_agent_evaluationFunc(gameState, action) for action in move_candidate]
        bestScore = max(scores)
        Index = [index for index in range(len(scores)) if scores[index] == bestScore]
        get_index = random.choice(Index)

        return move_candidate[get_index]

    def reflex_agent_evaluationFunc(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvalFunc(currentGameState):
    return currentGameState.getScore()


class AdversialSearchAgent(Agent):

    def __init__(self, getFunc='scoreEvalFunc', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(getFunc, globals())

        self.depth = int(depth)


class MinimaxAgent(AdversialSearchAgent):
    """
    [문제 01] MiniMaxAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
    def Action(self, gameState):
        ####################### Write Your Code Here ################################
        return self.minimax_agent_evaluationFunc(gameState,0,1,0)

    def minimax_agent_evaluationFunc(self, currentGameState, agent, maxnode, depth):
        if maxnode:
            retval = -1_000_000_000
            tmp = retval
        else:
            retval = 1_000_000_000
        if currentGameState.isWin() or currentGameState.isLose() or depth == self.depth:
            return self.evaluationFunction(currentGameState)
        for i, newaction in enumerate(currentGameState.getLegalActions(agent)):
            successorGameState = currentGameState.generateSuccessor(agent, newaction)

            if maxnode:
                scores = self.minimax_agent_evaluationFunc(successorGameState, 1, 1 - maxnode, depth)
                if depth==0:
                    if tmp < scores:
                        tmp = scores
                        retval = newaction
                else:
                    retval=max(retval,scores)

            elif agent < currentGameState.getNumAgents()-1:
                scores = self.minimax_agent_evaluationFunc(successorGameState, agent + 1, maxnode, depth)
                retval=min(retval,scores)
            else:
                scores = self.minimax_agent_evaluationFunc(successorGameState, 0, 1 - maxnode, depth + 1)
                retval=min(retval,scores)
        return retval
        ############################################################################

class AlphaBetaAgent(AdversialSearchAgent):
    """
    [문제 02] AlphaBetaAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """

    def Action(self, gameState):
        ####################### Write Your Code Here ################################
        return self.alphabeta_agent_evaluationFunc(gameState, 0, 1, 0,-1_000_000_000,1_000_000_000)

    def alphabeta_agent_evaluationFunc(self, currentGameState, agent, maxnode, depth, alpha, beta):
        if maxnode:
            retval = -1_000_000_000
            tmp = retval
        else:
            retval = 1_000_000_000
        if currentGameState.isWin() or currentGameState.isLose() or depth == self.depth:
            return self.evaluationFunction(currentGameState)
        for i, newaction in enumerate(currentGameState.getLegalActions(agent)):
            successorGameState = currentGameState.generateSuccessor(agent, newaction)

            if maxnode:
                scores = self.alphabeta_agent_evaluationFunc(successorGameState, 1, 1 - maxnode, depth, alpha, beta)
                if depth == 0:
                    if tmp < scores:
                        tmp = scores
                        retval = newaction
                        alpha = scores
                else:
                    retval = max(retval, scores)
                    alpha = retval

                if scores >= beta:
                    break

            elif agent < currentGameState.getNumAgents() - 1:
                scores = self.alphabeta_agent_evaluationFunc(successorGameState, agent + 1, maxnode, depth, alpha, beta)
                retval = min(retval, scores)
                beta = retval
                if scores <= alpha:
                    break
            else:
                scores = self.alphabeta_agent_evaluationFunc(successorGameState, 0, 1 - maxnode, depth + 1, alpha, beta)
                retval = min(retval, scores)
                beta = retval
                if scores <= alpha:
                    break
        return retval


class ExpectimaxAgent(AdversialSearchAgent):
    """
    [문제 03] ExpectimaxAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
        ####################### Write Your Code Here ################################
    def Action(self, gameState):
        ####################### Write Your Code Here ################################
        return self.expectimax_agent_evaluationFunc(gameState,0,1,0)

    def expectimax_agent_evaluationFunc(self, currentGameState, agent, maxnode, depth):
        if maxnode:
            retval = 0
            tmp = -1_000_000_000
        else:
            retval = 0
        if currentGameState.isWin() or currentGameState.isLose() or depth == self.depth:
            return self.evaluationFunction(currentGameState)
        p = 1/len(currentGameState.getLegalActions(agent))
        for i, newaction in enumerate(currentGameState.getLegalActions(agent)):
            successorGameState = currentGameState.generateSuccessor(agent, newaction)

            if maxnode:
                scores = self.expectimax_agent_evaluationFunc(successorGameState, 1, 1 - maxnode, depth)
                if depth==0:
                    if tmp < scores:
                        tmp = scores
                        retval = newaction
                else:
                    retval+=p*scores

            elif agent < currentGameState.getNumAgents()-1:
                scores = self.expectimax_agent_evaluationFunc(successorGameState, agent + 1, maxnode, depth)
                retval+=p*scores
            else:
                scores = self.expectimax_agent_evaluationFunc(successorGameState, 0, 1 - maxnode, depth + 1)
                retval+=p*scores
        return retval

        ############################################################################
