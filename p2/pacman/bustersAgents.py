# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pathlib import Path
import sys
import random
from distanceCalculator import Distancer
from game import Actions
import util
import os
import os.path
import numpy as np
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters


class NullGraphics:
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None,
                 observeEnable=True, elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)


'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  # Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:
            move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal:
            move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class RLAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        ########################### INSERTA TU CODIGO AQUI  ######################
        #
        # INSTRUCCIONES:
        #
        # Dependiendo de las caracteristicas que hayamos seleccionado para representar los estados,
        # tendremos un numero diferente de filas en nuestra tabla Q. Por ejemplo, imagina que hemos seleccionado
        # como caracteristicas de estado la direccion en la que se encuentra el fantasma mas cercano con respecto
        # a pacman, y si hay una pared en esa direccion. La primera caracteristica tiene 4 posibles valores: el
        # fantasma esta encima de pacman, por debajo, a la izquierda o a la derecha. La segunda tiene solo dos: hay
        # una pared en esa direccion o no. El numero de combinaciones posibles seria de 8 y por lo tanto tendriamos 8 estados:
        #
        # nearest_ghost_up, no_wall
        # nearest_ghost_down, no_wall
        # nearest_ghost_right, no_wall
        # nearest_ghost_left, no_wall
        # nearest_ghost_up, wall
        # nearest_ghost_down, wall
        # nearest_ghost_right, wall
        # nearest_ghost_left, wall
        #
        # Entonces, en este caso, estableceriamos que self.nRowsQTable = 8. Este es simplemente un ejemplo,
        # y es tarea del alumno seleccionar las caracteristicas que van a tener estos estados. Para ello, se puede utilizar
        # la informacion que se imprime en printInfo. La idea es seleccionar unas caracteristicas que representen
        # perfectamente en cada momento la situacion del juego, de forma que pacman pueda decidir que accion ejecutar
        # a partir de esa informacion. Despues, hay que seleccionar unos valores adecuados para los parametros self.alpha,
        # self.gamma y self.epsilon.
        #
        ##########################################################################

         # Dimensiones del mapa
        width, height = gameState.data.layout.width, gameState.data.layout.height
        num_ghosts = gameState.getNumAgents() - 1  # Número máximo de fantasmas

        # Calculando el número de filas de la tabla Q
        num_positions = width * height # Posiciones posibles
        num_ghost_alive_states = num_ghosts  # De 0 a num_ghosts fantasmas vivos
        num_direction_wall_states = 8  # 4 direcciones x 2 posibilidades de muro (hay muro, no hay muro)
        num_food_distance_categories = 4  # Categorías de distancia a la comida (cerca, medio, lejos, muy lejos)

        self.nRowsQTable = num_positions * num_ghost_alive_states * num_direction_wall_states * num_food_distance_categories
        self.alpha = 0.1  # Tasa de aprendizaje
        self.gamma = 0.9  # Factor de descuento 
        self.epsilon = 0.1  # Probabilidad de exploración 

        self.converged = False
        ##########################################################################
        self.nColumnsQTable = 5

        self.table_file = Path("qtable.txt")
        self.q_table = self.readQtable() or self.initQtable()
        self.previous_q_table = np.copy(self.q_table)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    def printInfo(self, gameState):
        # Dimensiones del mapa
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("\tWidth: ", width, " Height: ", height)
        # Posicion del Pacman
        print("\tPacman position: ", gameState.getPacmanPosition())
        # Acciones legales de pacman en la posicion actual
        print("\tLegal actions: ", gameState.getLegalPacmanActions())
        # Direccion de pacman
        print("\tPacman direction: ", gameState.data.agentStates[0].getDirection())
        # Numero de fantasmas
        print("\tNumber of ghosts: ", gameState.getNumAgents() - 1)
        # Fantasmas que estan vivos (el indice 0 del array que se devuelve
        # corresponde a pacman y siempre es false)
        print("\tLiving ghosts: ", gameState.getLivingGhosts()[1:])
        # Posicion de los fantasmas
        print("\tGhosts positions: ", gameState.getGhostPositions())
        # Direciones de los fantasmas
        print(
            "\tGhosts directions: ", [
                gameState.getGhostDirections().get(i) for i in range(
                    0, gameState.getNumAgents() - 1)])
        # Distancia de manhattan a los fantasmas
        print("\tGhosts distances: ", gameState.data.ghostDistances)
        # Puntos de comida restantes
        print("\tPac dots: ", gameState.getNumFood())
        # Distancia de manhattan a la comida mas cercada
        print("\tDistance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Paredes del mapa
        print("\tMap Walls:  \n", gameState.getWalls())
        # Comida en el mapa
        print("\tMap Food:  \n", gameState.data.food)
        # Estado terminal
        print("\tGana el juego: ", gameState.isWin())
        # Puntuacion
        print("\tScore: ", gameState.getScore())

    def initQtable(self):
        "Initialize qtable"
        return (np.zeros((self.nRowsQTable, self.nColumnsQTable)).tolist())

    def readQtable(self):
        "Read qtable from disc"
        if not self.table_file.is_file():
            return None

        content = self.table_file.read_text()

        if content == '':
            return None

        q_table = []

        for line in content.split('\n'):
            values = [float(x) for x in line.split()]
            q_table.append(values)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        with open(self.table_file, 'w') as f:
            first = True
            for line in self.q_table:
                if first:
                    first = False
                else:
                    f.write("\n")
                for item in line:
                    f.write(str(item) + " ")

    # Devuelve la dirección relativa al fantasma más cercano
    def getNearestGhostDirectionWithWall(self, state):
        pacman_pos = state.getPacmanPosition() 
        ghost_positions = state.getGhostPositions() 
        ghost_distances = state.data.ghostDistances 

        ghost_distances_no_none = [i for i in ghost_distances if i is not None]
        # Encuentra el índice del fantasma más cercano
        nearest_ghost_index = ghost_distances.index(min(ghost_distances_no_none))

        # Encuentra la posición del fantasma más cercano
        nearest_ghost_pos = ghost_positions[nearest_ghost_index]

        # Determina la dirección relativa al fantasma más cercano
        delta_x = nearest_ghost_pos[0] - pacman_pos[0]
        delta_y = nearest_ghost_pos[1] - pacman_pos[1]

        if abs(delta_x) > abs(delta_y):  # El movimiento principal es horizontal
            if delta_x > 0:
                direction = 'East' # El fantasma está a la derecha
            else:
                direction = 'West' # El fantasma está a la izquierda
        else:  # El movimiento principal es vertical
            if delta_y > 0:
                direction = 'North' # El fantasma está arriba
            else:
                direction = 'South' # El fantasma está abajo

        # Verifica si hay un muro en esa dirección
        legal_actions = state.getLegalPacmanActions()
        if direction == 'North' and 'North' in legal_actions:
            wall = False
        elif direction == 'South' and 'South' in legal_actions:
            wall = False
        elif direction == 'East' and 'East' in legal_actions:
            wall = False
        elif direction == 'West' and 'West' in legal_actions:
            wall = False
        else:
            wall = True

        # Convierte la dirección y presencia de muro a un índice único
        actions = ['North','South','West','East','Stop']
        direction_index = actions.index(direction) * 2 + (1 if wall else 0)

        return direction_index

    def computePosition(self, state):
            """
            Compute the row of the qtable for a given state.
            """
            ########################### INSERTA TU CODIGO AQUI  ######################
            #
            ##########################################################################
            
            # Obtén la información necesaria del estado
            pacman_position = state.getPacmanPosition()
            num_living_ghosts = sum(state.getLivingGhosts()[1:])  # Suma True como 1 y False como 0
            
            # Obtiene la dirección del fantasma más cercano con respecto
            nearest_ghost_direction_with_wall = self.getNearestGhostDirectionWithWall(state)
            
            # Categoriza la distancia como un índice de 0 a 3
            distance_food = state.getDistanceNearestFood()
            if distance_food is not None:
                if distance_food < 2: #cerca
                    nearest_food_distance_category  = 0
                elif distance_food < 3: #medio
                    nearest_food_distance_category = 1
                elif distance_food < 4: #lejos
                    nearest_food_distance_category = 2
                else: #muy lejos
                    nearest_food_distance_category = 3
            else:
                nearest_food_distance_category = 3
            
            # Calcula el índice de la posición de Pac-Man
            width = state.data.layout.width
            pacman_index = pacman_position[1] * width + pacman_position[0] 
        

            # Tamaños de dimensiones de características
            num_positions = width * state.data.layout.height
            num_ghost_alive_states = state.getNumAgents() - 1
            num_direction_wall_states = 8
            # Calcula el índice total
            row_index = pacman_index # Posición de PacMan
            row_index += num_living_ghosts * num_positions # Fantasmas vivos
            row_index += nearest_ghost_direction_with_wall * num_positions * num_ghost_alive_states # Dirección del fantasma más cercano
            row_index += nearest_food_distance_category * num_positions * num_ghost_alive_states * num_direction_wall_states # Categoría de distancia a la comida
   
            return row_index
        

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        actions = ['North','South','West','East','Stop']
        action_column = actions.index(action)

        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalActions(0)
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = state.getLegalActions(0)
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        legalActions = state.getLegalActions(0)
        action = None

        if len(legalActions) == 0:
            return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)

    def getReward(self, state, nextState):
        """
          Return a reward value based on the information of state and nextState
        """
        ########################### INSERTA TU CODIGO AQUI  ######################
        ##########################################################################
        reward = 0
        reward = nextState.getScore() - state.getScore() # Recompensa basada en la diferencia de puntuación
        
        pacman_pos = state.getPacmanPosition()
        ghost_positions = state.getGhostPositions()
        min_distance = 100 
        next_min_distance = 100
        
        # Calcula la distancia al fantasma más cercano en el estado actual
        for ghost_pos in ghost_positions:
            distance = self.distancer.getDistance(pacman_pos, ghost_pos)
            if distance < min_distance:
                min_distance = distance
        # Calcula la distancia al fantasma más cercano en el siguiente estado
        next_pacman_pos = nextState.getPacmanPosition()
        next_ghost_positions = nextState.getGhostPositions()
        for ghost_pos in next_ghost_positions: 
            distance = self.distancer.getDistance(next_pacman_pos, ghost_pos)
            if distance < next_min_distance:
                next_min_distance = distance

        distance_change = min_distance - next_min_distance
        # Recompensa por acercarse al fantasma
        if distance_change > 0:
            reward += 2
        # Penalización por alejarse del fantasma
        elif distance_change < 0: 
            reward -= 2
        
        # Calcula la distancia a la comida más cercana en el estado actual y en el siguiente estado
        if state.getDistanceNearestFood() is not None and nextState.getDistanceNearestFood() is not None:
            distance_food_change = state.getDistanceNearestFood() - nextState.getDistanceNearestFood()       
            # Recompensa por acercarse al fantasma
            if distance_food_change > 0:
                reward += 1
            # Penalización por alejarse del fantasma
            elif distance_food_change < 0: 
                reward -= 1
                
        if nextState.isWin():
            reward = nextState.getScore()  # Bonificacion ligada al score si se gana el juego
        
        return reward
    

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        print("Started in state:")
        self.printInfo(state)
        print("Took action: ", action)
        print("Ended in state:")
        self.printInfo(nextState)
        print("Got reward: ", reward)
        print("---------------------------------")
        ########################### INSERTA TU CODIGO AQUI #######################
        ##########################################################################
        # Obtener los índices de la tabla Q para el estado actual y la acción tomada
        currentStateIndex = self.computePosition(state)
        actions = ['North','South','West','East','Stop']
        action_column = actions.index(action)
        # Comprobar si nextState es un estado terminal
        if nextState.isWin():
            # Si es un estado terminal, asignar un valor de 0 a maxQNext
            maxQNext = 0
        else: # Si no es un estado terminal, calcular el valor máximo de Q para el siguiente estado
            nextStateIndex = self.computePosition(nextState)
            maxQNext = max(self.q_table[nextStateIndex]) 
        # Actualizar el valor de Q para el estado actual y la acción tomada
        # Usamos la fórmula Q(s,a) = Q(s,a) + alpha * (reward + gamma * maxQ(s',a') - Q(s,a))
        currentQValue = self.q_table[currentStateIndex][action_column]
        self.q_table[currentStateIndex][action_column] = (
            currentQValue + self.alpha * (reward + self.gamma * maxQNext - currentQValue)
        ) 
        ##########################################################################
        if nextState.isWin():
            # Calcula la diferencia máxima en la tabla Q desde la última actualización
            max_difference = np.max(np.abs(self.q_table - self.previous_q_table))
            print("Max difference: ", max_difference)
            if max_difference < 10:
                self.converged = True
                print("The model has converged.")
                    # Actualizar la tabla de referencia para la próxima comprobación
            self.previous_q_table = np.copy(self.q_table)

            # If a terminal state is reached
            self.writeQtable()
        

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)
