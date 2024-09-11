from Game import Game
import numpy as np
import random

ACTIONS = ['STRAIGHT', 'RIGHT', 'LEFT']

class NeuralNetwork:
    def __init__(self, layer_sizes):
        # Initialize weights for each layer
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1 for i in range(len(layer_sizes) - 1)]

    def forward(self, inputs):
        output = inputs
        for weight in self.weights:
            output = self.sigmoid(np.dot(output, weight))
        return self.softmax(output)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def mutate(self, mutation_rate=0.05):
        # Mutate each weight with a given probability
        for i in range(len(self.weights)):
            mutation_mask = np.random.rand(*self.weights[i].shape) < mutation_rate
            self.weights[i] += mutation_mask * np.random.randn(*self.weights[i].shape) * 0.5

    @staticmethod
    def crossover(parent1, parent2):
        # Uniform crossover where each weight is selected from one parent with 50% chance
        child = NeuralNetwork([w.shape[0] for w in parent1.weights] + [parent1.weights[-1].shape[1]])
        for i in range(len(parent1.weights)):
            mask = np.random.rand(*parent1.weights[i].shape) < 0.5
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
        return child

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.05, layer_sizes=[14, 16, 3], top_n_percent=10, elite_size=5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.layer_sizes = layer_sizes
        self.top_n_percent = top_n_percent  # percentage of top performers to select
        self.elite_size = elite_size  # number of top agents to keep across generations
        self.population = [NeuralNetwork(layer_sizes) for _ in range(self.population_size)]
        self.elite_agents = []  # Store the top agents across all generations

    def evolve(self, scores):
        # Sort population based on fitness scores (descending order)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]

        # Select the top n% as parents from the current generation
        num_parents = max(2, int(self.population_size * (self.top_n_percent / 100)))
        parents = sorted_population[:num_parents]

        # Add the elite agents to the parent pool
        if self.elite_agents:
            parents.extend(self.elite_agents)

        # Generate the next generation
        next_generation = []

        # Elitism: keep the best parent as is
        next_generation.append(parents[0])

        while len(next_generation) < self.population_size:
            # Randomly select two parents for crossover
            parent1, parent2 = random.sample(parents, 2)
            child = NeuralNetwork.crossover(parent1, parent2)
            child.mutate(self.mutation_rate)
            next_generation.append(child)

        # Replace old population with the new generation
        self.population = next_generation

        # Keep the top elite_size agents from this generation
        self.elite_agents = sorted_population[:self.elite_size]

    def get_last_agent(self):
        return self.population[-1]

def encode_state(state):
    direction = state['direction']
    dir_one_hot = [0, 0, 0, 0]
    if direction == (0, -1):
        dir_one_hot[0] = 1
    elif direction == (0, 1):
        dir_one_hot[1] = 1
    elif direction == (-1, 0):
        dir_one_hot[2] = 1
    elif direction == (1, 0):
        dir_one_hot[3] = 1

    head_x, head_y = state['snake_head']
    food_x, food_y = state['food']
    food_dir = [0, 0, 0, 0]
    if food_x < head_x:
        food_dir[2] = 1
    elif food_x > head_x:
        food_dir[3] = 1
    if food_y < head_y:
        food_dir[0] = 1
    elif food_y > head_y:
        food_dir[1] = 1

    # Distance to walls
    dist_up = head_y
    dist_down = state['height'] - head_y
    dist_left = head_x
    dist_right = state['width'] - head_x

    distance_features = [dist_up / state['height'], dist_down / state['height'], dist_left / state['width'], dist_right / state['width']]

    # Obstacle detection (front, left, right)
    front_dir = direction
    left_dir = (-direction[1], direction[0])
    right_dir = (direction[1], -direction[0])

    front_cell = (head_x + front_dir[0], head_y + front_dir[1])
    left_cell = (head_x + left_dir[0], head_y + left_dir[1])
    right_cell = (head_x + right_dir[0], head_y + right_dir[1])

    obstacles = [
        int(front_cell in state['snake'] or front_cell[0] < 0 or front_cell[0] >= state['width'] or front_cell[1] < 0 or front_cell[1] >= state['height']),
        int(left_cell in state['snake'] or left_cell[0] < 0 or left_cell[0] >= state['width'] or left_cell[1] < 0 or left_cell[1] >= state['height']),
        int(right_cell in state['snake'] or right_cell[0] < 0 or right_cell[0] >= state['width'] or right_cell[1] < 0 or right_cell[1] >= state['height']),
    ]

    score = state['score'] / (state['width'] * state['height'])
    moves_left = state['moves_left'] / 1000

    feature_vector = dir_one_hot + food_dir + distance_features + obstacles + [score, moves_left]
    return np.array(feature_vector, dtype=np.float32)

def choose_action(network, state_vector):
    output = network.forward(state_vector)
    action = np.argmax(output)
    return ACTIONS[action]

def calculate_fitness(game):
    head_x, head_y = game.snake[0]
    food_x, food_y = game.food
    distance_to_fruit = abs(head_x - food_x) + abs(head_y - food_y)
    survival_bonus = game.total_moves / 100  # Reward for staying alive longer
    fitness = game.score #- (distance_to_fruit / 10)
    return fitness

