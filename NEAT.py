
import neat
import numpy as np
from GameEngine import GameEngine

# Configuration Constants
VISION_RAYS = [-60, 0, 60]



def compute_obstacle_distances(gamestate):
    head_x, head_y = gamestate['snake'][0]
    width, height = gamestate['width'], gamestate['height']
    snake_body = set(gamestate['snake'][1:])  # Exclude the head
    max_distance = max(width, height)  # Max possible distance for normalization

    # Define directions for up, down, left, right as (dx, dy) pairs
    directions = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }

    # Compute distances for each direction
    distances = {}
    for direction, (dx, dy) in directions.items():
        distance = 0
        current_x, current_y = head_x, head_y

        # Move in the specified direction until an obstacle or wall is reached
        while 0 <= current_x < width and 0 <= current_y < height:
            # Check if current position is part of the snake's body
            if (int(current_x), int(current_y)) in snake_body:
                break
            # Move one step in the direction
            current_x += dx
            current_y += dy
            distance += 1

        # Normalize the distance
        normalized_distance = min(distance / max_distance, 1.0)
        distances[direction] = normalized_distance

    return distances['up'], distances['down'], distances['left'], distances['right']



def get_inputs_from_gamestate(gamestate):
    head_x, head_y = gamestate['snake'][0]
    food_x, food_y = gamestate['food']
    tail_x, tail_y = gamestate['snake'][-1]
    width, height = gamestate['width'], gamestate['height']

    # Get obstacle distances in up, down, left, and right directions
    up_distance, down_distance, left_distance, right_distance = compute_obstacle_distances(gamestate)

    # X and Y distances to food, normalized from 0 to 1
    food_distance_x = min(abs(food_x - head_x) / width, 1.0)
    food_distance_y = min(abs(food_y - head_y) / height, 1.0)

    # X and Y distances to tail, normalized from 0 to 1
    tail_distance_x = min(abs(tail_x - head_x) / width, 1.0)
    tail_distance_y = min(abs(tail_y - head_y) / height, 1.0)

    # Combine all inputs into a single normalized vector
    inputs = [
        up_distance, down_distance, left_distance, right_distance,
        food_distance_x, food_distance_y, tail_distance_x, tail_distance_y
    ]
    return inputs


def fitness_function(genome, config):
    game_engine = GameEngine()  # Create a new game instance
    net = neat.nn.FeedForwardNetwork.create(genome, config)  # Build network for the genome
    fitness = 0
    done = False

    while not done:
        gamestate = game_engine.get_state()
        inputs = get_inputs_from_gamestate(gamestate)
        action = net.activate(inputs)
        direction = np.argmax(action)
        actions = ['forward', 'left', 'right']
        done = not game_engine.MoveRelative(actions[direction])

    gamestate = game_engine.get_state()
    head_x, head_y = gamestate['snake'][0]
    food_x, food_y = gamestate['food']
    food_distance = np.linalg.norm([food_x - head_x, food_y - head_y]) / max(gamestate['width'], gamestate['height'])

    fitness += (game_engine.score * 100) + (game_engine.total_moves * 0.01)
    genome.fitness = fitness  # Assign the computed fitness to the genome

