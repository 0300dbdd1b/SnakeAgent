import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import neat
import pygame
from GameEngine import GameEngine
from GameRenderer import GameRenderer
from NEAT import fitness_function, get_inputs_from_gamestate

# Path to NEAT configuration file
CONFIG_PATH = "config-feedforward.txt"

def render_best_game(genome, config):
    game_engine = GameEngine()
    renderer = GameRenderer(width=game_engine.width, height=game_engine.height)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    done = False
    clock = pygame.time.Clock()

    while not done:
        gamestate = game_engine.get_state()
        inputs = get_inputs_from_gamestate(gamestate)
        action = net.activate(inputs)
        direction = ['forward', 'left', 'right'][np.argmax(action)]
        done = not game_engine.MoveRelative(direction)
        renderer.render_gamestate(gamestate)
        clock.tick(10)

    renderer.close()

def eval_genome_task(genome, config):
    fitness_function(genome, config)  # Runs the fitness function
    return genome.fitness

def eval_genomes(genomes, config):
    global high_score
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(eval_genome_task, genome, config): (genome_id, genome) for genome_id, genome in genomes}
        for future in as_completed(futures):
            genome_id, genome = futures[future]
            try:
                fitness = future.result()
                # print(f"fitness : {fitness}")
                if fitness > high_score:
                    print(f"New high score achieved: {fitness}")
                    high_score = fitness
                    render_best_game(genome, config)

            except Exception as e:
                print(f"Error evaluating genome {genome_id}: {e}")

    gc.collect()

def run_neat():
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                CONFIG_PATH)
    
    # Create a population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT and find the best agent
    winner = population.run(eval_genomes, 10000)
    print(f'\nBest genome:\n{winner}')

if __name__ == "__main__":
    high_score = 0  # Initialize the high score to zero
    run_neat()
