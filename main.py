import os
import neat
from game_logic import GameState
from renderer import GameRenderer
from neat_utils import GameStats, eval_genomes

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    # Initialize population and add reporters
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Initialize game statistics and renderer
    game_stats = GameStats()
    renderer = GameRenderer()
    # Run evolution
    winner = p.run(lambda genomes, config: eval_genomes(genomes, config, GameState, game_stats, renderer), 1000)
    print(f"\nBest Score Achieved: {game_stats.best_score}")
    print(f"Best Generation: {game_stats.best_generation}")
    renderer.close()
    return winner

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)
