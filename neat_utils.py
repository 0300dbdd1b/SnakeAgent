import neat
import numpy as np

class GameStats:
    def __init__(self):
        self.best_score = 0
        self.best_genome = None
        self.best_generation = 0
        self.best_genome_history = []

def eval_genomes(genomes, config, game_class, game_stats, renderer=None):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Determine if we should display this game
        display_this_game = False
        if len(game_stats.best_genome_history) > 0:
            last_best = game_stats.best_genome_history[-1][1]
            if last_best.fitness is not None and genome.fitness is not None:
                if genome.fitness >= last_best.fitness:
                    display_this_game = True
        game = game_class()
        current_state = game.reset()
        current_score = 0
        while True:
            final_move = predict_action(current_state, net)
            reward, done, score, new_state = game.step(final_move)
            current_score = score
            current_state = new_state
            if renderer and display_this_game:
                renderer.render(game, genome, config)
            if done:
                genome.fitness = current_score
                if current_score > game_stats.best_score:
                    game_stats.best_score = current_score
                    game_stats.best_genome = genome
                    game_stats.best_generation = len(game_stats.best_genome_history)
                    game_stats.best_genome_history.append((current_score, genome))
                    print(f"\nNew best score: {current_score}")
                    print(f"Generation: {game_stats.best_generation}")
                    if renderer:
                        replay_best_genome(config, genome, game_class, renderer)
                break

def predict_action(state, net):
    output = net.activate(state)
    move = [0, 0, 0]
    move[np.argmax(output)] = 1
    return move

def replay_best_genome(config, genome, game_class, renderer):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = game_class()
    current_state = game.reset()
    while True:
        final_move = predict_action(current_state, net)
        reward, done, score, new_state = game.step(final_move)
        current_state = new_state
        renderer.render(game, genome, config)
        if done:
            break
