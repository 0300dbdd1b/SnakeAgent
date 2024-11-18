
import neat
import numpy as np
import pygame
from renderer import Colors

class GameStats:
    def __init__(self):
        self.best_score = 0
        self.best_genome = None
        self.best_generation = 0
        self.best_genome_history = []


def eval_genomes(genomes, config, game_class, game_stats, renderer=None):
    """ Evaluate genomes, compute fitness for each generation, and render the best genome.
    """
    generation_best_score = 0
    generation_best_genome = None

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = game_class()
        current_state = game.reset()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        renderer.toggle_pause()
                    elif event.key == pygame.K_UP:
                        renderer.speed = min(300, renderer.speed + 5)
                    elif event.key == pygame.K_DOWN:
                        renderer.speed = max(1, renderer.speed - 5)

            # Pause handling
            if renderer.paused:
                renderer.handle_pause()
                continue

            final_move = predict_action(current_state, net)
            moves_total, done, score, new_state = game.step(final_move)
            current_state = new_state

            if done:
                genome.fitness = 100 * (score * score) + (moves_total * 0.01)
                if score > generation_best_score:
                    generation_best_score = score
                    generation_best_genome = genome
                break

    # Render the best genome of this generation
    if renderer:
        render_game(config, generation_best_genome, game_class, renderer, "")

    # Check for new high score
    if generation_best_score > game_stats.best_score:
        game_stats.best_score = generation_best_score
        game_stats.best_genome = generation_best_genome
        game_stats.best_generation += 1
        render_game(config, generation_best_genome, game_class, renderer, "New High Score!")



def render_game(config, genome, game_class, renderer, message):


    if genome is None or not hasattr(genome, 'connections'):
        print("Error: Invalid genome, skipping rendering.")
        return

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = game_class()
    current_state = game.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    renderer.toggle_pause()
                elif event.key == pygame.K_UP:
                    renderer.speed = min(300, renderer.speed + 5)
                elif event.key == pygame.K_DOWN:
                    renderer.speed = max(1, renderer.speed - 5)

        if renderer.paused:
            renderer.handle_pause()
            continue

        inputs = game.get_state()
        final_move = predict_action(inputs, net)
        reward, done, score, new_state = game.step(final_move)

        renderer.render(game, genome, config, net)

        # Display the message on screen
        message_text = renderer.font.render(message, True, Colors.WHITE)
        message_pos = (renderer.display.get_width() // 2 - message_text.get_width() // 2,
                       renderer.display.get_height() // 2 - 50)
        renderer.display.blit(message_text, message_pos)
        pygame.display.flip()

        if done:
            break


def predict_action(state, net):
    output = net.activate(state)
    
    # Apply softmax to get probability distribution
    exp_output = np.exp(output - np.max(output))  # Subtract max for numerical stability
    softmax_output = exp_output / exp_output.sum()
    
    # Create one-hot encoded move vector
    move = [0, 0, 0]
    move[np.argmax(softmax_output)] = 1
    return move

def predict_action2(state, net):
    output = net.activate(state)

    if all(o == 0 for o in output):
        return [0, 0, 0]  # Invalid move to simulate a crash

    move = [0, 0, 0]
    move[np.argmax(output)] = 1
    return move


