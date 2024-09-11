from Game import Game
from GeneticAlgorithm import *
import numpy as np
import random
import matplotlib.pyplot as plt




def main():
    # Hyperparameters
    GENERATIONS = 300
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.2
    MAX_MOVES = 200

    layer_sizes = [17, 8, 3]  # Example of a neural network with more features
    ga = GeneticAlgorithm(population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE, layer_sizes=layer_sizes, elite_size=7, top_n_percent=10)

    best_global_score = 0  # Track the best score across all generations
    game_renderer = Game(width=20, height=20, max_moves=MAX_MOVES, render=True)
    avg_scores = []  # List to track the average score of each generation
    best_scores = []  # List to track the best score of each generation

    for generation in range(GENERATIONS):
        print(f"=== Generation {generation + 1} ===")
        scores = []
        best_generation_score = 0  # Track the best score for this generation
        best_agent_states_actions = []  # Store the full state-action sequence for the best agent

        for idx, network in enumerate(ga.population):
            game = Game(width=20, height=20, max_moves=MAX_MOVES, render=False)

            # Store the full state-action sequence of the agent
            states_actions_taken = []

            while not game.game_over:
                state = game.get_state()  # Capture the state at each step
                state_vector = encode_state(state)
                action_str = choose_action(network, state_vector)

                current_dir = game.direction
                if action_str == 'STRAIGHT':
                    new_direction = current_dir
                elif action_str == 'RIGHT':
                    new_direction = (current_dir[1], -current_dir[0])
                elif action_str == 'LEFT':
                    new_direction = (-current_dir[1], current_dir[0])
                else:
                    new_direction = current_dir

                game.change_direction(new_direction)
                game.update()

                # Store the full state and action
                states_actions_taken.append((game.get_state(), action_str))

            # Calculate fitness
            fitness = calculate_fitness(game)
            scores.append(fitness)

            # Track the best score and agent of this generation
            if game.score > best_generation_score:
                best_generation_score = game.score
                best_agent_states_actions = states_actions_taken  # Store the full sequence of the best agent

            # Print Agent score, Generation best score, and Global best score
            print(f"Agent [{generation + 1}/{idx + 1}] Score: {game.score} - Best Generation Score: {best_generation_score} - Best Global Score: {best_global_score}")

        # Calculate and append the average score for this generation
        avg_score = sum(scores) / len(scores)
        avg_scores.append(avg_score)
        best_scores.append(best_generation_score)
        print(f"Average score for Generation {generation + 1}: {avg_score}")
        print(f"Best score for Generation {generation + 1}: {best_generation_score}")
        
        # Update global best score if this generation has a better score
        if best_generation_score > best_global_score:
            best_global_score = best_generation_score
            print(f"New global best score: {best_global_score}")

            # Only render the best agent's game if the best score sets a new global record
            game_renderer.reset()  # Reset the renderer game state for replay

            # Replay loop for the best agent
            for state, action_str in best_agent_states_actions:
                # Set the game renderer state to the stored state
                game_renderer.snake = state['snake']
                game_renderer.direction = state['direction']
                game_renderer.food = state['food']
                game_renderer.moves_left = state['moves_left']
                game_renderer.total_moves = state['total_moves']
                game_renderer.score = state['score']
                game_renderer.game_over = state['game_over']

                # Render the state
                game_renderer.render()

                # Display generation number on the screen
                game_renderer.display_text(f"Generation: {generation + 1}")

        # Evolve the population based on scores
        ga.evolve(scores)

    # Plot the evolution of average and best scores over generations
    plt.plot(avg_scores, label='Average Score')
    plt.plot(best_scores, label='Best Score')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Evolution of Scores over Generations')
    plt.legend()
    plt.show()

    # Print the global best score after all generations are done
    print(f"Global best score across all generations: {best_global_score}")
    game_renderer.close()

if __name__ == "__main__":
    main()
