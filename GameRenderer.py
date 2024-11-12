
import pygame
import numpy as np

class GameRenderer:
    def __init__(self, width=20, height=20, cell_size=20):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Extend screen width for stats area with brain visualization
        self.stats_area_width = 200
        self.screen_width = width * cell_size + self.stats_area_width
        self.screen_height = height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("NEAT Snake Game")

    def render_gamestate(self, gamestate):
        # Fill the gameplay area and stats area with different colors
        self.screen.fill((30, 30, 30))  # Gameplay area background (dark grey)
        pygame.draw.rect(
            self.screen,
            (50, 50, 50),  # Stats area background (slightly lighter grey)
            (self.width * self.cell_size, 0, self.stats_area_width, self.screen_height)
        )

        # Draw the grid for gameplay area
        for x in range(0, self.width * self.cell_size, self.cell_size):
            for y in range(0, self.height * self.cell_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (40, 40, 40), rect, 1)  # Light grey grid lines

        # Draw the food
        food_x, food_y = gamestate['food']
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (food_x * self.cell_size, food_y * self.cell_size, self.cell_size, self.cell_size)
        )

        # Draw the snake
        for x, y in gamestate['snake']:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            )

        # Display the score and remaining moves in the stats area
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {gamestate['score']}", True, (255, 255, 255))
        moves_text = font.render(f"Moves: {gamestate['moves_remaining']}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.width * self.cell_size + 10, 10))
        self.screen.blit(moves_text, (self.width * self.cell_size + 10, 50))

        pygame.display.flip()



    def render_brain(self, genome, config):
        """
        Render the neural network (brain) visualization in the stats area based on the genome and config.
        """
        # Clear only the stats area
        pygame.draw.rect(
            self.screen,
            (50, 50, 50),  # Stats area background color
            (self.width * self.cell_size, 0, self.stats_area_width, self.screen_height)
        )
        
        # Define positions for input, hidden, and output nodes
        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        hidden_nodes = [n for n in genome.nodes.keys() if n not in input_nodes and n not in output_nodes]
        
        # Layout parameters for node positioning
        node_radius = 10
        layer_spacing = 60
        node_spacing = 30
        
        # Position nodes
        node_positions = {}
        for i, node in enumerate(input_nodes):
            x = self.width * self.cell_size + 20  # Offset into stats area
            y = 100 + i * node_spacing
            node_positions[node] = (x, y)
        
        for i, node in enumerate(hidden_nodes):
            x = self.width * self.cell_size + 20 + layer_spacing
            y = 100 + i * node_spacing
            node_positions[node] = (x, y)
        
        for i, node in enumerate(output_nodes):
            x = self.width * self.cell_size + 20 + 2 * layer_spacing
            y = 100 + i * node_spacing
            node_positions[node] = (x, y)
        
        # Draw nodes
        for node, (x, y) in node_positions.items():
            # Activation (dummy value for now, adjust as needed based on genome)
            activation = genome.nodes[node].activation if node in genome.nodes else 0.5
            color = (int(activation * 255), 0, 255 - int(activation * 255))
            pygame.draw.circle(self.screen, color, (x, y), node_radius)
        
        # Draw connections
        for connection in genome.connections.values():
            if connection.enabled:
                from_pos = node_positions.get(connection.key[0])
                to_pos = node_positions.get(connection.key[1])
                if from_pos and to_pos:
                    weight = connection.weight
                    color = (0, int(weight * 255) if weight > 0 else 0, 0)
                    pygame.draw.line(self.screen, color, from_pos, to_pos, 2)
    
    def close(self):
        pygame.quit()

