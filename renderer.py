import pygame
import numpy as np
from neat.graphs import feed_forward_layers

class Colors:
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    ORANGE = (255, 165, 0)
    GRAY = (128, 128, 128)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)

class GameRenderer:
    def __init__(self, width=800, height=400, default_speed=15):
        pygame.init()
        self.game_width = width // 2
        self.viz_width = width // 2
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake AI with NEAT Visualization')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.speed = default_speed
        self.node_positions = {}
        self.current_genome_nodes = set()  # Track current genome's nodes
        
    def calculate_node_positions(self, genome, config, x_offset):
        self.node_positions.clear()
        
        # Get all nodes
        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        hidden_nodes = [key for key in genome.nodes.keys() 
                       if key not in input_nodes and key not in output_nodes]
        
        # Update current genome nodes set
        self.current_genome_nodes = set(input_nodes + output_nodes + hidden_nodes)
        
        # Calculate layer positions
        layer_spacing = self.viz_width // 4
        
        # Position input nodes
        num_inputs = len(input_nodes)
        input_spacing = self.height // (num_inputs + 1)
        for i, node_id in enumerate(input_nodes):
            self.node_positions[node_id] = (
                x_offset + layer_spacing,
                (i + 1) * input_spacing
            )
        
        # Position hidden nodes
        if hidden_nodes:
            hidden_spacing = self.height // (len(hidden_nodes) + 1)
            for i, node_id in enumerate(hidden_nodes):
                self.node_positions[node_id] = (
                    x_offset + 2 * layer_spacing,
                    (i + 1) * hidden_spacing
                )
        
        # Position output nodes
        num_outputs = len(output_nodes)
        output_spacing = self.height // (num_outputs + 1)
        for i, node_id in enumerate(output_nodes):
            self.node_positions[node_id] = (
                x_offset + 3 * layer_spacing,
                (i + 1) * output_spacing
            )

    def draw_neat_network(self, genome, config, x_offset):
        if not genome:
            return
        # Check if we need to recalculate positions
        genome_nodes = set(genome.nodes.keys())
        if genome_nodes != self.current_genome_nodes:
            self.calculate_node_positions(genome, config, x_offset)
        # Draw connections
        for conn_key, conn in genome.connections.items():
            if conn.enabled:
                input_node, output_node = conn_key
                if input_node in self.node_positions and output_node in self.node_positions:
                    input_pos = self.node_positions[input_node]
                    output_pos = self.node_positions[output_node]
                    # Calculate connection color based on weight
                    weight = conn.weight
                    if weight > 0:
                        color = tuple(map(lambda x: int(x * min(abs(weight), 1)), Colors.GREEN))
                    else:
                        color = tuple(map(lambda x: int(x * min(abs(weight), 1)), Colors.RED))
                    # Draw connection line with weight-based thickness
                    thickness = max(1, min(3, abs(int(weight * 3))))
                    pygame.draw.line(self.display, color, input_pos, output_pos, thickness)
        # Draw nodes
        node_radius = 10
        for node_id, pos in self.node_positions.items():
            # Determine node color and label based on type
            if node_id in config.genome_config.input_keys:
                color = Colors.BLUE
                label = f"I{node_id}"
            elif node_id in config.genome_config.output_keys:
                color = Colors.RED
                label = f"O{node_id}"
            else:
                color = Colors.YELLOW
                label = f"H{node_id}"
            
            # Draw node
            pygame.draw.circle(self.display, color, pos, node_radius)
            pygame.draw.circle(self.display, Colors.WHITE, pos, node_radius, 1)
            
            # Draw node label
            label_surface = self.small_font.render(label, True, Colors.WHITE)
            label_pos = (pos[0] - label_surface.get_width()//2, 
                        pos[1] - label_surface.get_height()//2)
            self.display.blit(label_surface, label_pos)
        
        # Draw fitness if available
        if hasattr(genome, 'fitness') and genome.fitness is not None:
            fitness_text = self.font.render(f"Fitness: {genome.fitness:.2f}", True, Colors.WHITE)
            self.display.blit(fitness_text, (x_offset + 10, 10))

    def render(self, game_state, genome=None, config=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.speed = min(300, self.speed + 5)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 5)
        
        self.display.fill(Colors.BLACK)
        
        # Draw game
        for pt in game_state.snake:
            pygame.draw.rect(self.display, Colors.BLUE, 
                           pygame.Rect(pt[0], pt[1], game_state.block_size, game_state.block_size))
            pygame.draw.rect(self.display, Colors.WHITE, 
                           pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))
        
        pygame.draw.rect(self.display, Colors.RED, 
                        pygame.Rect(game_state.food[0], game_state.food[1], 
                                  game_state.block_size, game_state.block_size))
        
        # Display stats
        score_text = self.font.render(f"Score: {game_state.score}", True, Colors.WHITE)
        self.display.blit(score_text, (0, 0))
        
        moves_color = (Colors.GREEN if game_state.moves_left > 20 
                      else Colors.ORANGE if game_state.moves_left > 10 
                      else Colors.RED)
        moves_text = self.font.render(f"Moves: {game_state.moves_left}", True, moves_color)
        self.display.blit(moves_text, (0, 40))
        speed_text = self.font.render(f"Speed: {self.speed}", True, Colors.WHITE)
        self.display.blit(speed_text, (0, 80))
        # Draw divider
        pygame.draw.line(self.display, Colors.WHITE, (self.game_width, 0), (self.game_width, self.height))

        self.draw_neat_network(genome, config, self.game_width)
        pygame.display.flip()
        self.clock.tick(self.speed)
    
    def close(self):
        pygame.quit()
