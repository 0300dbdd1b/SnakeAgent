import neat
import pygame
import neat
import numpy as np

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
        self.small_font = pygame.font.Font(None, 18)
        self.speed = default_speed
        self.node_positions = {}
        self.current_genome_nodes = set()
        self.paused = False

    def toggle_pause(self):
        self.paused = not self.paused

    def handle_pause(self):
        if self.paused:
            pause_text = self.font.render("Paused", True, Colors.WHITE)
            pause_pos = (self.display.get_width() // 2 - pause_text.get_width() // 2,
                         self.display.get_height() // 2 - pause_text.get_height() // 2)
            self.display.blit(pause_text, pause_pos)
            pygame.display.flip()
            pygame.time.wait(100)  # Avoid busy-waiting


    def calculate_node_positions(self, genome, config, x_offset):
        self.node_positions.clear()

        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        hidden_nodes = [key for key in genome.nodes.keys()
                        if key not in input_nodes and key not in output_nodes]

        self.current_genome_nodes = set(input_nodes + output_nodes + hidden_nodes)

        layer_spacing = self.viz_width // 4

        num_inputs = len(input_nodes)
        input_spacing = self.height // (num_inputs + 1)
        for i, node_id in enumerate(input_nodes):
            self.node_positions[node_id] = (
                x_offset + layer_spacing,
                (i + 1) * input_spacing
            )

        if hidden_nodes:
            hidden_spacing = self.height // (len(hidden_nodes) + 1)
            for i, node_id in enumerate(hidden_nodes):
                self.node_positions[node_id] = (
                    x_offset + 2 * layer_spacing,
                    (i + 1) * hidden_spacing
                )

        num_outputs = len(output_nodes)
        output_spacing = self.height // (num_outputs + 1)
        for i, node_id in enumerate(output_nodes):
            self.node_positions[node_id] = (
                x_offset + 3 * layer_spacing,
                (i + 1) * output_spacing
            )




    def draw_neat_network(self, genome, config, x_offset, inputs, outputs):
        if not genome:
            return

        genome_nodes = set(genome.nodes.keys())
        if genome_nodes != self.current_genome_nodes:
            self.calculate_node_positions(genome, config, x_offset)
        exp_output = np.exp(outputs - np.max(outputs))
        softmax_probs = exp_output / exp_output.sum()
        outputs = softmax_probs

        # Hardcoded input and output labels
        input_labels = [
            "Dist Ahead",
            "Dist Left",
            "Dist Right",
            "Dist Food",
            "Angle Food",
            "Score",
            "Direction Down",
            "Food Left",
            "Food Right",
            "Food Up",
            "Danger Ahead",
            "Danger Left",
            "Danger Right",
            "Direction Left",
            "Direction Right",
            "Direction Up",
            "Direction Down",
            "Food Left",
            "Food Right",
            "Food Up",
            "Food Down",
            "Food Down",
            "Danger Ahead"
        ]
        output_labels = ["Forward", "Right", "Left"]

        # Draw connections
        for conn_key, conn in genome.connections.items():
            if conn.enabled:
                input_node, output_node = conn_key
                if input_node in self.node_positions and output_node in self.node_positions:
                    input_pos = self.node_positions[input_node]
                    output_pos = self.node_positions[output_node]
                    weight = conn.weight
                    color = Colors.GREEN if weight > 0 else Colors.RED
                    thickness = max(1, min(3, int(abs(weight) * 3)))
                    pygame.draw.line(self.display, color, input_pos, output_pos, thickness)

        # Identify the index of the output node with the highest value
        max_output_idx = np.argmax(outputs) if outputs.size > 0 else -1

        # Draw nodes with labels and values
        node_radius = 15
        for node_id, pos in self.node_positions.items():
            if node_id in config.genome_config.input_keys:
                node_value = inputs[config.genome_config.input_keys.index(node_id)]
                label_text = input_labels[config.genome_config.input_keys.index(node_id)]
                label_pos = (pos[0] - 90, pos[1] - 10)  # Position labels to the left of the input nodes
                color = (255 - int(255 * max(0, min(1, node_value))), int(255 * max(0, min(1, node_value))), 0)
            elif node_id in config.genome_config.output_keys:
                node_value = outputs[config.genome_config.output_keys.index(node_id)]
                node_index = config.genome_config.output_keys.index(node_id)
                color = Colors.YELLOW if node_index == max_output_idx else (255 - int(255 * max(0, min(1, node_value))), int(255 * max(0, min(1, node_value))), 0)
                label_text = output_labels[node_index]
                label_pos = (pos[0] + 15, pos[1] - 10)  # Position labels to the right of the output nodes
            else:
                node_value = 0  # Hidden nodes
                intensity = int(255 * max(0, min(1, node_value)))
                color = (255 - intensity, intensity, 0)
                label_text = ""
                label_pos = pos  # No label for hidden nodes

            # Draw the node
            pygame.draw.circle(self.display, color, pos, node_radius)
            pygame.draw.circle(self.display, Colors.WHITE, pos, node_radius, 1)

            # Display the node value
            value_text = f"{node_value:.2f}"
            value_surface = self.small_font.render(value_text, True, Colors.WHITE)
            value_pos = (pos[0] - value_surface.get_width() // 2, pos[1] - value_surface.get_height() // 2)
            self.display.blit(value_surface, value_pos)

            # Display the label
            if label_text:
                label_surface = self.small_font.render(label_text, True, Colors.WHITE)
                self.display.blit(label_surface, label_pos)

    def render(self, game_state, genome=None, config=None, net=None):
        self.display.fill(Colors.BLACK)

        for pt in game_state.snake:
            pygame.draw.rect(self.display, Colors.BLUE,
                             pygame.Rect(pt[0], pt[1], game_state.block_size, game_state.block_size))
            pygame.draw.rect(self.display, Colors.WHITE,
                             pygame.Rect(pt[0] + 4, pt[1] + 4, 12, 12))

        pygame.draw.rect(self.display, Colors.RED,
                         pygame.Rect(game_state.food[0], game_state.food[1],
                                     game_state.block_size, game_state.block_size))

        score_text = self.font.render(f"Score: {game_state.score}", True, Colors.WHITE)
        self.display.blit(score_text, (0, 0))

        moves_color = (Colors.GREEN if game_state.moves_left > 20
                       else Colors.ORANGE if game_state.moves_left > 10
                       else Colors.RED)
        moves_text = self.font.render(f"Moves: {game_state.moves_left}", True, moves_color)
        self.display.blit(moves_text, (0, 40))
        speed_text = self.font.render(f"Speed: {self.speed}", True, Colors.WHITE)
        self.display.blit(speed_text, (0, 80))

        pygame.draw.line(self.display, Colors.WHITE, (self.game_width, 0), (self.game_width, self.height))

        inputs = game_state.get_state()
        if net:
            outputs = net.activate(inputs)
            self.draw_neat_network(genome, config, self.game_width, inputs, outputs)

        pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self):
        pygame.quit()



