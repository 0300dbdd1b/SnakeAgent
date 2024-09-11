
# Game.py
import pygame
import random

FPS = 60
class Game:
    def __init__(self, width=20, height=20, max_moves=200, render=True):
        self.width = width
        self.height = height
        self.block_size = 20
        self.max_moves = max_moves
        self.rendering = render
        self.snake = [(width // 2, height // 2)]  # Start with the snake in the middle of the grid
        self.direction = (0, -1)  # Snake starts moving up
        self.food = self.generate_food()
        self.game_over = False
        self.moves_left = max_moves
        self.total_moves = 0  # Track the total number of moves
        self.score = 0

        if self.rendering:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size + 50))  # Extra space for the HUD
            pygame.display.set_caption('Snake AI Game')
            self.font = pygame.font.SysFont(None, 24)
            self.clock = pygame.time.Clock()  # Initialize a clock for frame rate control

    def reset(self):
        """Reset the game state to start a new game."""
        self.snake = [(self.width // 2, self.height // 2)]  # Reset snake to starting position
        self.direction = (0, -1)  # Reset direction
        self.food = self.generate_food()  # Generate new food
        self.game_over = False
        self.moves_left = self.max_moves  # Reset moves
        self.total_moves = 0
        self.score = 0

    def generate_food(self):
        """Generate food in a random position not occupied by the snake."""
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def change_direction(self, new_direction):
        """Change snake direction if the new direction is not the opposite."""
        if (new_direction[0], new_direction[1]) != (-self.direction[0], -self.direction[1]):
            self.direction = new_direction

    def update(self):
        """Update the game state, move the snake, and check for collisions."""
        if self.game_over:
            return

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collisions (with walls or itself)
        if (new_head in self.snake) or (new_head[0] < 0 or new_head[0] >= self.width) or (new_head[1] < 0 or new_head[1] >= self.height):
            self.game_over = True
            return

        # Move the snake
        self.snake = [new_head] + self.snake[:-1]

        # Check if snake eats the food
        if new_head == self.food:
            self.snake.append(self.snake[-1])  # Grow the snake
            self.food = self.generate_food()
            self.score += 1
            self.moves_left = 200  # Reward some moves for eating food

        # Update moves left
        self.moves_left -= 1
        self.total_moves += 1

        # Check if moves are depleted
        if self.moves_left <= 0:
            self.game_over = True

    def get_state(self):
        """Return the full game state, with as much detail as possible."""
        state = {
            'snake': self.snake,                # Snake's body (list of (x, y) coordinates)
            'snake_head': self.snake[0],        # The head of the snake
            'food': self.food,                  # Current food position
            'direction': self.direction,        # Current direction of the snake
            'moves_left': self.moves_left,      # Remaining moves before game over
            'total_moves': self.total_moves,    # Total moves made by the snake
            'score': self.score,                # Current score (food eaten)
            'game_over': self.game_over,        # Whether the game is over
            'width': self.width,                # Grid width
            'height': self.height,              # Grid height
        }
        return state

    def render(self):
        """Render the game onto the screen with additional HUD."""
        if not self.rendering:
            return  # Skip rendering if not enabled

        self.screen.fill((0, 0, 0))  # Clear the screen (black background)

        # Draw the snake
        for segment in self.snake:
            pygame.draw.rect(
                self.screen, (0, 255, 0),
                (segment[0] * self.block_size, segment[1] * self.block_size, self.block_size, self.block_size)
            )

        # Draw the food
        pygame.draw.rect(
            self.screen, (255, 0, 0),
            (self.food[0] * self.block_size, self.food[1] * self.block_size, self.block_size, self.block_size)
        )

        # Draw HUD (number of moves left and score)
        hud_text = f'Moves Left: {self.moves_left} | Total Moves: {self.total_moves} | Score: {self.score}'
        hud_surface = self.font.render(hud_text, True, (255, 255, 255))
        self.screen.blit(hud_surface, (10, self.height * self.block_size + 10))

        pygame.display.flip()
        #self.clock.tick(FPS) #NOTE: SET FPS BUGGY



    def display_text(self, text):
        font = pygame.font.SysFont(None, 48)
        text_surface = font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        pygame.display.update()
    
    def close(self):
        """Close the game and shut down Pygame."""
        if self.rendering:
            pygame.quit()
