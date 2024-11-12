
import random

class GameEngine:
    def __init__(self, width=20, height=20, initial_moves=20, move_increment=20):
        self.width = width
        self.height = height
        self.initial_moves = initial_moves
        self.move_increment = move_increment
        self.total_moves = 0
        self.reset()

    def reset(self):
        # Initialize the game state
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (1, 0)  # Start by facing right
        self.food = (self.snake[0][0] + 1, self.snake[0][1] - 1)  # Fixed starting position for food
        self.score = 0
        self.moves_remaining = self.initial_moves

    def place_food(self):
        # Deterministically generate the next food position based on the current state
        x, y = self.food
        dx, dy = self.direction
        while True:
            x = (((x + dx) * self.total_moves) - dy) % self.width
            y = (((y + dy) * self.total_moves) - dx) % self.height
            new_food_position = (x, y)
            if new_food_position not in self.snake:
                return new_food_position

    def rotate_direction(self, action):
        # Define direction rotation based on the current direction
        if action == "left":
            self.direction = (-self.direction[1], self.direction[0])  # Rotate left 90 degrees
        elif action == "right":
            self.direction = (self.direction[1], -self.direction[0])  # Rotate right 90 degrees
        # "forward" keeps the current direction unchanged

    def MoveAbsolute(self, direction):
        # Map directions to tuples
        direction_map = {
            'UP': (0, -1),
            'RIGHT': (1, 0),
            'DOWN': (0, 1),
            'LEFT': (-1, 0)
        }

        # Prevent reversing direction
        if (direction_map[direction][0] == -self.direction[0] and
            direction_map[direction][1] == -self.direction[1]):
            return False  # Game over if moving in the opposite direction

        # Update direction
        self.direction = direction_map[direction]
        return self._move_snake()

    def MoveRelative(self, action):
        # Adjust direction based on relative action
        self.rotate_direction(action)
        return self._move_snake()

    def _move_snake(self):
        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collisions
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            return False  # Game over

        # Move the snake
        self.snake.insert(0, new_head)

        # Check if the snake has eaten food
        if new_head == self.food:
            self.score += 1
            self.moves_remaining += self.move_increment
            self.food = self.place_food()  # Generate new food position
        else:
            self.snake.pop()  # Remove the tail if no food eaten

        self.moves_remaining -= 1
        if self.moves_remaining <= 0:
            return False  # Game over if out of moves

        self.total_moves += 1
        return True  # Game continues

    def get_state(self):
        # Return the current game state for rendering or AI input
        return {
            'snake': self.snake,
            'food': self.food,
            'score': self.score,
            'moves_remaining': self.moves_remaining,
            'width': self.width,
            'height': self.height,
        }

